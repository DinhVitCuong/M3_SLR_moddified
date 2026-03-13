import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from utils.misc import load_config
from utils.utils import load_model


def parse_args():
    parser = argparse.ArgumentParser("Real-time / single-video inference for M3-SLR UFOneView")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to repo config YAML, e.g. configs/Uniformer/test_cfg/UFOneView_MMAuslan.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained .pth checkpoint. This will override cfg['training']['pretrained_model']",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam id like 0, or a video file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda:0 / cpu. If omitted, use cfg device when available",
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=3,
        help="Run inference every N sampled frames after buffer is full",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=None,
        help="Sample 1 frame every N raw frames. Default = cfg['data']['temporal_stride']",
    )
    parser.add_argument(
        "--smooth-windows",
        type=int,
        default=5,
        help="Number of recent window logits to average for smoother prediction",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Top-k predictions to show",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default=None,
        help="Optional CSV mapping label id -> gloss text",
    )
    parser.add_argument(
        "--gt-label",
        type=int,
        default=None,
        help="Optional ground-truth label id for single video evaluation",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable cv2 window (useful for headless / server runs)",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Optional output video path with overlayed predictions",
    )
    
    # parser.add_argument("--window-size", type=int, default=None)
    # parser.add_argument("--infer-every", type=int, default=1)
    return parser.parse_args()


def resolve_device(cfg, device_arg):
    if device_arg is not None:
        return device_arg
    cfg_device = cfg.get("training", {}).get("device", "cuda:0")
    if "cuda" in cfg_device and not torch.cuda.is_available():
        return "cpu"
    return cfg_device


def load_label_map(csv_path):
    if csv_path is None:
        return {}

    df = pd.read_csv(csv_path)
    if df.empty:
        return {}

    # Flexible parsing:
    # 1) columns like [label, gloss]
    # 2) columns like [id, name]
    # 3) first col = id, second col = gloss
    cols_lower = [str(c).strip().lower() for c in df.columns]

    id_col = None
    text_col = None

    for cand in ["label", "label_id", "id", "class_id", "target"]:
        if cand in cols_lower:
            id_col = df.columns[cols_lower.index(cand)]
            break

    for cand in ["gloss", "name", "class_name", "text", "word"]:
        if cand in cols_lower:
            text_col = df.columns[cols_lower.index(cand)]
            break

    if id_col is None or text_col is None:
        if df.shape[1] >= 2:
            id_col = df.columns[0]
            text_col = df.columns[1]
        else:
            return {}

    mapping = {}
    for _, row in df.iterrows():
        try:
            k = int(row[id_col])
            v = str(row[text_col])
            mapping[k] = v
        except Exception:
            continue
    return mapping


def label_to_text(label_id, label_map):
    if label_map and label_id in label_map:
        return f"{label_id} ({label_map[label_id]})"
    return str(label_id)


def build_preprocess(cfg):
    img_size = int(cfg["data"]["vid_transform"]["IMAGE_SIZE"])
    mean = np.array(cfg["data"]["vid_transform"]["NORM_MEAN_IMGNET"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(cfg["data"]["vid_transform"]["NORM_STD_IMGNET"], dtype=np.float32).reshape(3, 1, 1)

    def preprocess(frame_bgr):
        # Repo transforms operate on RGB-like HWC arrays, then ToFloatTensor -> PermuteImage -> Normalize
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(frame_rgb.copy()).float()          # H W C
        tensor = tensor.permute(2, 0, 1).contiguous()               # C H W
        tensor = tensor / 255.0
        mean_t = torch.from_numpy(mean)
        std_t = torch.from_numpy(std)
        tensor = (tensor - mean_t) / std_t
        return tensor

    return preprocess


def prepare_model(cfg, device_str, checkpoint_override=None):
    cfg = dict(cfg)
    cfg["training"] = dict(cfg["training"])

    if checkpoint_override is not None:
        cfg["training"]["pretrained"] = True
        cfg["training"]["pretrained_model"] = checkpoint_override

    if cfg["training"].get("pretrained", False):
        ckpt = cfg["training"].get("pretrained_model", None)
        if ckpt in [None, "None", ""]:
            raise ValueError(
                "Config is in pretrained/test mode but pretrained_model is empty. "
                "Please pass --checkpoint path/to/model.pth"
            )

    cfg["training"]["device"] = device_str

    model = load_model(cfg)
    model = model.to(device_str)
    model.eval()
    return model, cfg


def make_clip_from_buffer(frame_buffer, device):
    # frame_buffer stores tensors [C,H,W]
    # repo collate expects stack -> [B,T,C,H,W] then permute -> [B,C,T,H,W]
    clip = torch.stack(list(frame_buffer), dim=0)   # [T,C,H,W]
    clip = clip.permute(1, 0, 2, 3).unsqueeze(0)    # [1,C,T,H,W]
    return clip.to(device, non_blocking=True)


@torch.inference_mode()
def infer_one_window(model, clip, use_amp=True):
    if use_amp and clip.device.type == "cuda":
        with torch.cuda.amp.autocast():
            outputs = model(clip=clip)
    else:
        outputs = model(clip=clip)

    if not isinstance(outputs, dict) or "logits" not in outputs:
        raise RuntimeError("Model output does not contain 'logits'.")
    logits = outputs["logits"]
    return logits


def format_topk(mean_logits, k, label_map):
    probs = torch.softmax(mean_logits, dim=1)
    top_probs, top_ids = torch.topk(probs, k=min(k, probs.shape[1]), dim=1)
    items = []
    for p, idx in zip(top_probs[0].tolist(), top_ids[0].tolist()):
        items.append(f"{label_to_text(int(idx), label_map)}: {p:.3f}")
    return " | ".join(items), int(top_ids[0, 0].item()), float(top_probs[0, 0].item())


def draw_overlay(frame, pred_text, conf, topk_text, fps_text):
    y = 30
    cv2.putText(frame, f"Pred: {pred_text}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y += 35
    cv2.putText(frame, f"Conf: {conf:.3f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    y += 35
    cv2.putText(frame, fps_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    # Wrap top-k roughly into multiple lines if too long
    y += 40
    chunks = [topk_text[i:i + 95] for i in range(0, len(topk_text), 95)]
    for chunk in chunks[:3]:
        cv2.putText(frame, chunk, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 28
    return frame


def open_source(source_str):
    if source_str.isdigit():
        return cv2.VideoCapture(int(source_str)), True
    return cv2.VideoCapture(source_str), False


def run_stream(args):
    cfg = load_config(args.config)
    device_str = resolve_device(cfg, args.device)

    model, cfg = prepare_model(cfg, device_str, checkpoint_override=args.checkpoint)
    preprocess = build_preprocess(cfg)
    label_map = load_label_map(args.label_map)

    window_size = int(cfg["data"]["num_output_frames"])
    sample_stride = int(args.sample_stride or cfg["data"].get("temporal_stride", 1))

    cap, is_webcam = open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    writer = None
    if args.save_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1e-6:
            fps = 25.0
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))

    frame_buffer = deque(maxlen=window_size)
    logits_buffer = deque(maxlen=max(1, args.smooth_windows))
    window_preds = []
    window_logits_all = []

    raw_frame_idx = 0
    sampled_frame_idx = 0
    infer_counter = 0

    prev_time = time.time()
    display_fps = 0.0

    last_pred_text = "warming up..."
    last_conf = 0.0
    last_topk = "waiting for enough frames..."
    print(f"[INFO] device         : {device_str}")
    print(f"[INFO] source         : {args.source}")
    print(f"[INFO] window_size    : {window_size}")
    print(f"[INFO] sample_stride  : {sample_stride}")
    print(f"[INFO] infer_every    : {args.infer_every}")
    print(f"[INFO] smooth_windows : {args.smooth_windows}")
    last_printed_pred = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        raw_frame_idx += 1

        # FPS estimate for display
        now = time.time()
        dt = max(now - prev_time, 1e-6)
        display_fps = 0.9 * display_fps + 0.1 * (1.0 / dt) if display_fps > 0 else (1.0 / dt)
        prev_time = now

        if (raw_frame_idx - 1) % sample_stride == 0:
            sampled_frame_idx += 1
            frame_tensor = preprocess(frame)
            frame_buffer.append(frame_tensor)

            if len(frame_buffer) == window_size:
                infer_counter += 1
                if (infer_counter - 1) % max(1, args.infer_every) == 0:
                    clip = make_clip_from_buffer(frame_buffer, device=device_str)
                    logits = infer_one_window(model, clip, use_amp=True).detach().float().cpu()
                    logits_buffer.append(logits)
                    window_logits_all.append(logits.squeeze(0))

                    mean_logits = torch.stack(list(logits_buffer), dim=0).mean(dim=0)
                    topk_text, pred_id, conf = format_topk(mean_logits, args.topk, label_map)

                    window_preds.append(pred_id)
                    print(f"[DEBUG] pred={label_to_text(pred_id, label_map)} | conf={conf:.3f} | topk={topk_text}")
                    if conf >=0.5:
                        last_pred_text = label_to_text(pred_id, label_map)
                        last_conf = conf
                        last_topk = topk_text
                    
        if last_pred_text != last_printed_pred:
            print(f"[REALTIME] pred={last_pred_text} | conf={last_conf:.3f} | topk={last_topk}")
            last_printed_pred = last_pred_text

        fps_text = f"FPS(display): {display_fps:.1f} | sampled={sampled_frame_idx} | windows={len(window_preds)}"
        frame_vis = frame.copy()
        # frame_vis = draw_overlay(frame_vis, fps_text)
        frame_vis = draw_overlay(frame_vis, last_pred_text, last_conf, last_topk, fps_text)

        if writer is not None:
            writer.write(frame_vis)

        if not args.no_display:
            cv2.imshow("M3-SLR UFOneView realtime inference", frame_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    if len(window_logits_all) == 0:
        print("[WARN] No prediction was produced. Usually this means the source is too short.")
        return

    final_mean_logits = torch.stack(window_logits_all, dim=0).mean(dim=0, keepdim=True)
    final_topk_text, final_pred_id, final_conf = format_topk(final_mean_logits, args.topk, label_map)

    vote_pred_id = Counter(window_preds).most_common(1)[0][0]
    vote_pred_text = label_to_text(vote_pred_id, label_map)
    mean_pred_text = label_to_text(final_pred_id, label_map)

    print("\n========== FINAL SUMMARY ==========")
    print(f"Majority-vote prediction : {vote_pred_text}")
    print(f"Mean-logits prediction   : {mean_pred_text}")
    print(f"Confidence (mean logits) : {final_conf:.4f}")
    print(f"Top-{args.topk}          : {final_topk_text}")
    print(f"Number of windows        : {len(window_preds)}")

    if args.gt_label is not None:
        gt = int(args.gt_label)
        vote_ok = int(vote_pred_id == gt)
        mean_ok = int(final_pred_id == gt)
        print(f"Ground-truth label       : {gt}")
        print(f"Vote correct?            : {vote_ok}")
        print(f"Mean-logits correct?     : {mean_ok}")


if __name__ == "__main__":
    args = parse_args()
    run_stream(args)