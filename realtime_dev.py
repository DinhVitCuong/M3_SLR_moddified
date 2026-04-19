import argparse
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.misc import load_config, load_label_map
from utils.utils import load_model
from utils.video_augmentation import Compose, Resize, ToFloatTensor, PermuteImage, Normalize
from slr_cropper import SLRCropper


def build_test_transform(cfg: dict):
    image_size = cfg["data"]["vid_transform"]["IMAGE_SIZE"]
    mean = cfg["data"]["vid_transform"]["NORM_MEAN_IMGNET"]
    std = cfg["data"]["vid_transform"]["NORM_STD_IMGNET"]
    return Compose(
        Resize(image_size),
        ToFloatTensor(),
        PermuteImage(),
        Normalize(mean, std),
    )


def create_model_from_config(config_path: str, checkpoint_path: str, device: str):
    cfg = load_config(config_path)
    cfg["training"]["test"] = True
    cfg["training"]["pretrained"] = True
    cfg["training"]["pretrained_model"] = checkpoint_path
    cfg["training"]["device"] = device
    cfg["training"]["batch_size"] = 1

    model = load_model(cfg)
    model = model.to(device)
    model.eval()
    return model, cfg


def open_camera_capture(camera_id: int, camera_url: Optional[str], backend: str):
    if camera_url:
        return cv2.VideoCapture(camera_url)
    if backend == "dshow":
        return cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if backend == "msmf":
        return cv2.VideoCapture(camera_id, cv2.CAP_MSMF)
    if backend == "default":
        return cv2.VideoCapture(camera_id)

    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    cap.release()

    cap = cv2.VideoCapture(camera_id, cv2.CAP_MSMF)
    if cap.isOpened():
        return cap
    cap.release()

    return cv2.VideoCapture(camera_id)


def open_input_capture(input_mode: str, camera_id: int, camera_url: Optional[str], backend: str, video_path: Optional[str]):
    if input_mode == "video":
        if not video_path:
            raise ValueError("input_mode=video nhưng chưa truyền --video-path")
        return cv2.VideoCapture(video_path)
    return open_camera_capture(camera_id, camera_url, backend)


def try_set_camera_fps(cap, fps_lock: float, input_mode: str):
    if input_mode != "webcam":
        return
    if fps_lock and fps_lock > 0:
        try:
            cap.set(cv2.CAP_PROP_FPS, float(fps_lock))
        except Exception:
            pass


def estimate_camera_fps(cap, seconds: float = 1.5) -> float:
    start = time.perf_counter()
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
        if time.perf_counter() - start >= seconds:
            break
    elapsed = time.perf_counter() - start
    if elapsed <= 0 or count <= 1:
        return 0.0
    return count / elapsed


def get_input_fps(cap, args) -> Tuple[float, float, float]:
    reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if args.input_mode == "video":
        measured_fps = reported_fps
        source_fps = reported_fps if reported_fps > 1.0 else args.ref_video_fps
        return reported_fps, measured_fps, float(source_fps)

    measured_fps = estimate_camera_fps(cap, seconds=args.fps_est_seconds)

    if args.fps_source == "reported":
        camera_fps = reported_fps
    elif args.fps_source == "target":
        camera_fps = float(args.fps_lock) if args.fps_lock > 0 else 0.0
    else:
        camera_fps = measured_fps

    if camera_fps <= 1.0:
        camera_fps = reported_fps if reported_fps > 1.0 else (float(args.fps_lock) if args.fps_lock > 1.0 else 25.0)

    return reported_fps, measured_fps, float(camera_fps)


def preprocess_frame_bgr(frame_bgr: np.ndarray, transform) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return transform(frame_rgb)


def _pad_indices(indices: List[int], target_len: int) -> List[int]:
    if not indices:
        return [0] * target_len
    while len(indices) < target_len:
        indices.append(indices[-1])
    return indices


def select_segment_indices(vlen: int, num_frames: int, mode: str, rng: np.random.Generator) -> List[int]:
    if vlen <= 0:
        return [0] * num_frames
    if vlen < num_frames:
        return _pad_indices(list(range(vlen)), num_frames)

    chunks = np.array_split(np.arange(vlen), num_frames)
    indices: List[int] = []
    for chunk in chunks:
        if len(chunk) == 0:
            indices.append(indices[-1] if indices else 0)
            continue
        indices.append(int(rng.choice(chunk)) if mode == "segment_random" else int(chunk[len(chunk) // 2]))
    return indices


def build_clip_from_buffer(frame_buffer: List[np.ndarray], transform, num_frames: int, sampling_mode: str, rng: np.random.Generator) -> torch.Tensor:
    indices = select_segment_indices(len(frame_buffer), num_frames, sampling_mode, rng)
    frames = [preprocess_frame_bgr(frame_buffer[i], transform) for i in indices]
    clip_tchw = torch.stack(frames, dim=0)
    return clip_tchw.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


@torch.no_grad()
def forward_one_clip(model, clip_bcthw: torch.Tensor, device: str, topk: int = 5):
    clip_bcthw = clip_bcthw.to(device, non_blocking=True)
    outputs = model(clip=clip_bcthw)
    logits = outputs["logits"]
    probs = F.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    k = min(topk, probs.shape[-1])
    topk_prob, topk_idx = torch.topk(probs, k=k, dim=-1)
    return int(pred.item()), float(conf.item()), topk_idx[0].tolist(), topk_prob[0].tolist()


def overlay_preview(base_img: np.ndarray, preview_img: Optional[np.ndarray], title: str = "SLR crop", max_w_ratio: float = 0.30, margin: int = 16) -> np.ndarray:
    out = base_img.copy()
    if preview_img is None or preview_img.size == 0:
        return out
    H, W = out.shape[:2]
    ph, pw = preview_img.shape[:2]
    target_w = max(1, int(W * max_w_ratio))
    scale = target_w / max(1, pw)
    target_h = max(1, int(ph * scale))
    if target_h > H - 2 * margin:
        target_h = H - 2 * margin
        scale = target_h / max(1, ph)
        target_w = max(1, int(pw * scale))
    preview_rs = cv2.resize(preview_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    x1 = W - margin - target_w
    y1 = margin
    x2 = x1 + target_w
    y2 = y1 + target_h
    roi = out[y1:y2, x1:x2].copy()
    black = np.zeros_like(roi)
    out[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.35, black, 0.65, 0)
    out[y1:y2, x1:x2] = preview_rs
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(out, title, (x1 + 8, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def main():
    parser = argparse.ArgumentParser("Realtime/Offline UFOneView MultiVSL200 inference")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--config", type=str, default="configs/UFOneView_MultiVSL200_realtime.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--input-mode", type=str, default="webcam", choices=["webcam", "video"])
    parser.add_argument("--video-path", type=str, default=None, help="Đường dẫn tới file .mp4 khi chạy offline mode")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--camera-url", type=str, default=None)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "dshow", "msmf", "default"])

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--show", action="store_true")

    parser.add_argument("--fps-lock", type=float, default=0.0)
    parser.add_argument("--fps-est-seconds", type=float, default=1.5)
    parser.add_argument("--fps-source", type=str, default="measured", choices=["measured", "reported", "target"])

    parser.add_argument("--ref-video-fps", type=float, default=25.0)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--temporal-stride", type=int, default=3)
    parser.add_argument("--window-seconds", type=float, default=0.0)
    parser.add_argument("--overlap-ratio", type=float, default=0.75)
    parser.add_argument("--sampling-mode", type=str, default="segment_center", choices=["segment_center", "segment_random"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--label-map-csv", type=str, default="data/lookuptable.csv", help="Đường dẫn tới file CSV có cột: id_label_in_documents, name")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--print-mode", type=str, default="change", choices=["always", "change"])
    parser.add_argument("--topk", type=int, default=3)

    parser.add_argument("--use-slr-crop", action="store_true", help="Bật crop SLR trước khi feed vào model")
    parser.add_argument("--pose-model", type=str, default="yolo11n-pose.pt")
    parser.add_argument("--clear-buffer-on-bad-crop", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config_path = (repo_root / args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    label_map_csv = ((repo_root / args.label_map_csv).resolve() if args.label_map_csv is not None else None)
    video_path = ((repo_root / args.video_path).resolve() if args.video_path is not None else None)

    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")
    if args.input_mode == "video":
        if video_path is None:
            raise ValueError("Chạy offline mode cần truyền --video-path")
        if not video_path.exists():
            raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA không khả dụng, chuyển sang CPU.")
        device = "cpu"

    print("[INFO] Loading model...")
    model, cfg = create_model_from_config(str(config_path), str(checkpoint_path), device)
    if cfg["data"]["model_name"] != "UFOneView":
        raise ValueError(f"Script này chỉ dành cho UFOneView, nhưng config hiện tại là {cfg['data']['model_name']}")

    num_classes = int(cfg["model"]["num_classes"])
    transform = build_test_transform(cfg)

    label_map, label_map_path = load_label_map(label_map_csv, num_classes)
    if label_map_path is not None:
        print(f"[INFO] Loaded label map csv: {label_map_path}")
    else:
        print("[WARN] Chưa truyền --label-map-csv. Sẽ fallback về cls_{idx}.")

    cropper = None
    if args.use_slr_crop:
        print("[INFO] Loading cropper...")
        cropper = SLRCropper(pose_model_path=args.pose_model)
    else:
        print("[INFO] SLR crop is OFF. Model sẽ nhận frame raw.")

    print(f"[INFO] Opening input... mode={args.input_mode}")
    cap = open_input_capture(
        input_mode=args.input_mode,
        camera_id=args.camera_id,
        camera_url=args.camera_url,
        backend=args.backend,
        video_path=str(video_path) if video_path is not None else None,
    )
    if not cap.isOpened():
        raise RuntimeError(
            f"Không mở được input. input_mode={args.input_mode}, camera_id={args.camera_id}, "
            f"camera_url={args.camera_url}, video_path={video_path}, backend={args.backend}"
        )

    try_set_camera_fps(cap, args.fps_lock, args.input_mode)
    reported_fps, measured_fps, camera_fps = get_input_fps(cap, args)

    if args.window_seconds > 0:
        window_sec = args.window_seconds
    else:
        window_sec = ((args.num_frames - 1) * args.temporal_stride + 1) / args.ref_video_fps

    buffer_size = max(args.num_frames, int(round(camera_fps * window_sec)))
    hop_sec = window_sec * max(0.01, 1.0 - args.overlap_ratio)
    infer_every = max(1, int(round(camera_fps * hop_sec)))

    print(f"[INFO] reported_fps         = {reported_fps:.2f}")
    print(f"[INFO] measured_fps         = {measured_fps:.2f}")
    print(f"[INFO] selected input_fps   = {camera_fps:.2f}")
    print(f"[INFO] num_frames           = {args.num_frames}")
    print(f"[INFO] temporal_stride      = {args.temporal_stride}")
    print(f"[INFO] window_sec           = {window_sec:.3f}")
    print(f"[INFO] buffer_size          = {buffer_size}")
    print(f"[INFO] infer_every          = {infer_every}")
    print(f"[INFO] approx_infer_fps     = {camera_fps / infer_every:.2f}")
    print("[INFO] Press 'q' to quit." if args.show else "[INFO] Press Ctrl+C to quit.")

    frame_buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
    loop_times: deque[float] = deque(maxlen=120)
    infer_times: deque[float] = deque(maxlen=120)
    frame_counter = 0
    accepted_counter = 0
    last_pred_text = None
    rng = np.random.default_rng(args.seed)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if args.input_mode == "video":
                    print("[INFO] Đã đọc hết video.")
                else:
                    print("[WARN] Không đọc được frame từ camera.")
                break

            frame_counter += 1
            now = time.perf_counter()
            loop_times.append(now)

            loop_fps = 0.0
            if len(loop_times) >= 2:
                dt = loop_times[-1] - loop_times[0]
                if dt > 0:
                    loop_fps = (len(loop_times) - 1) / dt

            crop = None
            crop_state = None
            debug_frame = frame.copy()

            if args.use_slr_crop:
                crop, crop_state, debug_frame, _ = cropper.process(frame)
                feed_frame = crop if crop is not None else None
            else:
                feed_frame = frame.copy()

            if feed_frame is not None:
                frame_buffer.append(feed_frame)
                accepted_counter += 1
            elif args.use_slr_crop and args.clear_buffer_on_bad_crop and crop_state is not None:
                frame_buffer.clear()
                accepted_counter = 0

            if len(frame_buffer) >= buffer_size and accepted_counter > 0 and accepted_counter % infer_every == 0:
                clip_bcthw = build_clip_from_buffer(list(frame_buffer), transform, args.num_frames, args.sampling_mode, rng)
                pred_idx, conf, topk_idx, topk_prob = forward_one_clip(model, clip_bcthw, device, topk=args.topk)

                infer_now = time.perf_counter()
                infer_times.append(infer_now)
                infer_fps = 0.0
                if len(infer_times) >= 2:
                    dt_inf = infer_times[-1] - infer_times[0]
                    if dt_inf > 0:
                        infer_fps = (len(infer_times) - 1) / dt_inf

                pred_text = label_map.get(pred_idx, f"cls_{pred_idx}")
                enough_conf = conf >= args.min_confidence
                tag = "PRED" if enough_conf else "INFO"
                topk_text = " | ".join(f"{label_map.get(i, f'cls_{i}')}: {p:.3f}" for i, p in zip(topk_idx, topk_prob))

                if enough_conf:
                    should_print = True if args.print_mode == "always" else (pred_text != last_pred_text)
                else:
                    should_print = True

                if should_print:
                    print(
                        f"[{tag}] pred={pred_text} | idx={pred_idx} | conf={conf:.4f} | "
                        f"min_conf={args.min_confidence:.4f} | loop_fps={loop_fps:.2f} | infer_fps={infer_fps:.2f} | "
                        f"accepted_frames={accepted_counter} | input_mode={args.input_mode} | crop={'on' if args.use_slr_crop else 'off'} | "
                        f"topk=[{topk_text}]"
                    )
                    if enough_conf:
                        last_pred_text = pred_text

            if args.show:
                show_frame = debug_frame.copy()
                if args.use_slr_crop:
                    show_frame = overlay_preview(show_frame, crop, title="SLR crop")
                cv2.putText(show_frame, f"mode={args.input_mode}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(show_frame, f"crop={'on' if args.use_slr_crop else 'off'}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(show_frame, f"buffer={len(frame_buffer)}/{buffer_size}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(show_frame, f"accepted={accepted_counter}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.imshow("UFOneView Realtime/Offline", show_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    
'''
Nhớ sửa biến default của label-map-csv
Bật crop thì nhớ thêm tag "--use-slr-crop" vào mỗi cái script
Có các input mode:
    "--input-mode webcam" thì nhớ bỏ thêm biến "--camera-id "
    "--input-mode video" thì nhớ bỏ thêm biến "--video-path"
    Còn riêng với dùng esp32 cam thì
    "--input-mode webcam" + "--camera-url"
'''
# realtime webcam, cameraid:
#python realtime_VLS200_cropped.py --input-mode webcam --camera-id 1 --checkpoint "Z:/SignLanguageReg/M3-SLR/checkpoint/uniformer_VSL.pth" --use-slr-crop --show

# realtime webcam, camera_url
#python realtime_VLS200_cropped.py --input-mode webcam --camera-url "http://192.168.1.123:81/stream" --checkpoint "Z:/SignLanguageReg/M3-SLR/checkpoint/uniformer_VSL.pth" --use-slr-crop --show

# video
#python realtime_VLS200_cropped.py --input-mode video --video-path "Z:\SignLanguageReg\VSL_data\New folder\14_Bao-Nam_1-200_13-14-15_0112___center_device19_signer14_center_ord1_36.mp4" --checkpoint "Z:/SignLanguageReg/M3-SLR/checkpoint/uniformer_VSL.pth" --show 
