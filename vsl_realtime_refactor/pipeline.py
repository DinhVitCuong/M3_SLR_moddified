from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.misc import load_config, load_label_map
from utils.utils import load_model
from utils.video_augmentation import Compose, Normalize, PermuteImage, Resize, ToFloatTensor
from .slr_cropper import SLRCropper

@dataclass(frozen=True)
class ModelBundle:
    model: torch.nn.Module
    cfg: dict
    transform: Compose
    label_map: Dict[int, str]
    label_map_path: Optional[str]
    device: str


@dataclass(frozen=True)
class WindowSpec:
    window_seconds: float
    buffer_size: int
    infer_every: int
    approx_infer_fps: float


@dataclass(frozen=True)
class InferenceResult:
    pred_idx: int
    confidence: float
    pred_text: str
    enough_confidence: bool
    topk_indices: List[int]
    topk_probabilities: List[float]

    def topk_text(self, label_map: Dict[int, str]) -> str:
        return " | ".join(
            f"{label_map.get(idx, f'cls_{idx}')}: {prob:.3f}"
            for idx, prob in zip(self.topk_indices, self.topk_probabilities)
        )


class ClipBuilder:
    def __init__(self, num_frames: int, sampling_mode: str, seed: int, transform):
        self.num_frames = num_frames
        self.sampling_mode = sampling_mode
        self.rng = np.random.default_rng(seed)
        self.transform = transform

    def build(self, frame_buffer: List[np.ndarray]) -> torch.Tensor:
        indices = self._select_segment_indices(len(frame_buffer), self.num_frames)
        frames = [self._preprocess_frame_bgr(frame_buffer[i]) for i in indices]
        clip_tchw = torch.stack(frames, dim=0)
        return clip_tchw.permute(1, 0, 2, 3).unsqueeze(0).contiguous()

    def _preprocess_frame_bgr(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(frame_rgb)

    def _select_segment_indices(self, video_len: int, num_frames: int) -> List[int]:
        if video_len <= 0:
            return [0] * num_frames
        if video_len < num_frames:
            return self._pad_indices(list(range(video_len)), num_frames)

        chunks = np.array_split(np.arange(video_len), num_frames)
        indices: List[int] = []
        for chunk in chunks:
            if len(chunk) == 0:
                indices.append(indices[-1] if indices else 0)
                continue
            if self.sampling_mode == "segment_random":
                indices.append(int(self.rng.choice(chunk)))
            else:
                indices.append(int(chunk[len(chunk) // 2]))
        return indices

    @staticmethod
    def _pad_indices(indices: List[int], target_len: int) -> List[int]:
        if not indices:
            return [0] * target_len
        while len(indices) < target_len:
            indices.append(indices[-1])
        return indices


class CropProcessor:
    def __init__(self, cropper: Optional[SLRCropper]):
        self.cropper = cropper

    @property
    def enabled(self) -> bool:
        return self.cropper is not None

    def process(self, frame: np.ndarray):
        if self.cropper is None:
            return frame.copy(), None, frame.copy(), None
        crop, crop_state, debug_frame, extra = self.cropper.process(frame)
        return crop, crop_state, debug_frame, extra


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


def create_model_bundle(config_path: str, checkpoint_path: str, device: str, label_map_csv) -> ModelBundle:
    cfg = load_config(config_path)
    cfg["training"]["test"] = True
    cfg["training"]["pretrained"] = True
    cfg["training"]["pretrained_model"] = checkpoint_path
    cfg["training"]["device"] = device
    cfg["training"]["batch_size"] = 1

    model = load_model(cfg).to(device)
    model.eval()

    if cfg["data"]["model_name"] != "UFOneView":
        raise ValueError(
            f"Script này chỉ dành cho UFOneView, nhưng config hiện tại là {cfg['data']['model_name']}"
        )

    transform = build_test_transform(cfg)
    num_classes = int(cfg["model"]["num_classes"])
    label_map, label_map_path = load_label_map(label_map_csv, num_classes)

    return ModelBundle(
        model=model,
        cfg=cfg,
        transform=transform,
        label_map=label_map,
        label_map_path=str(label_map_path) if label_map_path is not None else None,
        device=device,
    )


def build_window_spec(selected_fps: float, args) -> WindowSpec:
    if args.window_seconds > 0:
        window_seconds = float(args.window_seconds)
    else:
        window_seconds = ((args.num_frames - 1) * args.temporal_stride + 1) / args.ref_video_fps

    buffer_size = max(args.num_frames, int(round(selected_fps * window_seconds)))
    hop_seconds = window_seconds * max(0.01, 1.0 - args.overlap_ratio)
    infer_every = max(1, int(round(selected_fps * hop_seconds)))
    approx_infer_fps = selected_fps / infer_every

    return WindowSpec(
        window_seconds=window_seconds,
        buffer_size=buffer_size,
        infer_every=infer_every,
        approx_infer_fps=approx_infer_fps,
    )


@torch.no_grad()
def forward_one_clip(model_bundle: ModelBundle, clip_bcthw: torch.Tensor, min_confidence: float, topk: int) -> InferenceResult:
    clip_bcthw = clip_bcthw.to(model_bundle.device, non_blocking=True)
    outputs = model_bundle.model(clip=clip_bcthw)
    logits = outputs["logits"]
    probs = F.softmax(logits, dim=-1)
    confidence, pred = probs.max(dim=-1)
    k = min(topk, probs.shape[-1])
    topk_prob, topk_idx = torch.topk(probs, k=k, dim=-1)

    pred_idx = int(pred.item())
    confidence_value = float(confidence.item())
    pred_text = model_bundle.label_map.get(pred_idx, f"cls_{pred_idx}")

    return InferenceResult(
        pred_idx=pred_idx,
        confidence=confidence_value,
        pred_text=pred_text,
        enough_confidence=confidence_value >= min_confidence,
        topk_indices=topk_idx[0].tolist(),
        topk_probabilities=topk_prob[0].tolist(),
    )
