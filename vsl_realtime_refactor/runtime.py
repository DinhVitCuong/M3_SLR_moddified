from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .pipeline import CropProcessor, create_model_bundle


@dataclass(frozen=True)
class AppPaths:
    repo_root: Path
    config_path: Path
    checkpoint_path: Path
    label_map_csv: Optional[Path]
    video_path: Optional[Path]


@dataclass(frozen=True)
class RuntimeResources:
    args: argparse.Namespace
    paths: AppPaths
    model_bundle: object
    crop_processor: CropProcessor


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="segment_center",
        choices=["segment_center", "segment_random"],
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--label-map-csv",
        type=str,
        default="data/lookuptable.csv",
        help="Đường dẫn tới file CSV có cột: id_label_in_documents, name",
    )
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--print-mode", type=str, default="change", choices=["always", "change"])
    parser.add_argument("--topk", type=int, default=3)

    parser.add_argument("--use-slr-crop", action="store_true", help="Bật crop SLR trước khi feed vào model")
    parser.add_argument("--pose-model", type=str, default="yolo11n-pose.pt")
    parser.add_argument("--clear-buffer-on-bad-crop", action="store_true")
    return parser


def resolve_paths(args: argparse.Namespace) -> AppPaths:
    repo_root = Path(args.repo_root).resolve()
    config_path = (repo_root / args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    label_map_csv = (repo_root / args.label_map_csv).resolve() if args.label_map_csv is not None else None
    video_path = (repo_root / args.video_path).resolve() if args.video_path is not None else None
    return AppPaths(
        repo_root=repo_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        label_map_csv=label_map_csv,
        video_path=video_path,
    )


def validate_paths(paths: AppPaths, input_mode: str) -> None:
    if not paths.config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {paths.config_path}")
    if not paths.checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {paths.checkpoint_path}")

    if input_mode == "video":
        if paths.video_path is None:
            raise ValueError("Chạy offline mode cần truyền --video-path")
        if not paths.video_path.exists():
            raise FileNotFoundError(f"Không tìm thấy video: {paths.video_path}")


def normalize_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA không khả dụng, chuyển sang CPU.")
        return "cpu"
    return device


def prepare_runtime(args: argparse.Namespace) -> RuntimeResources:
    paths = resolve_paths(args)
    validate_paths(paths, args.input_mode)

    device = normalize_device(args.device)

    print("[INFO] Loading model...")
    model_bundle = create_model_bundle(
        config_path=str(paths.config_path),
        checkpoint_path=str(paths.checkpoint_path),
        device=device,
        label_map_csv=paths.label_map_csv,
    )

    if model_bundle.label_map_path is not None:
        print(f"[INFO] Loaded label map csv: {model_bundle.label_map_path}")
    else:
        print("[WARN] Chưa truyền --label-map-csv. Sẽ fallback về cls_{idx}.")

    crop_processor = CropProcessor(None)
    if args.use_slr_crop:
        print("[INFO] Loading cropper...")
        from slr_cropper import SLRCropper

        crop_processor = CropProcessor(SLRCropper(pose_model_path=args.pose_model))
    else:
        print("[INFO] SLR crop is OFF. Model sẽ nhận frame raw.")

    args.device = device
    return RuntimeResources(
        args=args,
        paths=paths,
        model_bundle=model_bundle,
        crop_processor=crop_processor,
    )
