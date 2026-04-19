from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2


@dataclass(frozen=True)
class FPSInfo:
    reported_fps: float
    measured_fps: float
    selected_fps: float


def open_camera_capture(camera_id: int, camera_url: Optional[str], backend: str):
    if camera_url:
        return cv2.VideoCapture(camera_url)
    if backend == "dshow":
        return cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if backend == "msmf":
        return cv2.VideoCapture(camera_id, cv2.CAP_MSMF)
    if backend == "default":
        return cv2.VideoCapture(camera_id)

    for api_pref in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cap = cv2.VideoCapture(camera_id, api_pref)
        if cap.isOpened():
            return cap
        cap.release()

    return cv2.VideoCapture(camera_id)


def open_input_capture(
    input_mode: str,
    camera_id: int,
    camera_url: Optional[str],
    backend: str,
    video_path: Optional[Path],
):
    if input_mode == "video":
        if video_path is None:
            raise ValueError("input_mode=video nhưng chưa truyền --video-path")
        return cv2.VideoCapture(str(video_path))
    return open_camera_capture(camera_id, camera_url, backend)


def try_set_camera_fps(cap, fps_lock: float, input_mode: str) -> None:
    if input_mode != "webcam" or fps_lock <= 0:
        return
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


def get_input_fps(cap, args) -> FPSInfo:
    reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if args.input_mode == "video":
        measured_fps = reported_fps
        selected_fps = reported_fps if reported_fps > 1.0 else float(args.ref_video_fps)
        return FPSInfo(
            reported_fps=reported_fps,
            measured_fps=measured_fps,
            selected_fps=selected_fps,
        )

    measured_fps = estimate_camera_fps(cap, seconds=args.fps_est_seconds)

    if args.fps_source == "reported":
        selected_fps = reported_fps
    elif args.fps_source == "target":
        selected_fps = float(args.fps_lock) if args.fps_lock > 0 else 0.0
    else:
        selected_fps = measured_fps

    if selected_fps <= 1.0:
        if reported_fps > 1.0:
            selected_fps = reported_fps
        elif args.fps_lock > 1.0:
            selected_fps = float(args.fps_lock)
        else:
            selected_fps = 25.0

    return FPSInfo(
        reported_fps=reported_fps,
        measured_fps=measured_fps,
        selected_fps=float(selected_fps),
    )
