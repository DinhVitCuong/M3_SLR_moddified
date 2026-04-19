from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def overlay_preview(
    base_img: np.ndarray,
    preview_img: Optional[np.ndarray],
    title: str = "SLR crop",
    max_w_ratio: float = 0.30,
    margin: int = 16,
) -> np.ndarray:
    output = base_img.copy()
    if preview_img is None or preview_img.size == 0:
        return output

    height, width = output.shape[:2]
    preview_h, preview_w = preview_img.shape[:2]

    target_w = max(1, int(width * max_w_ratio))
    scale = target_w / max(1, preview_w)
    target_h = max(1, int(preview_h * scale))

    if target_h > height - 2 * margin:
        target_h = height - 2 * margin
        scale = target_h / max(1, preview_h)
        target_w = max(1, int(preview_w * scale))

    preview_resized = cv2.resize(preview_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    x1 = width - margin - target_w
    y1 = margin
    x2 = x1 + target_w
    y2 = y1 + target_h

    output[y1:y2, x1:x2] = preview_resized
    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(output, title, (x1 + 8, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return output


def draw_status_panel(
    frame: np.ndarray,
    input_mode: str,
    crop_enabled: bool,
    buffer_size: int,
    buffer_len: int,
    accepted_counter: int,
) -> np.ndarray:
    output = frame.copy()
    items = (
        f"mode={input_mode}",
        f"crop={'on' if crop_enabled else 'off'}",
        f"buffer={buffer_len}/{buffer_size}",
        f"accepted={accepted_counter}",
    )

    y = 35
    for text in items:
        cv2.putText(output, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y += 35

    return output
