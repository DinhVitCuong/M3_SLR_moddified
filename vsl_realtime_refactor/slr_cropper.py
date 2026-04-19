import time
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple


class SLRCropper:
    def __init__(
        self,
        pose_model_path: str = "yolo11n-pose.pt",
        kp_conf: float = 0.35,
        face_conf: float = 0.30,
        arm_conf: float = 0.25,
        head_top_extra_ratio: float = 0.75,
        side_pad_by_shoulder: float = 1.15,
        bottom_extra_ratio: float = 0.18,
        min_box_h_ratio: float = 0.38,
        min_box_w_ratio: float = 0.16,
        smooth_alpha: float = 0.75,
        center_x_tol_ratio: float = 0.24,
        max_shoulder_slope_ratio: float = 0.20,
        edge_margin_ratio: float = 0.01,
        min_face_points_for_frontal: int = 3,
        min_visible_eyes: int = 1,
        min_visible_ears: int = 1,
        min_shoulder_width_ratio: float = 0.06,
        max_face_center_offset_ratio: float = 0.18,
        max_nose_to_shoulder_center_ratio: float = 0.30,
    ):
        self.model = YOLO(pose_model_path)
        self.kp_conf = kp_conf
        self.face_conf = face_conf
        self.arm_conf = arm_conf
        self.head_top_extra_ratio = head_top_extra_ratio
        self.side_pad_by_shoulder = side_pad_by_shoulder
        self.bottom_extra_ratio = bottom_extra_ratio
        self.min_box_h_ratio = min_box_h_ratio
        self.min_box_w_ratio = min_box_w_ratio
        self.smooth_alpha = smooth_alpha
        self.center_x_tol_ratio = center_x_tol_ratio
        self.max_shoulder_slope_ratio = max_shoulder_slope_ratio
        self.edge_margin_ratio = edge_margin_ratio
        self.min_face_points_for_frontal = min_face_points_for_frontal
        self.min_visible_eyes = min_visible_eyes
        self.min_visible_ears = min_visible_ears
        self.min_shoulder_width_ratio = min_shoulder_width_ratio
        self.max_face_center_offset_ratio = max_face_center_offset_ratio
        self.max_nose_to_shoulder_center_ratio = max_nose_to_shoulder_center_ratio
        self.prev_box = None
        self.last_warning_state = None
        self.last_warning_print_time = 0.0
        self.warning_repeat_interval_sec = 5.0

    @staticmethod
    def clamp(v, lo, hi):
        return max(lo, min(v, hi))

    def valid_kp(self, conf_arr, idx, th=None):
        th = self.kp_conf if th is None else th
        return idx < len(conf_arr) and conf_arr[idx] >= th

    def valid_face_kp(self, conf_arr, idx):
        return self.valid_kp(conf_arr, idx, self.face_conf)

    def valid_arm_kp(self, conf_arr, idx):
        return self.valid_kp(conf_arr, idx, self.arm_conf)

    def smooth_box(self, cur_box):
        if cur_box is None:
            return None
        if self.prev_box is None:
            self.prev_box = cur_box
            return cur_box
        if self.smooth_alpha <= 0:
            self.prev_box = cur_box
            return cur_box
        smoothed = []
        for a, b in zip(self.prev_box, cur_box):
            smoothed.append(int(self.smooth_alpha * a + (1 - self.smooth_alpha) * b))
        self.prev_box = tuple(smoothed)
        return self.prev_box

    def get_main_person_index(self, result):
        if result.boxes is None or len(result.boxes) == 0:
            return None
        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        if len(areas) == 0:
            return None
        return int(np.argmax(areas))

    def draw_points(self, frame, kpts_xy, kpts_conf):
        ids = list(range(min(len(kpts_xy), 17)))
        for idx in ids:
            th = self.kp_conf
            if idx <= 4:
                th = self.face_conf
            elif idx in [7, 8, 9, 10]:
                th = self.arm_conf
            if idx < len(kpts_conf) and kpts_conf[idx] >= th:
                x, y = map(int, kpts_xy[idx])
                cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

    def check_frontal_pose(self, frame, kpts_xy, kpts_conf):
        visible_face = [i for i in range(5) if self.valid_face_kp(kpts_conf, i)]
        visible_eyes = [i for i in [1, 2] if self.valid_face_kp(kpts_conf, i)]
        visible_ears = [i for i in [3, 4] if self.valid_face_kp(kpts_conf, i)]

        if len(visible_face) < self.min_face_points_for_frontal:
            return False, "insufficient_face_points"
        if len(visible_eyes) < self.min_visible_eyes:
            return False, "eyes_not_visible"
        if len(visible_ears) < self.min_visible_ears:
            return False, "ears_not_visible"
        if not (self.valid_kp(kpts_conf, 5) and self.valid_kp(kpts_conf, 6)):
            return False, "missing_shoulders"

        ls = kpts_xy[5]
        rs = kpts_xy[6]
        shoulder_center = (ls + rs) / 2.0
        shoulder_w = float(np.linalg.norm(ls - rs))
        _, W = frame.shape[:2]
        if shoulder_w < self.min_shoulder_width_ratio * W:
            return False, "shoulder_width_too_small"

        shoulder_slope_ratio = abs(float(ls[1] - rs[1])) / max(shoulder_w, 1e-6)
        if shoulder_slope_ratio > self.max_shoulder_slope_ratio:
            return False, "shoulder_tilted"

        if self.valid_face_kp(kpts_conf, 0):
            nose = kpts_xy[0]
            nose_offset_ratio = abs(float(nose[0] - shoulder_center[0])) / max(shoulder_w, 1e-6)
            if nose_offset_ratio > self.max_nose_to_shoulder_center_ratio:
                return False, "nose_off_center"

        face_pts = [kpts_xy[i] for i in visible_face]
        face_center = np.mean(face_pts, axis=0)
        face_offset_ratio = abs(float(face_center[0] - shoulder_center[0])) / max(shoulder_w, 1e-6)
        if face_offset_ratio > self.max_face_center_offset_ratio:
            return False, "face_off_center"

        return True, "frontal_ok"

    def build_slr_crop_box(self, frame, kpts_xy, kpts_conf):
        H, W = frame.shape[:2]
        req = [5, 6, 11, 12]
        if not all(self.valid_kp(kpts_conf, i) for i in req):
            return None

        ls = kpts_xy[5]
        rs = kpts_xy[6]
        lh = kpts_xy[11]
        rh = kpts_xy[12]

        shoulder_center = (ls + rs) / 2.0
        shoulder_y = float((ls[1] + rs[1]) / 2.0)
        hip_y = float((lh[1] + rh[1]) / 2.0)
        torso_h = hip_y - shoulder_y
        if torso_h < 10:
            return None

        if self.valid_face_kp(kpts_conf, 0):
            nose_y = float(kpts_xy[0][1])
            head_to_shoulder = max(10.0, shoulder_y - nose_y)
            top_y = nose_y - self.head_top_extra_ratio * head_to_shoulder
        else:
            top_y = shoulder_y - 0.95 * torso_h

        left_shoulder_x = float(min(ls[0], rs[0]))
        right_shoulder_x = float(max(ls[0], rs[0]))
        shoulder_span_x = max(1.0, right_shoulder_x - left_shoulder_x)
        side_pad = self.side_pad_by_shoulder * shoulder_span_x
        x1 = left_shoulder_x - side_pad
        x2 = right_shoulder_x + side_pad

        y_candidates = [hip_y]
        for idx in [7, 8, 9, 10]:
            if self.valid_arm_kp(kpts_conf, idx):
                y_candidates.append(float(kpts_xy[idx][1]))
        bottom_y = max(y_candidates)
        bottom_y = max(bottom_y, hip_y + self.bottom_extra_ratio * torso_h)

        x1 = int(self.clamp(x1, 0, W - 1))
        x2 = int(self.clamp(x2, 0, W - 1))
        y1 = int(self.clamp(top_y, 0, H - 1))
        y2 = int(self.clamp(bottom_y, 0, H - 1))

        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def is_pose_good_for_slr(self, frame, kpts_xy, kpts_conf, box):
        H, W = frame.shape[:2]
        needed = [5, 6, 11, 12]
        if not all(self.valid_kp(kpts_conf, i) for i in needed):
            return False, "missing_keypoints"

        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        if box_h < self.min_box_h_ratio * H:
            return False, "box_too_small_h"
        if box_w < self.min_box_w_ratio * W:
            return False, "box_too_small_w"

        ls = kpts_xy[5]
        rs = kpts_xy[6]
        shoulder_center_x = float((ls[0] + rs[0]) / 2.0)
        frame_center_x = W / 2.0
        if abs(shoulder_center_x - frame_center_x) > self.center_x_tol_ratio * W:
            return False, "off_center"

        margin_x = self.edge_margin_ratio * W
        margin_y = self.edge_margin_ratio * H
        if x1 <= margin_x or x2 >= W - margin_x:
            return False, "touching_side_edge"
        if y1 <= margin_y or y2 >= H - margin_y:
            return False, "touching_top_bottom_edge"
        return True, "ok"

    def emit_warning_once(self, state):
        if state is None:
            self.last_warning_state = None
            return

        now = time.perf_counter()
        is_same_warning = (state == self.last_warning_state)
        enough_time_passed = (now - self.last_warning_print_time) >= self.warning_repeat_interval_sec

        if is_same_warning and not enough_time_passed:
            return

        self.last_warning_state = state
        self.last_warning_print_time = now

        if state == "no_person":
            print("[WARNING] không tìm thấy ai")
        elif state == "frontal_required":
            print("[WARNING] yêu cầu đứng chính diện")
        elif state == "bad_pose":
            print("[WARNING] không có đảm bảo output của model SLR sẽ hoạt động tốt")

    def process(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str], np.ndarray, Optional[Tuple[int, int, int, int]]]:
        vis = frame.copy()
        results = self.model.predict(frame, conf=0.4, verbose=False)
        state = "no_person"
        crop = None
        box = None

        if len(results) > 0:
            result = results[0]
            has_person = (
                result.boxes is not None
                and len(result.boxes) > 0
                and result.keypoints is not None
                and len(result.keypoints) > 0
            )
            if has_person:
                idx = self.get_main_person_index(result)
                if idx is not None:
                    kpts_xy = result.keypoints.xy[idx].cpu().numpy()
                    kpts_conf = result.keypoints.conf[idx].cpu().numpy()
                    is_frontal, reason = self.check_frontal_pose(frame, kpts_xy, kpts_conf)
                    self.draw_points(vis, kpts_xy, kpts_conf)
                    if not is_frontal:
                        state = "frontal_required"
                        cv2.putText(vis, "Front-facing required", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    else:
                        raw_box = self.build_slr_crop_box(frame, kpts_xy, kpts_conf)
                        if raw_box is None:
                            state = "bad_pose"
                        else:
                            box = self.smooth_box(raw_box)
                            if box is None:
                                state = "bad_pose"
                            else:
                                ok_pose, pose_reason = self.is_pose_good_for_slr(frame, kpts_xy, kpts_conf, box)
                                x1, y1, x2, y2 = box
                                if ok_pose:
                                    state = None
                                    crop = frame[y1:y2, x1:x2].copy()
                                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(vis, "SLR-ready", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                else:
                                    state = "bad_pose"
                                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
                                    cv2.putText(vis, f"Pose not ideal for SLR: {pose_reason}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        self.emit_warning_once(state)
        return crop, state, vis, box
