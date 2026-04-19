import cv2
import numpy as np
import argparse
from ultralytics import YOLO

MODEL_PATH = "yolo11n-pose.pt"

KP_CONF = 0.35
FACE_CONF = 0.30
ARM_CONF = 0.25

# ===== Crop rule mới =====
HEAD_TOP_EXTRA_RATIO = 0.75      # tăng thêm để phần đầu thoáng hơn
SIDE_PAD_BY_SHOULDER = 1.5      # mỗi bên chừa thêm = 1.15 * shoulder_width
BOTTOM_EXTRA_RATIO = 0.18        # fallback nếu không thấy tay rõ
MIN_BOX_H_RATIO = 0.38
MIN_BOX_W_RATIO = 0.16

SMOOTH_ALPHA = 0.75

# ===== Rule cho SLR =====
CENTER_X_TOL_RATIO = 0.24
MAX_SHOULDER_SLOPE_RATIO = 0.20
EDGE_MARGIN_RATIO = 0.01

# frontal / non-frontal
MIN_FACE_POINTS_FOR_FRONTAL = 3
MIN_VISIBLE_EYES = 1
MIN_VISIBLE_EARS = 1
MIN_SHOULDER_WIDTH_RATIO = 0.06
MAX_FACE_CENTER_OFFSET_RATIO = 0.18
MAX_NOSE_TO_SHOULDER_CENTER_RATIO = 0.30

prev_box = None
last_warning_state = None


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def smooth_box(cur_box, alpha=0.75):
    global prev_box
    if cur_box is None:
        return None

    if prev_box is None:
        prev_box = cur_box
        return cur_box

    smoothed = []
    for a, b in zip(prev_box, cur_box):
        smoothed.append(int(alpha * a + (1 - alpha) * b))
    prev_box = tuple(smoothed)
    return prev_box


def get_main_person_index(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if len(areas) == 0:
        return None
    return int(np.argmax(areas))


def valid_kp(conf_arr, idx, th=KP_CONF):
    return idx < len(conf_arr) and conf_arr[idx] >= th


def valid_face_kp(conf_arr, idx, th=FACE_CONF):
    return idx < len(conf_arr) and conf_arr[idx] >= th


def valid_arm_kp(conf_arr, idx, th=ARM_CONF):
    return idx < len(conf_arr) and conf_arr[idx] >= th


def draw_points(frame, kpts_xy, kpts_conf):
    ids = list(range(min(len(kpts_xy), 17)))
    for idx in ids:
        th = KP_CONF
        if idx <= 4:
            th = FACE_CONF
        elif idx in [7, 8, 9, 10]:
            th = ARM_CONF

        if idx < len(kpts_conf) and kpts_conf[idx] >= th:
            x, y = map(int, kpts_xy[idx])
            cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)


def check_frontal_pose(frame, kpts_xy, kpts_conf):
    visible_face = [i for i in range(5) if valid_face_kp(kpts_conf, i)]
    visible_eyes = [i for i in [1, 2] if valid_face_kp(kpts_conf, i)]
    visible_ears = [i for i in [3, 4] if valid_face_kp(kpts_conf, i)]

    if len(visible_face) < MIN_FACE_POINTS_FOR_FRONTAL:
        return False, "insufficient_face_points"

    if len(visible_eyes) < MIN_VISIBLE_EYES:
        return False, "eyes_not_visible"

    if len(visible_ears) < MIN_VISIBLE_EARS:
        return False, "ears_not_visible"

    if not (valid_kp(kpts_conf, 5) and valid_kp(kpts_conf, 6)):
        return False, "missing_shoulders"

    ls = kpts_xy[5]
    rs = kpts_xy[6]
    shoulder_center = (ls + rs) / 2.0
    shoulder_w = float(np.linalg.norm(ls - rs))

    H, W = frame.shape[:2]
    if shoulder_w < MIN_SHOULDER_WIDTH_RATIO * W:
        return False, "shoulder_width_too_small"

    shoulder_slope_ratio = abs(float(ls[1] - rs[1])) / max(shoulder_w, 1e-6)
    if shoulder_slope_ratio > MAX_SHOULDER_SLOPE_RATIO:
        return False, "shoulder_tilted"

    if valid_face_kp(kpts_conf, 0):
        nose = kpts_xy[0]
        nose_offset_ratio = abs(float(nose[0] - shoulder_center[0])) / max(shoulder_w, 1e-6)
        if nose_offset_ratio > MAX_NOSE_TO_SHOULDER_CENTER_RATIO:
            return False, "nose_off_center"

    face_pts = [kpts_xy[i] for i in visible_face]
    face_center = np.mean(face_pts, axis=0)
    face_offset_ratio = abs(float(face_center[0] - shoulder_center[0])) / max(shoulder_w, 1e-6)
    if face_offset_ratio > MAX_FACE_CENTER_OFFSET_RATIO:
        return False, "face_off_center"

    return True, "frontal_ok"


def build_slr_crop_box(frame, kpts_xy, kpts_conf):
    """
    Crop theo rule mới:
    - top rộng phần đầu hơn
    - ngang chỉ dựa vào span ngang của 2 vai
    - mỗi bên pad thêm theo shoulder_span_x
    - bottom vẫn dùng tay để đủ lọt tay buông thõng
    """
    H, W = frame.shape[:2]

    req = [5, 6, 11, 12]
    if not all(valid_kp(kpts_conf, i) for i in req):
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

    center_x = float(shoulder_center[0])

    # ===== TOP: rộng phần đầu ra thêm =====
    if valid_face_kp(kpts_conf, 0):  # nose
        nose_y = float(kpts_xy[0][1])
        head_to_shoulder = max(10.0, shoulder_y - nose_y)
        top_y = nose_y - HEAD_TOP_EXTRA_RATIO * head_to_shoulder
    else:
        top_y = shoulder_y - 0.95 * torso_h

    # ===== LEFT/RIGHT: chỉ theo span ngang của 2 vai =====
    left_shoulder_x = float(min(ls[0], rs[0]))
    right_shoulder_x = float(max(ls[0], rs[0]))
    shoulder_span_x = max(1.0, right_shoulder_x - left_shoulder_x)

    # mỗi bên chừa thêm 1.0 -> 1.5 lần bề ngang vai
    side_pad = SIDE_PAD_BY_SHOULDER * shoulder_span_x

    x1 = left_shoulder_x - side_pad
    x2 = right_shoulder_x + side_pad

    # ===== BOTTOM: đủ tay buông thõng =====
    y_candidates = [hip_y]

    if valid_arm_kp(kpts_conf, 7):   # left elbow
        y_candidates.append(float(kpts_xy[7][1]))
    if valid_arm_kp(kpts_conf, 8):   # right elbow
        y_candidates.append(float(kpts_xy[8][1]))
    if valid_arm_kp(kpts_conf, 9):   # left wrist
        y_candidates.append(float(kpts_xy[9][1]))
    if valid_arm_kp(kpts_conf, 10):  # right wrist
        y_candidates.append(float(kpts_xy[10][1]))

    bottom_y = max(y_candidates)
    bottom_y = max(bottom_y, hip_y + BOTTOM_EXTRA_RATIO * torso_h)

    # clamp
    x1 = int(clamp(x1, 0, W - 1))
    x2 = int(clamp(x2, 0, W - 1))
    y1 = int(clamp(top_y, 0, H - 1))
    y2 = int(clamp(bottom_y, 0, H - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def is_pose_good_for_slr(frame, kpts_xy, kpts_conf, box):
    H, W = frame.shape[:2]

    needed = [5, 6, 11, 12]
    if not all(valid_kp(kpts_conf, i) for i in needed):
        return False, "missing_keypoints"

    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1

    if box_h < MIN_BOX_H_RATIO * H:
        return False, "box_too_small_h"
    if box_w < MIN_BOX_W_RATIO * W:
        return False, "box_too_small_w"

    ls = kpts_xy[5]
    rs = kpts_xy[6]
    shoulder_center_x = float((ls[0] + rs[0]) / 2.0)
    frame_center_x = W / 2.0

    if abs(shoulder_center_x - frame_center_x) > CENTER_X_TOL_RATIO * W:
        return False, "off_center"

    margin_x = EDGE_MARGIN_RATIO * W
    margin_y = EDGE_MARGIN_RATIO * H

    if x1 <= margin_x or x2 >= W - margin_x:
        return False, "touching_side_edge"

    if y1 <= margin_y or y2 >= H - margin_y:
        return False, "touching_top_bottom_edge"

    return True, "ok"


def emit_warning_once(state):
    global last_warning_state

    if state is None:
        last_warning_state = None
        return

    if state == last_warning_state:
        return

    last_warning_state = state

    if state == "no_person":
        print("không tìm thấy ai")
    elif state == "frontal_required":
        print("yêu cầu đứng chính diện")
    elif state == "bad_pose":
        print("không có đảm bảo output của model SLR sẽ hoạt động tốt")


def resize_to_same_height(img, target_h):
    if img is None or img.size == 0:
        return np.zeros((target_h, 320, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)


def main(camera_id):
    global prev_box, last_warning_state
    prev_box = None
    last_warning_state = None

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"Không mở được camera_id={camera_id}")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        crop_panel = np.zeros((frame.shape[0], 320, 3), dtype=np.uint8)

        results = model.predict(frame, conf=0.4, verbose=False)
        current_state = "no_person"

        if len(results) > 0:
            result = results[0]

            has_person = (
                result.boxes is not None
                and len(result.boxes) > 0
                and result.keypoints is not None
                and len(result.keypoints) > 0
            )

            if has_person:
                idx = get_main_person_index(result)

                if idx is not None:
                    kpts_xy = result.keypoints.xy[idx].cpu().numpy()
                    kpts_conf = result.keypoints.conf[idx].cpu().numpy()

                    is_frontal, _ = check_frontal_pose(frame, kpts_xy, kpts_conf)

                    if not is_frontal:
                        current_state = "frontal_required"
                        draw_points(vis, kpts_xy, kpts_conf)
                        cv2.putText(
                            vis,
                            "Front-facing required",
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            2
                        )
                    else:
                        box = build_slr_crop_box(frame, kpts_xy, kpts_conf)

                        if box is None:
                            current_state = "bad_pose"
                        else:
                            box = smooth_box(box, alpha=SMOOTH_ALPHA)

                            if box is None:
                                current_state = "bad_pose"
                            else:
                                is_good, _ = is_pose_good_for_slr(frame, kpts_xy, kpts_conf, box)

                                x1, y1, x2, y2 = box
                                draw_points(vis, kpts_xy, kpts_conf)

                                if is_good:
                                    current_state = None
                                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(
                                        vis,
                                        "SLR-ready",
                                        (x1, max(20, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8,
                                        (0, 255, 0),
                                        2
                                    )

                                    crop = frame[y1:y2, x1:x2]
                                    crop_panel = resize_to_same_height(crop, frame.shape[0])
                                else:
                                    current_state = "bad_pose"
                                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
                                    cv2.putText(
                                        vis,
                                        "Pose not ideal for SLR",
                                        (x1, max(20, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 165, 255),
                                        2
                                    )

        emit_warning_once(current_state)

        cv2.putText(
            vis,
            f"Original - camera_id={camera_id}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        cv2.putText(
            crop_panel,
            "SLR crop",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        if crop_panel.shape[0] != vis.shape[0]:
            crop_panel = resize_to_same_height(crop_panel, vis.shape[0])

        combined = np.hstack([vis, crop_panel])
        cv2.imshow("SLR Crop Demo", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", type=int, default=0)
    args = parser.parse_args()

    main(args.camera_id)

#python upper_body_crop.py --camera_id 1