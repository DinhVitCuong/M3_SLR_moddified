from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .capture import FPSInfo, get_input_fps, open_input_capture, try_set_camera_fps
from .pipeline import ClipBuilder, build_window_spec, forward_one_clip
from .render import draw_status_panel, overlay_preview
from .runtime import RuntimeResources


@dataclass
class RuntimeState:
    frame_buffer: deque
    loop_times: deque
    infer_times: deque
    accepted_counter: int = 0
    frame_counter: int = 0
    last_pred_text: Optional[str] = None


class RealtimeVSLApp:
    def __init__(self, runtime: RuntimeResources):
        self.runtime = runtime
        self.args = runtime.args
        self.model_bundle = runtime.model_bundle
        self.crop_processor = runtime.crop_processor

    def run(self) -> None:
        print(f"[INFO] Opening input... mode={self.args.input_mode}")
        capture = open_input_capture(
            input_mode=self.args.input_mode,
            camera_id=self.args.camera_id,
            camera_url=self.args.camera_url,
            backend=self.args.backend,
            video_path=self.runtime.paths.video_path,
        )
        if not capture.isOpened():
            raise RuntimeError(
                f"Không mở được input. input_mode={self.args.input_mode}, camera_id={self.args.camera_id}, "
                f"camera_url={self.args.camera_url}, video_path={self.runtime.paths.video_path}, "
                f"backend={self.args.backend}"
            )

        try:
            try_set_camera_fps(capture, self.args.fps_lock, self.args.input_mode)
            fps_info = get_input_fps(capture, self.args)
            window_spec = build_window_spec(fps_info.selected_fps, self.args)
            self._print_runtime_summary(fps_info, window_spec)
            state = RuntimeState(
                frame_buffer=deque(maxlen=window_spec.buffer_size),
                loop_times=deque(maxlen=120),
                infer_times=deque(maxlen=120),
            )
            clip_builder = ClipBuilder(
                num_frames=self.args.num_frames,
                sampling_mode=self.args.sampling_mode,
                seed=self.args.seed,
                transform=self.model_bundle.transform,
            )
            self._event_loop(capture, state, clip_builder, window_spec)
        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")
        finally:
            capture.release()
            if self.args.show:
                cv2.destroyAllWindows()

    def _event_loop(self, capture, state: RuntimeState, clip_builder: ClipBuilder, window_spec) -> None:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                self._handle_input_end()
                break

            state.frame_counter += 1
            loop_fps = self._append_and_compute_fps(state.loop_times)

            crop, crop_state, debug_frame, _ = self.crop_processor.process(frame)
            feed_frame = crop if self.crop_processor.enabled else frame.copy()

            if feed_frame is not None:
                state.frame_buffer.append(feed_frame)
                state.accepted_counter += 1
            elif self.crop_processor.enabled and self.args.clear_buffer_on_bad_crop and crop_state is not None:
                state.frame_buffer.clear()
                state.accepted_counter = 0

            if self._should_run_inference(state, window_spec.infer_every):
                self._run_inference(state, clip_builder, loop_fps)

            if self.args.show:
                self._render_frame(
                    debug_frame=debug_frame,
                    crop=crop,
                    state=state,
                    buffer_size=window_spec.buffer_size,
                )

    def _should_run_inference(self, state: RuntimeState, infer_every: int) -> bool:
        return (
            len(state.frame_buffer) >= state.frame_buffer.maxlen
            and state.accepted_counter > 0
            and state.accepted_counter % infer_every == 0
        )

    def _run_inference(self, state: RuntimeState, clip_builder: ClipBuilder, loop_fps: float) -> None:
        clip = clip_builder.build(list(state.frame_buffer))
        result = forward_one_clip(
            model_bundle=self.model_bundle,
            clip_bcthw=clip,
            min_confidence=self.args.min_confidence,
            topk=self.args.topk,
        )
        infer_fps = self._append_and_compute_fps(state.infer_times)
        if self._should_print_prediction(result.pred_text, result.enough_confidence, state.last_pred_text):
            tag = "PRED" if result.enough_confidence else "INFO"
            print(
                f"[{tag}] pred={result.pred_text} | idx={result.pred_idx} | conf={result.confidence:.4f} | "
                f"min_conf={self.args.min_confidence:.4f} | loop_fps={loop_fps:.2f} | infer_fps={infer_fps:.2f} | "
                f"accepted_frames={state.accepted_counter} | input_mode={self.args.input_mode} | "
                f"crop={'on' if self.crop_processor.enabled else 'off'} | "
                f"topk=[{result.topk_text(self.model_bundle.label_map)}]"
            )
            if result.enough_confidence:
                state.last_pred_text = result.pred_text

    def _render_frame(self, debug_frame: np.ndarray, crop: Optional[np.ndarray], state: RuntimeState, buffer_size: int) -> None:
        show_frame = debug_frame.copy()
        if self.crop_processor.enabled:
            show_frame = overlay_preview(show_frame, crop, title="SLR crop")
        show_frame = draw_status_panel(
            frame=show_frame,
            input_mode=self.args.input_mode,
            crop_enabled=self.crop_processor.enabled,
            buffer_size=buffer_size,
            buffer_len=len(state.frame_buffer),
            accepted_counter=state.accepted_counter,
        )
        cv2.imshow("UFOneView Realtime/Offline", show_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    def _print_runtime_summary(self, fps_info: FPSInfo, window_spec) -> None:
        print(f"[INFO] reported_fps         = {fps_info.reported_fps:.2f}")
        print(f"[INFO] measured_fps         = {fps_info.measured_fps:.2f}")
        print(f"[INFO] selected input_fps   = {fps_info.selected_fps:.2f}")
        print(f"[INFO] num_frames           = {self.args.num_frames}")
        print(f"[INFO] temporal_stride      = {self.args.temporal_stride}")
        print(f"[INFO] window_sec           = {window_spec.window_seconds:.3f}")
        print(f"[INFO] buffer_size          = {window_spec.buffer_size}")
        print(f"[INFO] infer_every          = {window_spec.infer_every}")
        print(f"[INFO] approx_infer_fps     = {window_spec.approx_infer_fps:.2f}")
        print("[INFO] Press 'q' to quit." if self.args.show else "[INFO] Press Ctrl+C to quit.")

    def _handle_input_end(self) -> None:
        if self.args.input_mode == "video":
            print("[INFO] Đã đọc hết video.")
        else:
            print("[WARN] Không đọc được frame từ camera.")

    def _should_print_prediction(self, pred_text: str, enough_confidence: bool, last_pred_text: Optional[str]) -> bool:
        if not enough_confidence:
            return True
        if self.args.print_mode == "always":
            return True
        return pred_text != last_pred_text

    @staticmethod
    def _append_and_compute_fps(time_buffer: deque) -> float:
        time_buffer.append(time.perf_counter())
        if len(time_buffer) < 2:
            return 0.0
        delta = time_buffer[-1] - time_buffer[0]
        if delta <= 0:
            return 0.0
        return (len(time_buffer) - 1) / delta
