"""
B1 · 质心提取器 (MediaPipe Tasks API)
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import json
from pathlib import Path
from dataclasses import dataclass

LANDMARK_GROUPS = {
    "head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "left_hand": [15, 17, 19, 21],
    "right_hand": [16, 18, 20, 22],
    "torso": [11, 12, 23, 24],
    "left_leg": [25, 27, 29, 31],
    "right_leg": [26, 28, 30, 32],
}

MODEL_PATH = "/home/claude/dance_aesthetics/pose_landmarker_lite.task"


@dataclass
class ExtractionResult:
    video_name: str
    fps: float
    frame_count: int
    timestamps: np.ndarray
    positions: dict[str, np.ndarray]
    confidence: dict[str, np.ndarray]

    def save(self, path: str):
        p = Path(path)
        np.savez_compressed(
            p.with_suffix('.npz'),
            timestamps=self.timestamps,
            **{f"pos_{k}": v for k, v in self.positions.items()},
            **{f"conf_{k}": v for k, v in self.confidence.items()},
        )
        meta = {
            "video_name": self.video_name,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration": float(self.timestamps[-1]) if len(self.timestamps) > 0 else 0,
            "nodes": list(self.positions.keys()),
        }
        p.with_suffix('.json').write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')


class PoseExtractor:
    def __init__(self):
        pass

    def extract(self, video_path, start_sec=0.0, end_sec=None,
                sample_fps=None, progress_interval=100):
        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if sample_fps is None or sample_fps >= orig_fps:
            sample_fps = orig_fps
            skip = 1
        else:
            skip = max(1, int(round(orig_fps / sample_fps)))

        start_frame = int(start_sec * orig_fps)
        end_frame = int(end_sec * orig_fps) if end_sec else total_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        video_name = Path(video_path).stem

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
        )
        landmarker = PoseLandmarker.create_from_options(options)

        timestamps_list = []
        positions = {k: [] for k in LANDMARK_GROUPS}
        confidences = {k: [] for k in LANDMARK_GROUPS}

        frame_idx = start_frame
        processed = 0

        print(f"  提取: {video_name}")
        print(f"  范围: {start_sec:.1f}s – {end_sec or total_frames/orig_fps:.1f}s, "
              f"采样: {sample_fps:.0f}fps (skip={skip})")

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % skip != 0:
                frame_idx += 1
                continue

            t = frame_idx / orig_fps
            ts_ms = int(t * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            try:
                result = landmarker.detect_for_video(mp_image, ts_ms)
            except Exception:
                frame_idx += 1
                continue

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lm = result.pose_landmarks[0]
                for group_name, indices in LANDMARK_GROUPS.items():
                    xs, ys, zs, vs = [], [], [], []
                    for idx in indices:
                        if idx < len(lm):
                            p = lm[idx]
                            vis = p.visibility if hasattr(p, 'visibility') and p.visibility else 0.5
                            if vis > 0.2:
                                xs.append(p.x)
                                ys.append(p.y)
                                zs.append(p.z)
                            vs.append(vis)

                    if xs:
                        positions[group_name].append([np.mean(xs), np.mean(ys), np.mean(zs)])
                        confidences[group_name].append(np.mean(vs))
                    else:
                        prev = positions[group_name][-1] if positions[group_name] else [0, 0, 0]
                        positions[group_name].append(prev)
                        confidences[group_name].append(0.0)

                timestamps_list.append(t)
                processed += 1

            if processed % progress_interval == 0 and processed > 0:
                pct = (frame_idx - start_frame) / max(1, end_frame - start_frame) * 100
                print(f"    {processed}帧 t={t:.2f}s ({pct:.0f}%)")

            frame_idx += 1

        landmarker.close()
        cap.release()
        print(f"  完成: {processed}帧")

        return ExtractionResult(
            video_name=video_name,
            fps=sample_fps,
            frame_count=processed,
            timestamps=np.array(timestamps_list),
            positions={k: np.array(v) for k, v in positions.items()},
            confidence={k: np.array(v) for k, v in confidences.items()},
        )
