"""
recognize_attendance.py - Real-Time Face Recognition Attendance Module
Continuously captures webcam frames, detects faces, compares 128-d encodings
against enrolled students in SQLite (distance threshold <= 0.6), logs attendance
with duplicate prevention for the current day, displays bounding boxes with
names/confidence, and supports multi-face recognition and liveness blink detection.
"""

import sys
import time
import threading
from datetime import datetime, date
import numpy as np
import cv2

from database import (
    init_db,
    get_all_students,
    mark_attendance,
    is_attendance_marked
)
import face_engine


class AttendanceRecognizer:
    """
    Main real-time face recognition processor.
    Can be run as a standalone loop or embedded inside Tkinter/GUI via callbacks.
    """
    def __init__(
        self,
        camera_index: int = 0,
        distance_threshold: float = 0.6,
        enable_liveness: bool = True,
        scale_factor: float = 0.5
    ):
        """
        Args:
            camera_index: Webcam device index.
            distance_threshold: Euclidean distance cutoff (default 0.6).
            enable_liveness: Enable blink detection for anti-spoofing.
            scale_factor: Image scaling factor for real-time performance (0.5 = 2x speedup).
        """
        self.camera_index = camera_index
        self.distance_threshold = distance_threshold
        self.enable_liveness = enable_liveness
        self.scale_factor = scale_factor

        self.is_running = False
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None

        # Cached students data: list of {'id', 'name', 'roll_no', 'face_encoding'}
        self.known_students: list[dict] = []
        self.known_encodings: list[np.ndarray] = []
        
        # In-memory fast cache of students marked today to minimize database writes
        self.marked_today_ids: set[int] = set()
        self.today_date_str = date.today().isoformat()

        # Track recent alert banners (e.g. "Marked: John Doe")
        self.recent_alerts: list[dict] = []

        # Liveness trackers per face index
        self.liveness_trackers: dict[int, face_engine.LivenessTracker] = {}

        # Callbacks
        self.on_frame_callback = None
        self.on_attendance_marked_callback = None
        self.on_status_callback = None

    def reload_known_faces(self):
        """Reloads registered students and their encodings from SQLite database."""
        students = get_all_students()
        self.known_students = students
        self.known_encodings = [s["face_encoding"] for s in students]

        # Reset daily cache if day changed
        current_date = date.today().isoformat()
        if current_date != self.today_date_str:
            self.today_date_str = current_date
            self.marked_today_ids.clear()

        # Pre-populate marked_today_ids from database
        self.marked_today_ids = {
            s["id"] for s in self.known_students
            if is_attendance_marked(s["id"], self.today_date_str)
        }

    def start(self, frame_callback=None, attendance_callback=None, status_callback=None):
        """Starts real-time recognition in a background worker thread."""
        if self.is_running:
            return

        self.on_frame_callback = frame_callback
        self.on_attendance_marked_callback = attendance_callback
        self.on_status_callback = status_callback

        self.is_running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stops real-time recognition loop and releases camera resources."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self._cap and self._cap.isOpened():
            self._cap.release()
            self._cap = None

    def _worker_loop(self):
        """Internal capture and processing loop."""
        self.reload_known_faces()
        self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            err_msg = f"Cannot open camera (Index: {self.camera_index}). Check webcam connection."
            if self.on_status_callback:
                self.on_status_callback(err_msg, is_error=True)
            self.is_running = False
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fps_timer = time.time()
        frame_counter = 0
        fps = 0.0

        if self.on_status_callback:
            self.on_status_callback(f"Live scanner active. {len(self.known_students)} enrolled students loaded.")

        try:
            while self.is_running:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # Mirror frame
                frame = cv2.flip(frame, 1)

                # Process frame
                annotated_frame = self.process_frame(frame)

                # Calculate FPS
                frame_counter += 1
                if frame_counter % 10 == 0:
                    now = time.time()
                    dt = now - fps_timer
                    fps = 10.0 / dt if dt > 0 else 0.0
                    fps_timer = now

                # Draw top HUD bar (FPS, engine, date)
                self._draw_hud(annotated_frame, fps)

                # Callback or GUI display
                if self.on_frame_callback:
                    self.on_frame_callback(annotated_frame)
                else:
                    cv2.imshow("Face Recognition Attendance - Press 'q' to Quit", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        self.is_running = False
                        break

        finally:
            if self._cap and self._cap.isOpened():
                self._cap.release()
                self._cap = None
            if not self.on_frame_callback:
                cv2.destroyAllWindows()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Performs multi-face detection, distance matching, attendance logging,
        and graphic overlays on a single video frame.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Downscale for faster real-time processing
        if self.scale_factor != 1.0:
            small_frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        else:
            small_frame = frame

        # Detect faces (bounding boxes in small frame coordinates)
        small_locations = face_engine.detect_faces(small_frame)

        if not small_locations:
            return annotated

        # Upscale face locations back to original frame dimensions
        inv_scale = 1.0 / self.scale_factor
        face_locations = []
        for (top, right, bottom, left) in small_locations:
            face_locations.append((
                int(top * inv_scale),
                int(right * inv_scale),
                int(bottom * inv_scale),
                int(left * inv_scale)
            ))

        # Compute face encodings
        encodings = face_engine.encode_faces(frame, face_locations)

        current_time_str = datetime.now().strftime("%H:%M:%S")
        current_date_str = date.today().isoformat()

        # Iterate over all faces detected in this single frame (Multi-Face Support)
        for idx, (face_loc, encoding) in enumerate(zip(face_locations, encodings)):
            top, right, bottom, left = face_loc

            # Clamp coordinates
            top = max(0, top)
            left = max(0, left)
            bottom = min(h, bottom)
            right = min(w, right)

            # Liveness / Blink Detection (Stretch Goal)
            is_live = True
            liveness_label = ""
            if self.enable_liveness:
                if idx not in self.liveness_trackers:
                    self.liveness_trackers[idx] = face_engine.LivenessTracker()
                tracker = self.liveness_trackers[idx]
                is_live, metric = tracker.update(frame, face_loc)
                liveness_label = "Live: Verified" if is_live else "Blink to verify"

            # Face Matching Logic
            name = "Unknown"
            roll_no = "N/A"
            confidence = 0.0
            is_match = False
            student_id = None
            already_marked = False

            if len(self.known_encodings) > 0:
                # 1. Calculate Euclidean distances to all registered students
                distances = face_engine.face_distance(self.known_encodings, encoding)
                best_match_idx = int(np.argmin(distances))
                min_distance = distances[best_match_idx]

                # 2. Check distance threshold (0.6)
                if min_distance <= self.distance_threshold:
                    matched_student = self.known_students[best_match_idx]
                    student_id = matched_student["id"]
                    name = matched_student["name"]
                    roll_no = matched_student["roll_no"]
                    confidence = face_engine.calculate_confidence(min_distance, self.distance_threshold)
                    is_match = True
                else:
                    confidence = face_engine.calculate_confidence(min_distance, self.distance_threshold)

            # Attendance Recording Logic
            if is_match and student_id is not None:
                if student_id in self.marked_today_ids:
                    already_marked = True
                    status_text = "Already Marked"
                else:
                    # Mark attendance in database (anti-spoofing check can be required)
                    if not self.enable_liveness or is_live:
                        success, _ = mark_attendance(
                            student_id=student_id,
                            date_str=current_date_str,
                            time_str=current_time_str,
                            status="Present"
                        )
                        if success:
                            self.marked_today_ids.add(student_id)
                            already_marked = False
                            status_text = "Attendance Marked!"

                            # Add recent alert banner
                            self.recent_alerts.append({
                                "text": f"Marked: {name} ({roll_no}) at {current_time_str}",
                                "time": time.time(),
                                "color": (0, 220, 0)
                            })

                            # Invoke callback
                            if self.on_attendance_marked_callback:
                                self.on_attendance_marked_callback({
                                    "id": student_id,
                                    "name": name,
                                    "roll_no": roll_no,
                                    "time_in": current_time_str,
                                    "status": "Present"
                                })
                        else:
                            self.marked_today_ids.add(student_id)
                            already_marked = True
                            status_text = "Already Marked"
                    else:
                        status_text = "Blink to confirm"
            else:
                status_text = "Unknown"

            # Visual Rendering on Frame
            # Colors: Green for newly marked, Cyan for already marked, Red for Unknown
            if is_match:
                if already_marked:
                    box_color = (255, 191, 0)  # Cyan/Deep Sky Blue
                else:
                    box_color = (0, 220, 0)    # Bright Green
            else:
                box_color = (0, 0, 230)        # Crimson Red

            # 1. Draw corner brackets around face for modern styling
            self._draw_styled_box(annotated, left, top, right, bottom, box_color)

            # 2. Draw stylish label tag above or below face
            label_y = top - 10 if top - 45 > 10 else bottom + 25
            info_tag = f"{name} ({roll_no})" if is_match else "Unknown"
            sub_tag = f"{confidence:.0f}% | {status_text}"
            if liveness_label:
                sub_tag += f" | {liveness_label}"

            # Tag background pill
            tag_width = max(len(info_tag), len(sub_tag)) * 9 + 20
            tag_top = label_y - 20
            cv2.rectangle(
                annotated,
                (left, tag_top),
                (left + tag_width, tag_top + 38),
                (20, 20, 20),
                -1
            )
            cv2.rectangle(
                annotated,
                (left, tag_top),
                (left + tag_width, tag_top + 38),
                box_color,
                1
            )

            # Text labels
            cv2.putText(
                annotated,
                info_tag,
                (left + 8, tag_top + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            cv2.putText(
                annotated,
                sub_tag,
                (left + 8, tag_top + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                box_color,
                1
            )

        # Draw pop-up notification banners on frame
        self._draw_alert_banners(annotated)

        return annotated

    def _draw_styled_box(self, img, l, t, r, b, color, corner_len=18, thickness=2):
        """Draws bounding rectangle with accentuated modern aesthetic corners."""
        # Main thin bounding rectangle
        cv2.rectangle(img, (l, t), (r, b), color, 1)

        # Top-left corner
        cv2.line(img, (l, t), (l + corner_len, t), color, thickness)
        cv2.line(img, (l, t), (l, t + corner_len), color, thickness)
        # Top-right corner
        cv2.line(img, (r, t), (r - corner_len, t), color, thickness)
        cv2.line(img, (r, t), (r, t + corner_len), color, thickness)
        # Bottom-left corner
        cv2.line(img, (l, b), (l + corner_len, b), color, thickness)
        cv2.line(img, (l, b), (l, b - corner_len), color, thickness)
        # Bottom-right corner
        cv2.line(img, (r, b), (r - corner_len, b), color, thickness)
        cv2.line(img, (r, b), (r, b - corner_len), color, thickness)

    def _draw_hud(self, img, fps: float):
        """Draws top information bar displaying engine, date, time and FPS."""
        w = img.shape[1]
        cv2.rectangle(img, (0, 0), (w, 30), (20, 20, 20), -1)

        engine_info = face_engine.get_engine_info()
        engine_str = "Dlib" if engine_info["is_dlib"] else "OpenCV"
        time_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

        hud_text = f"FPS: {fps:.1f} | Engine: {engine_str} | Enrolled: {len(self.known_students)} | {time_str}"
        cv2.putText(
            img,
            hud_text,
            (12, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (210, 210, 210),
            1
        )

    def _draw_alert_banners(self, img):
        """Renders temporary recognition notification banners at bottom of video."""
        now = time.time()
        # Keep alerts for 3 seconds
        self.recent_alerts = [a for a in self.recent_alerts if now - a["time"] < 3.0]

        h, w = img.shape[:2]
        y_offset = h - 25
        for alert in reversed(self.recent_alerts[-2:]):
            text = alert["text"]
            # Draw banner background
            cv2.rectangle(img, (15, y_offset - 20), (min(w - 15, len(text) * 11 + 35), y_offset + 5), (15, 15, 15), -1)
            cv2.rectangle(img, (15, y_offset - 20), (min(w - 15, len(text) * 11 + 35), y_offset + 5), alert["color"], 1)
            cv2.putText(img, text, (25, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert["color"], 1)
            y_offset -= 30


def run_standalone():
    """Runs the recognition attendance system as a standalone OpenCV application."""
    init_db()
    print("=" * 60)
    print(" FACE RECOGNITION ATTENDANCE SYSTEM - LIVE RECOGNITION")
    print(" Distance Threshold: 0.6 | Press 'q' or 'ESC' to exit")
    print("=" * 60)

    recognizer = AttendanceRecognizer(camera_index=0, distance_threshold=0.6, enable_liveness=True)
    recognizer._worker_loop()


if __name__ == "__main__":
    run_standalone()
