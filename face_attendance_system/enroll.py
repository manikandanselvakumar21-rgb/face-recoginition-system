"""
enroll.py - Student Face Enrollment Module
Captures 5 face images of a new student via webcam, computes a stable
normalized average 128-dimensional face encoding, and saves the student
details to the SQLite database.
"""

import sys
import time
import argparse
import numpy as np
import cv2

from database import add_student, init_db, get_student_by_roll_no
import face_engine


class EnrollmentManager:
    """
    Manages capturing multiple face samples and enrolling a student into the database.
    Supports both standalone interactive OpenCV UI and GUI callbacks.
    """
    def __init__(self, camera_index: int = 0, samples_needed: int = 5):
        self.camera_index = camera_index
        self.samples_needed = samples_needed

    def capture_and_enroll(
        self,
        name: str,
        roll_no: str,
        camera_index: int = 0,
        frame_callback=None,
        progress_callback=None,
        status_callback=None
    ) -> tuple[bool, str]:
        """
        Captures 5 face samples from webcam, computes the average encoding,
        and registers the student into the database.
        
        Args:
            name: Student full name.
            roll_no: Student unique roll number.
            camera_index: OpenCV VideoCapture device index.
            frame_callback: Optional callable(frame_bgr, prompt_text) for GUI video rendering.
            progress_callback: Optional callable(current_samples, total_samples).
            status_callback: Optional callable(message_str).
            
        Returns:
            (success: bool, message: str)
        """
        clean_name = name.strip()
        clean_roll = roll_no.strip()
        
        if not clean_name:
            return False, "Error: Student name cannot be empty."
        if not clean_roll:
            return False, "Error: Roll number cannot be empty."

        # Pre-check duplicate roll number in database
        existing = get_student_by_roll_no(clean_roll)
        if existing:
            return False, f"Duplicate enrollment: Roll number '{clean_roll}' is already registered to '{existing['name']}'."

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return False, f"Camera error: Unable to open webcam at index {camera_index}. Please check connection or permissions."

        # Optimize camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        encodings: list[np.ndarray] = []
        last_capture_time = 0.0
        capture_delay_sec = 0.8  # Brief interval between auto-captures for angle diversity
        
        info_message = "Look at the camera. Capturing face samples..."
        if status_callback:
            status_callback(info_message)

        try:
            while len(encodings) < self.samples_needed:
                ret, frame = cap.read()
                if not ret or frame is None:
                    return False, "Camera stream error: Failed to receive frame from webcam."

                # Horizontal flip for natural selfie mirror effect
                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()

                # Detect faces in frame
                faces = face_engine.detect_faces(frame)
                now = time.time()

                if len(faces) == 0:
                    status_text = "No face detected. Please look directly at the camera."
                    box_color = (0, 165, 255)  # Orange
                elif len(faces) > 1:
                    status_text = f"Multiple faces ({len(faces)}) detected! Please ensure only one person is in view."
                    box_color = (0, 0, 255)  # Red
                else:
                    # Exactly one face detected
                    top, right, bottom, left = faces[0]
                    box_color = (0, 255, 0)  # Green

                    # Draw bounding box
                    cv2.rectangle(display_frame, (left, top), (right, bottom), box_color, 2)

                    # Auto-capture if delay elapsed
                    if now - last_capture_time >= capture_delay_sec:
                        sample_encs = face_engine.encode_faces(frame, [faces[0]])
                        if sample_encs and len(sample_encs) > 0:
                            encodings.append(sample_encs[0])
                            last_capture_time = now
                            if progress_callback:
                                progress_callback(len(encodings), self.samples_needed)

                    status_text = f"Capturing sample {len(encodings)} of {self.samples_needed}..."

                # Render UI banner on display frame
                h, w = display_frame.shape[:2]
                cv2.rectangle(display_frame, (0, 0), (w, 65), (25, 25, 25), -1)
                cv2.putText(
                    display_frame,
                    f"Enrolling: {clean_name} ({clean_roll})",
                    (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    display_frame,
                    status_text,
                    (15, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    1
                )

                # Progress bar at bottom
                progress_ratio = len(encodings) / self.samples_needed
                cv2.rectangle(display_frame, (0, h - 14), (w, h), (40, 40, 40), -1)
                cv2.rectangle(display_frame, (0, h - 14), (int(w * progress_ratio), h), (0, 200, 0), -1)

                if frame_callback:
                    frame_callback(display_frame, status_text)
                else:
                    # Default standalone OpenCV window
                    cv2.imshow("Face Enrollment - Press 'q' to cancel", display_frame)
                    key = cv2.waitKey(20) & 0xFF
                    if key == ord('q') or key == 27:  # ESC or q
                        return False, "Enrollment was cancelled by user."

        finally:
            cap.release()
            if not frame_callback:
                cv2.destroyAllWindows()

        if len(encodings) < self.samples_needed:
            return False, f"Enrollment incomplete: only {len(encodings)} of {self.samples_needed} samples collected."

        # Compute robust average face encoding and normalize
        avg_encoding = np.mean(encodings, axis=0)
        norm = np.linalg.norm(avg_encoding)
        if norm > 1e-6:
            avg_encoding = avg_encoding / norm

        try:
            student_id = add_student(clean_name, clean_roll, avg_encoding)
            msg = f"Successfully enrolled student '{clean_name}' (Roll No: {clean_roll}, ID: {student_id}) with {len(encodings)} face samples!"
            if status_callback:
                status_callback(msg)
            return True, msg
        except ValueError as e:
            return False, str(e)


def run_enrollment_cli():
    """Runs interactive command line student enrollment."""
    init_db()
    engine_meta = face_engine.get_engine_info()
    print("=" * 60)
    print(" FACE RECOGNITION ATTENDANCE SYSTEM - STUDENT ENROLLMENT")
    print(" Engine Active:", engine_meta["backend"])
    print("=" * 60)

    try:
        name = input("Enter Student Full Name: ").strip()
        while not name:
            name = input("Name cannot be empty. Please enter Student Full Name: ").strip()

        roll_no = input("Enter Student Roll Number: ").strip()
        while not roll_no:
            roll_no = input("Roll number cannot be empty. Please enter Roll Number: ").strip()

        print("\nStarting camera. Look directly into the webcam...")
        print("Keep your face inside the green box until 5 samples are captured.\n")

        manager = EnrollmentManager(camera_index=0, samples_needed=5)
        success, message = manager.capture_and_enroll(name, roll_no)

        if success:
            print("\n[SUCCESS]", message)
        else:
            print("\n[FAILED]", message)

    except KeyboardInterrupt:
        print("\nEnrollment aborted by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a new student into the Face Recognition Attendance System.")
    parser.add_argument("--name", type=str, help="Student full name")
    parser.add_argument("--roll", type=str, help="Student roll number")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    init_db()
    if args.name and args.roll:
        mgr = EnrollmentManager(camera_index=args.camera)
        ok, msg = mgr.capture_and_enroll(args.name, args.roll, camera_index=args.camera)
        print(msg)
        sys.exit(0 if ok else 1)
    else:
        run_enrollment_cli()
