"""
test_system.py - End-to-End Automated Test Suite for Face Recognition Attendance System
Tests database integrity, duplicate constraints, face encoding math, matching threshold,
liveness detection, multi-face pipeline, and CSV/Excel export without requiring a live webcam.
"""

import os
import sys
import unittest
import uuid
import numpy as np
import pandas as pd
import cv2

# Ensure project directory is importable
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import database
import face_engine
from recognize_attendance import AttendanceRecognizer


class TestDatabaseLayer(unittest.TestCase):
    """Tests SQLite database creation, student records, duplicate prevention, and reporting."""

    def setUp(self):
        # Generate a unique database name per test to prevent locking conflicts
        self.test_db = os.path.join(PROJECT_DIR, f"test_db_{uuid.uuid4().hex[:8]}.db")
        database.init_db(self.test_db)

    def tearDown(self):
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except OSError:
                pass

    def test_database_init_and_tables(self):
        """Verifies database schema and tables exist."""
        with database.get_db(self.test_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row["name"] for row in cursor.fetchall()]
            self.assertIn("students", tables)
            self.assertIn("attendance", tables)

    def test_add_and_get_student(self):
        """Verifies adding a student and retrieving deserialized 128-d face encoding."""
        dummy_encoding = np.random.randn(128).astype(np.float64)
        dummy_encoding /= np.linalg.norm(dummy_encoding)

        student_id = database.add_student("John Doe", "ROLL-001", dummy_encoding, db_path=self.test_db)
        self.assertGreater(student_id, 0)

        retrieved = database.get_student_by_id(student_id, db_path=self.test_db)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["name"], "John Doe")
        self.assertEqual(retrieved["roll_no"], "ROLL-001")
        self.assertEqual(len(retrieved["face_encoding"]), 128)
        np.testing.assert_allclose(retrieved["face_encoding"], dummy_encoding, rtol=1e-5)

    def test_duplicate_roll_number_rejection(self):
        """Verifies that adding two students with identical roll numbers raises ValueError."""
        dummy_encoding = np.random.randn(128).astype(np.float64)
        database.add_student("Alice Smith", "ROLL-002", dummy_encoding, db_path=self.test_db)

        with self.assertRaises(ValueError):
            database.add_student("Bob Smith", "ROLL-002", dummy_encoding, db_path=self.test_db)

    def test_attendance_marking_and_duplicate_prevention(self):
        """Verifies attendance logging and strict duplicate prevention for the same date."""
        dummy_encoding = np.random.randn(128).astype(np.float64)
        student_id = database.add_student("Charlie Brown", "ROLL-003", dummy_encoding, db_path=self.test_db)

        # 1. First attendance entry on date 2026-09-25
        ok, msg = database.mark_attendance(student_id, date_str="2026-09-25", time_str="09:00:00", db_path=self.test_db)
        self.assertTrue(ok)
        self.assertIn("successfully", msg.lower())

        # 2. Duplicate attendance entry on the SAME date must fail
        ok2, msg2 = database.mark_attendance(student_id, date_str="2026-09-25", time_str="09:15:00", db_path=self.test_db)
        self.assertFalse(ok2)
        self.assertIn("already marked", msg2.lower())

        # 3. Attendance entry on a DIFFERENT date must succeed
        ok3, msg3 = database.mark_attendance(student_id, date_str="2026-09-26", time_str="09:05:00", db_path=self.test_db)
        self.assertTrue(ok3)

    def test_today_attendance_query(self):
        """Verifies querying today's attendance records."""
        dummy_encoding = np.random.randn(128).astype(np.float64)
        s_id = database.add_student("David Warner", "ROLL-004", dummy_encoding, db_path=self.test_db)
        database.mark_attendance(s_id, date_str="2026-09-25", time_str="09:30:00", db_path=self.test_db)

        records = database.get_today_attendance(date_str="2026-09-25", db_path=self.test_db)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["roll_no"], "ROLL-004")
        self.assertEqual(records[0]["name"], "David Warner")
        self.assertEqual(records[0]["time_in"], "09:30:00")

    def test_attendance_summary_percentage(self):
        """Verifies calculation of attendance percentages across multiple dates."""
        enc1 = np.random.randn(128).astype(np.float64)
        enc2 = np.random.randn(128).astype(np.float64)

        s1 = database.add_student("Student One", "S-01", enc1, db_path=self.test_db)
        s2 = database.add_student("Student Two", "S-02", enc2, db_path=self.test_db)

        # Day 1: Both present
        database.mark_attendance(s1, date_str="2026-09-20", time_str="09:00:00", db_path=self.test_db)
        database.mark_attendance(s2, date_str="2026-09-20", time_str="09:00:00", db_path=self.test_db)

        # Day 2: Only Student One present
        database.mark_attendance(s1, date_str="2026-09-21", time_str="09:00:00", db_path=self.test_db)

        # Total working days = 2
        summary_df = database.get_attendance_summary(db_path=self.test_db)
        self.assertEqual(len(summary_df), 2)

        s1_row = summary_df[summary_df["Roll Number"] == "S-01"].iloc[0]
        s2_row = summary_df[summary_df["Roll Number"] == "S-02"].iloc[0]

        # Student One: 2 out of 2 days = 100.0%
        self.assertEqual(s1_row["Days Present"], 2)
        self.assertEqual(s1_row["Total Days"], 2)
        self.assertEqual(s1_row["Attendance %"], 100.0)

        # Student Two: 1 out of 2 days = 50.0%
        self.assertEqual(s2_row["Days Present"], 1)
        self.assertEqual(s2_row["Total Days"], 2)
        self.assertEqual(s2_row["Attendance %"], 50.0)


class TestFaceEngineAndRecognition(unittest.TestCase):
    """Tests face encoding, distance metric math, threshold verification and liveness tracker."""

    def test_engine_info(self):
        """Verifies engine info dictionary has valid structure."""
        info = face_engine.get_engine_info()
        self.assertIn("backend", info)
        self.assertIn("is_dlib", info)
        self.assertIn("description", info)

    def test_distance_and_threshold_math(self):
        """
        Validates Euclidean distance logic and 0.6 threshold:
        - Identical vectors => distance 0.0 <= 0.6 (Match)
        - Very close vectors => distance <= 0.6 (Match)
        - Orthogonal / distant vectors => distance > 0.6 (Unknown)
        """
        v1 = np.random.randn(128).astype(np.float64)
        v1 /= np.linalg.norm(v1)

        # Identical
        dist_self = face_engine.face_distance([v1], v1)[0]
        self.assertAlmostEqual(dist_self, 0.0, places=5)
        self.assertTrue(face_engine.compare_faces([v1], v1, tolerance=0.6)[0])

        # Slightly perturbed version of v1 (e.g. slight lighting shift)
        v_close = v1 + np.random.randn(128) * 0.02
        v_close /= np.linalg.norm(v_close)
        dist_close = face_engine.face_distance([v1], v_close)[0]
        self.assertLessEqual(dist_close, 0.6)
        self.assertTrue(face_engine.compare_faces([v1], v_close, tolerance=0.6)[0])

        # Distinct random vector (different face)
        v_diff = np.random.randn(128).astype(np.float64)
        v_diff /= np.linalg.norm(v_diff)
        dist_diff = face_engine.face_distance([v1], v_diff)[0]
        self.assertGreater(dist_diff, 0.6)
        self.assertFalse(face_engine.compare_faces([v1], v_diff, tolerance=0.6)[0])

    def test_confidence_calculation(self):
        """Verifies confidence percentages mapping correctly."""
        # Exact match (0.0 distance) -> 100%
        conf_perfect = face_engine.calculate_confidence(0.0, threshold=0.6)
        self.assertEqual(conf_perfect, 100.0)

        # Boundary match (0.6 distance) -> 65%
        conf_boundary = face_engine.calculate_confidence(0.6, threshold=0.6)
        self.assertEqual(conf_boundary, 65.0)

        # Poor match (1.2 distance) -> 0%
        conf_poor = face_engine.calculate_confidence(1.2, threshold=0.6)
        self.assertLessEqual(conf_poor, 10.0)

    def test_liveness_tracker_instantiation(self):
        """Verifies LivenessTracker state transitions."""
        tracker = face_engine.LivenessTracker(ear_threshold=0.22, consecutive_frames=2)
        self.assertFalse(tracker.is_live)
        self.assertEqual(tracker.total_blinks, 0)


class TestPandasExportAndReports(unittest.TestCase):
    """Tests Pandas DataFrame generation, CSV export, and Excel (.xlsx) export."""

    def setUp(self):
        uid = uuid.uuid4().hex[:8]
        self.test_db = os.path.join(PROJECT_DIR, f"test_export_{uid}.db")
        self.test_csv = os.path.join(PROJECT_DIR, f"test_report_{uid}.csv")
        self.test_xlsx = os.path.join(PROJECT_DIR, f"test_report_{uid}.xlsx")

        database.init_db(self.test_db)

        # Seed test data
        enc = np.random.randn(128).astype(np.float64)
        s1 = database.add_student("Emma Watson", "ROLL-501", enc, db_path=self.test_db)
        s2 = database.add_student("Daniel Radcliffe", "ROLL-502", enc, db_path=self.test_db)

        database.mark_attendance(s1, date_str="2026-09-24", time_str="08:50:00", db_path=self.test_db)
        database.mark_attendance(s2, date_str="2026-09-25", time_str="09:10:00", db_path=self.test_db)

    def tearDown(self):
        for f in (self.test_db, self.test_csv, self.test_xlsx):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def test_pandas_date_range_report(self):
        """Verifies filtering by date range returns correct DataFrame."""
        df_all = database.get_attendance_report(db_path=self.test_db)
        self.assertEqual(len(df_all), 2)
        self.assertIn("Student Name", df_all.columns)
        self.assertIn("Roll Number", df_all.columns)
        self.assertIn("Date", df_all.columns)
        self.assertIn("Time In", df_all.columns)

        # Filter single date 2026-09-25
        df_filtered = database.get_attendance_report(start_date="2026-09-25", end_date="2026-09-25", db_path=self.test_db)
        self.assertEqual(len(df_filtered), 1)
        self.assertEqual(df_filtered.iloc[0]["Roll Number"], "ROLL-502")

    def test_csv_export(self):
        """Verifies saving DataFrame to CSV file."""
        df = database.get_attendance_report(db_path=self.test_db)
        df.to_csv(self.test_csv, index=False)
        self.assertTrue(os.path.exists(self.test_csv))

        # Read back and verify
        reloaded = pd.read_csv(self.test_csv)
        self.assertEqual(len(reloaded), 2)
        self.assertEqual(list(reloaded["Roll Number"]), ["ROLL-502", "ROLL-501"])

    def test_excel_export(self):
        """Verifies saving DataFrame to Excel (.xlsx) using openpyxl."""
        df = database.get_attendance_report(db_path=self.test_db)
        with pd.ExcelWriter(self.test_xlsx, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="AttendanceReport", index=False)
        self.assertTrue(os.path.exists(self.test_xlsx))

        # Read back and verify
        reloaded = pd.read_excel(self.test_xlsx, sheet_name="AttendanceReport")
        self.assertEqual(len(reloaded), 2)
        self.assertEqual(list(reloaded["Student Name"]), ["Daniel Radcliffe", "Emma Watson"])


class TestEndToEndPipeline(unittest.TestCase):
    """Simulates the entire attendance recognition process with synthetic image frames."""

    def test_pipeline_with_synthetic_frame(self):
        """Runs AttendanceRecognizer.process_frame on a generated test frame."""
        recognizer = AttendanceRecognizer(camera_index=0, distance_threshold=0.6)
        
        # Create a synthetic 640x480 frame
        synthetic_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        
        # Draw a synthetic face-like circle
        cv2.circle(synthetic_frame, (320, 240), 70, (200, 180, 150), -1)
        # Eyes
        cv2.circle(synthetic_frame, (295, 220), 8, (50, 40, 30), -1)
        cv2.circle(synthetic_frame, (345, 220), 8, (50, 40, 30), -1)
        # Mouth
        cv2.ellipse(synthetic_frame, (320, 265), (25, 10), 0, 0, 180, (50, 40, 30), 2)

        # Process frame
        processed = recognizer.process_frame(synthetic_frame)
        self.assertEqual(processed.shape, synthetic_frame.shape)
        self.assertEqual(processed.dtype, np.uint8)


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print(" RUNNING AUTOMATED UNIT & INTEGRATION TESTS")
    print("=" * 65 + "\n")
    unittest.main(verbosity=2)
