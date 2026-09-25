"""
main.py - Entry Point for Face Recognition Attendance System
Validates runtime environment, initializes SQLite database schema,
and launches the graphical desktop dashboard.
"""

import sys
import os

# Ensure local directory is on python import path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from database import init_db, DEFAULT_DB_PATH
import face_engine
from dashboard import AttendanceDashboard


def main():
    print("=" * 65)
    print("      FACE RECOGNITION ATTENDANCE MANAGEMENT SYSTEM")
    print("=" * 65)
    
    # 1. Initialize SQLite Database
    init_db(DEFAULT_DB_PATH)
    print(f"[OK] Database initialized: {DEFAULT_DB_PATH}")

    # 2. Check Face Recognition Engine
    engine_info = face_engine.get_engine_info()
    print(f"[OK] Active Vision Engine: {engine_info['backend']}")
    print(f"     Details: {engine_info['description']}")
    if not engine_info["is_dlib"]:
        print("     [Note] To enable dlib/face_recognition, see setup in README.md.")

    # 3. Launch Tkinter GUI Dashboard
    print("[OK] Launching Desktop GUI Dashboard...")
    app = AttendanceDashboard()
    app.mainloop()
    print("\nApplication closed. Thank you for using Face Recognition Attendance System!")


if __name__ == "__main__":
    main()
