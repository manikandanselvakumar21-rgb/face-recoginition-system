"""
database.py - Database Layer for Face Recognition Attendance System
Handles SQLite database connection, schema setup, student records,
attendance logging, duplicate prevention, and reporting queries.
"""

import sqlite3
import pickle
import os
from datetime import datetime, date
from contextlib import contextmanager
import pandas as pd
import numpy as np

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")


@contextmanager
def get_db(db_path: str = DEFAULT_DB_PATH):
    """
    Context manager that provides an open SQLite connection with Row factory
    and guarantees connection closure on exit to prevent file locking on Windows.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initializes the SQLite database schema if not already present.
    Creates:
      1. students table:
         - id: INTEGER PRIMARY KEY AUTOINCREMENT
         - name: TEXT NOT NULL
         - roll_no: TEXT UNIQUE NOT NULL
         - face_encoding: BLOB NOT NULL (128-dimensional embedding)
         - registered_on: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      2. attendance table:
         - id: INTEGER PRIMARY KEY AUTOINCREMENT
         - student_id: INTEGER NOT NULL, FOREIGN KEY -> students(id)
         - date: TEXT NOT NULL (YYYY-MM-DD)
         - time_in: TEXT NOT NULL (HH:MM:SS)
         - status: TEXT DEFAULT 'Present'
         - UNIQUE(student_id, date) constraint to prevent duplicate attendance on same day
    """
    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # Students Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                roll_no TEXT NOT NULL UNIQUE,
                face_encoding BLOB NOT NULL,
                registered_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Attendance Table with composite unique constraint (student_id, date)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                time_in TEXT NOT NULL,
                status TEXT DEFAULT 'Present',
                FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
                UNIQUE (student_id, date)
            )
        """)

        # Indexes for fast lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_students_roll_no ON students(roll_no)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_student_date ON attendance(student_id, date)")

        conn.commit()


def add_student(name: str, roll_no: str, face_encoding: np.ndarray, db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Registers a new student with their face encoding.
    
    Args:
        name: Full name of the student.
        roll_no: Unique roll number or identifier.
        face_encoding: 128-dimensional numpy array representing face features.
        db_path: Path to SQLite database.
        
    Returns:
        The inserted student id.
        
    Raises:
        ValueError: If roll number already exists or input is invalid.
    """
    clean_name = name.strip()
    clean_roll = roll_no.strip()
    
    if not clean_name:
        raise ValueError("Student name cannot be empty.")
    if not clean_roll:
        raise ValueError("Roll number cannot be empty.")
    if face_encoding is None or not isinstance(face_encoding, np.ndarray):
        raise ValueError("Valid face encoding numpy array is required.")

    # Serialize face encoding to BLOB using pickle
    encoding_blob = pickle.dumps(face_encoding.astype(np.float64))

    try:
        with get_db(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO students (name, roll_no, face_encoding) VALUES (?, ?, ?)",
                (clean_name, clean_roll, encoding_blob)
            )
            conn.commit()
            return cursor.lastrowid
    except sqlite3.IntegrityError as e:
        if "UNIQUE constraint failed: students.roll_no" in str(e) or "roll_no" in str(e):
            raise ValueError(f"Student with roll number '{clean_roll}' is already registered.")
        raise e


def get_all_students(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """
    Retrieves all registered students, deserializing the face encodings.
    
    Returns:
        List of dictionaries with student records:
        [{'id': int, 'name': str, 'roll_no': str, 'face_encoding': np.ndarray, 'registered_on': str}, ...]
    """
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, roll_no, face_encoding, registered_on FROM students ORDER BY name ASC")
        rows = cursor.fetchall()
        
        students = []
        for r in rows:
            encoding = pickle.loads(r["face_encoding"])
            students.append({
                "id": r["id"],
                "name": r["name"],
                "roll_no": r["roll_no"],
                "face_encoding": encoding,
                "registered_on": r["registered_on"]
            })
        return students


def get_student_by_roll_no(roll_no: str, db_path: str = DEFAULT_DB_PATH) -> dict | None:
    """Retrieves a student record by roll number."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, roll_no, face_encoding, registered_on FROM students WHERE roll_no = ?", (roll_no.strip(),))
        row = cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "name": row["name"],
                "roll_no": row["roll_no"],
                "face_encoding": pickle.loads(row["face_encoding"]),
                "registered_on": row["registered_on"]
            }
        return None


def get_student_by_id(student_id: int, db_path: str = DEFAULT_DB_PATH) -> dict | None:
    """Retrieves a student record by ID."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, roll_no, face_encoding, registered_on FROM students WHERE id = ?", (student_id,))
        row = cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "name": row["name"],
                "roll_no": row["roll_no"],
                "face_encoding": pickle.loads(row["face_encoding"]),
                "registered_on": row["registered_on"]
            }
        return None


def is_attendance_marked(student_id: int, date_str: str | None = None, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Checks whether attendance has already been logged for student on the given date."""
    if date_str is None:
        date_str = date.today().isoformat()
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM attendance WHERE student_id = ? AND date = ?",
            (student_id, date_str)
        )
        return cursor.fetchone() is not None


def mark_attendance(
    student_id: int,
    date_str: str | None = None,
    time_str: str | None = None,
    status: str = "Present",
    db_path: str = DEFAULT_DB_PATH
) -> tuple[bool, str]:
    """
    Logs attendance for a student, strictly preventing duplicate entries for the same date.
    
    Args:
        student_id: ID of the student from students table.
        date_str: Date string in 'YYYY-MM-DD' format (default: today).
        time_str: Time string in 'HH:MM:SS' format (default: now).
        status: Attendance status (e.g. 'Present', 'Late').
        db_path: Database path.
        
    Returns:
        tuple (success: bool, message: str)
        - (True, "Attendance marked successfully.") if newly recorded
        - (False, "Attendance already marked for today.") if duplicate
    """
    now = datetime.now()
    if date_str is None:
        date_str = now.strftime("%Y-%m-%d")
    if time_str is None:
        time_str = now.strftime("%H:%M:%S")

    try:
        with get_db(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO attendance (student_id, date, time_in, status) VALUES (?, ?, ?, ?)",
                (student_id, date_str, time_str, status)
            )
            conn.commit()
            return True, "Attendance marked successfully."
    except sqlite3.IntegrityError:
        return False, "Attendance already marked for today."


def get_today_attendance(date_str: str | None = None, db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """
    Returns attendance records for today (or specified date), joined with student details.
    
    Returns:
        List of dicts: [{'attendance_id': int, 'student_id': int, 'name': str, 'roll_no': str, 'date': str, 'time_in': str, 'status': str}, ...]
    """
    if date_str is None:
        date_str = date.today().isoformat()

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                a.id as attendance_id,
                s.id as student_id,
                s.name,
                s.roll_no,
                a.date,
                a.time_in,
                a.status
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.date = ?
            ORDER BY a.time_in DESC
        """, (date_str,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_attendance_report(
    start_date: str | None = None,
    end_date: str | None = None,
    db_path: str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """
    Retrieves attendance records filtered by date range and returns a Pandas DataFrame.
    
    Args:
        start_date: 'YYYY-MM-DD' (inclusive)
        end_date: 'YYYY-MM-DD' (inclusive)
        db_path: Database path
        
    Returns:
        pd.DataFrame with columns: ['Date', 'Time In', 'Roll Number', 'Student Name', 'Status']
    """
    query = """
        SELECT 
            a.date as "Date",
            a.time_in as "Time In",
            s.roll_no as "Roll Number",
            s.name as "Student Name",
            a.status as "Status"
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE 1=1
    """
    params = []
    if start_date:
        query += " AND a.date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND a.date <= ?"
        params.append(end_date)
        
    query += " ORDER BY a.date DESC, a.time_in DESC"

    with get_db(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
        return df


def get_attendance_summary(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Calculates overall attendance summary per student (Stretch Goal):
    - Total working/recorded attendance days across system
    - Days present for each student
    - Attendance percentage: (Days Present / Total Days) * 100
    
    Returns:
        pd.DataFrame with summary columns.
    """
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT date) as total_days FROM attendance")
        row = cursor.fetchone()
        total_days = row["total_days"] if row and row["total_days"] else 0

        query = """
            SELECT 
                s.id as student_id,
                s.roll_no as "Roll Number",
                s.name as "Student Name",
                s.registered_on as "Registered Date",
                COUNT(a.id) as "Days Present"
            FROM students s
            LEFT JOIN attendance a ON s.id = a.student_id
            GROUP BY s.id, s.roll_no, s.name, s.registered_on
            ORDER BY s.roll_no ASC
        """
        df = pd.read_sql_query(query, conn)
        df["Total Days"] = total_days
        
        # Calculate attendance percentage
        if total_days > 0:
            df["Attendance %"] = ((df["Days Present"] / total_days) * 100).round(1)
        else:
            df["Attendance %"] = 0.0

        return df


def delete_student(student_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Deletes a student and their attendance history."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()
        return cursor.rowcount > 0


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DEFAULT_DB_PATH)
