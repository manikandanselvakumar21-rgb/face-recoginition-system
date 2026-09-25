"""
dashboard.py - Tkinter Desktop GUI Dashboard for Face Recognition Attendance System
Features:
  - Live embedded camera feed with Start / Stop attendance recognition
  - Interactive Student Enrollment wizard (5-sample capture with preview)
  - Live 'Today's Attendance' table with real-time updates and search filtering
  - Export attendance reports to Excel (.xlsx) and CSV (.csv) filtered by date range
  - Student Attendance Percentage Summary & Analytics (Stretch Goal)
  - Engine status, clock, and error handling
"""

import sys
import os
import time
import threading
from datetime import datetime, date, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import pandas as pd

from database import (
    init_db,
    get_all_students,
    get_today_attendance,
    get_attendance_report,
    get_attendance_summary,
    get_student_by_roll_no,
    add_student,
    DEFAULT_DB_PATH
)
import face_engine
from recognize_attendance import AttendanceRecognizer
from enroll import EnrollmentManager


class AttendanceDashboard(tk.Tk):
    def __init__(self):
        super().__init__()

        # Ensure database is created
        init_db()

        self.title("Face Recognition Attendance System")
        self.geometry("1240x750")
        self.minsize(1050, 680)

        # Style and Theme Configuration
        self._configure_styles()

        # State Variables
        self.recognizer = AttendanceRecognizer(camera_index=0, distance_threshold=0.6, enable_liveness=True)
        self.current_tk_image = None
        self.is_camera_running = False

        # Build UI Components
        self._build_header()
        self._build_main_layout()
        self._build_status_bar()

        # Start Clock and Auto-Refresh loops
        self._update_clock()
        self.refresh_today_attendance()
        self.refresh_summary()

        # Protocol for clean shutdown
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_styles(self):
        """Sets up ttk styles and modern color palette."""
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        # Color constants
        self.BG_MAIN = "#1e1e24"
        self.BG_CARD = "#2b2d42"
        self.ACCENT_PRIMARY = "#4361ee"
        self.ACCENT_SUCCESS = "#2ec4b6"
        self.ACCENT_DANGER = "#e63946"
        self.ACCENT_WARNING = "#ffb703"
        self.TEXT_MAIN = "#f8f9fa"
        self.TEXT_MUTED = "#adb5bd"

        self.configure(bg=self.BG_MAIN)

        # Custom ttk styles
        self.style.configure(".", background=self.BG_MAIN, foreground=self.TEXT_MAIN, font=("Segoe UI", 10))
        self.style.configure("TFrame", background=self.BG_MAIN)
        self.style.configure("Card.TFrame", background=self.BG_CARD, relief="flat")
        self.style.configure("TLabel", background=self.BG_MAIN, foreground=self.TEXT_MAIN)
        self.style.configure("Card.TLabel", background=self.BG_CARD, foreground=self.TEXT_MAIN)
        self.style.configure("Muted.TLabel", background=self.BG_CARD, foreground=self.TEXT_MUTED, font=("Segoe UI", 9))
        self.style.configure("Title.TLabel", font=("Segoe UI", 15, "bold"), foreground=self.TEXT_MAIN)

        # Buttons
        self.style.configure(
            "Primary.TButton",
            font=("Segoe UI", 10, "bold"),
            background=self.ACCENT_PRIMARY,
            foreground="white",
            padding=(12, 6)
        )
        self.style.map("Primary.TButton", background=[("active", "#3a56d4")])

        self.style.configure(
            "Success.TButton",
            font=("Segoe UI", 10, "bold"),
            background=self.ACCENT_SUCCESS,
            foreground="black",
            padding=(12, 6)
        )
        self.style.map("Success.TButton", background=[("active", "#25a89c")])

        self.style.configure(
            "Danger.TButton",
            font=("Segoe UI", 10, "bold"),
            background=self.ACCENT_DANGER,
            foreground="white",
            padding=(12, 6)
        )
        self.style.map("Danger.TButton", background=[("active", "#c92a37")])

        # Notebook tabs
        self.style.configure("TNotebook", background=self.BG_MAIN)
        self.style.configure("TNotebook.Tab", background="#3a3d52", foreground="white", padding=(14, 6), font=("Segoe UI", 10, "bold"))
        self.style.map("TNotebook.Tab", background=[("selected", self.BG_CARD)], foreground=[("selected", self.ACCENT_PRIMARY)])

        # Treeview table
        self.style.configure(
            "Treeview",
            background=self.BG_CARD,
            foreground=self.TEXT_MAIN,
            fieldbackground=self.BG_CARD,
            rowheight=28,
            font=("Segoe UI", 9)
        )
        self.style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background="#383b54", foreground="white")
        self.style.map("Treeview", background=[("selected", self.ACCENT_PRIMARY)])

    def _build_header(self):
        """Constructs top header bar with branding, engine info and live clock."""
        header_frame = tk.Frame(self, bg="#161822", height=65, padx=20, pady=10)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        # Title & Subtitle
        title_box = tk.Frame(header_frame, bg="#161822")
        title_box.pack(side=tk.LEFT)

        app_title = tk.Label(
            title_box,
            text="Face Recognition Attendance System",
            font=("Segoe UI", 15, "bold"),
            fg="white",
            bg="#161822"
        )
        app_title.pack(anchor="w")

        engine_info = face_engine.get_engine_info()
        engine_label = tk.Label(
            title_box,
            text=f"Engine: {engine_info['backend']}  |  DB: attendance.db",
            font=("Segoe UI", 9),
            fg=self.ACCENT_SUCCESS if engine_info["is_dlib"] else self.ACCENT_WARNING,
            bg="#161822"
        )
        engine_label.pack(anchor="w")

        # Right side: Live Digital Clock & Enrolled Counter
        info_box = tk.Frame(header_frame, bg="#161822")
        info_box.pack(side=tk.RIGHT)

        self.clock_label = tk.Label(
            info_box,
            text="",
            font=("Segoe UI", 12, "bold"),
            fg="#e2e8f0",
            bg="#161822"
        )
        self.clock_label.pack(anchor="e")

        self.enrolled_count_label = tk.Label(
            info_box,
            text="Registered Students: 0",
            font=("Segoe UI", 9),
            fg=self.TEXT_MUTED,
            bg="#161822"
        )
        self.enrolled_count_label.pack(anchor="e")

    def _build_main_layout(self):
        """Constructs the two-column layout: Camera on Left, Tabs on Right."""
        main_container = tk.Frame(self, bg=self.BG_MAIN, padx=15, pady=15)
        main_container.pack(fill=tk.BOTH, expand=True)

        # ----------------- Left Panel: Live Camera & Actions -----------------
        left_panel = ttk.Frame(main_container, style="Card.TFrame", padding=12)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        # Panel Title
        cam_title_frame = tk.Frame(left_panel, bg=self.BG_CARD)
        cam_title_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            cam_title_frame,
            text="Live Scanner Feed",
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg=self.BG_CARD
        ).pack(side=tk.LEFT)

        self.cam_badge = tk.Label(
            cam_title_frame,
            text="● OFF",
            font=("Segoe UI", 9, "bold"),
            fg=self.ACCENT_DANGER,
            bg=self.BG_CARD
        )
        self.cam_badge.pack(side=tk.RIGHT)

        # Video Canvas (560x420 standard preview)
        self.video_canvas = tk.Canvas(left_panel, width=540, height=405, bg="#0d0e15", highlightthickness=1, highlightbackground="#3d405b")
        self.video_canvas.pack(fill=tk.BOTH, expand=False)
        self._draw_placeholder_canvas("Camera Standby\nClick 'Start Scanner' to Begin")

        # Action Buttons Row
        btn_frame = tk.Frame(left_panel, bg=self.BG_CARD)
        btn_frame.pack(fill=tk.X, pady=(12, 6))

        self.btn_toggle_cam = ttk.Button(
            btn_frame,
            text="▶ Start Scanner",
            style="Success.TButton",
            command=self.toggle_scanner
        )
        self.btn_toggle_cam.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.btn_enroll = ttk.Button(
            btn_frame,
            text="👤 Enroll New Student",
            style="Primary.TButton",
            command=self.open_enrollment_dialog
        )
        self.btn_enroll.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(6, 0))

        # Recognition Alert Card
        self.alert_card = tk.Frame(left_panel, bg="#1a1c29", padx=10, pady=8, relief="groove", bd=1)
        self.alert_card.pack(fill=tk.X, pady=(8, 0))

        self.lbl_alert_title = tk.Label(
            self.alert_card,
            text="Recent Recognition Event:",
            font=("Segoe UI", 9, "bold"),
            fg=self.TEXT_MUTED,
            bg="#1a1c29"
        )
        self.lbl_alert_title.pack(anchor="w")

        self.lbl_alert_msg = tk.Label(
            self.alert_card,
            text="Scanner idle. Waiting for recognized faces...",
            font=("Segoe UI", 10),
            fg="#94d2bd",
            bg="#1a1c29",
            wraplength=500,
            justify="left"
        )
        self.lbl_alert_msg.pack(anchor="w", pady=(2, 0))

        # ----------------- Right Panel: Notebook Tabs -----------------
        right_panel = tk.Frame(main_container, bg=self.BG_MAIN)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Today's Attendance
        self.tab_today = ttk.Frame(self.notebook, style="Card.TFrame", padding=10)
        self.notebook.add(self.tab_today, text="  Today's Attendance  ")
        self._build_today_tab()

        # Tab 2: Attendance Reports & Export
        self.tab_reports = ttk.Frame(self.notebook, style="Card.TFrame", padding=10)
        self.notebook.add(self.tab_reports, text="  Reports & Export  ")
        self._build_reports_tab()

        # Tab 3: Attendance Analytics & Summary (Stretch Goal)
        self.tab_summary = ttk.Frame(self.notebook, style="Card.TFrame", padding=10)
        self.notebook.add(self.tab_summary, text="  Student Summary (%)  ")
        self._build_summary_tab()

    def _draw_placeholder_canvas(self, text: str):
        """Draws a clean placeholder message on video canvas when camera is off."""
        self.video_canvas.delete("all")
        w = int(self.video_canvas.cget("width"))
        h = int(self.video_canvas.cget("height"))
        self.video_canvas.create_rectangle(0, 0, w, h, fill="#0d0e15", outline="")
        self.video_canvas.create_text(
            w // 2, h // 2,
            text=text,
            fill="#6c757d",
            font=("Segoe UI", 13, "bold"),
            justify="center"
        )

    # -------------------------------------------------------------------------
    # Tab 1: Today's Attendance
    # -------------------------------------------------------------------------
    def _build_today_tab(self):
        # Top toolbar
        toolbar = tk.Frame(self.tab_today, bg=self.BG_CARD)
        toolbar.pack(fill=tk.X, pady=(0, 8))

        tk.Label(toolbar, text="Search:", font=("Segoe UI", 9, "bold"), bg=self.BG_CARD, fg=self.TEXT_MUTED).pack(side=tk.LEFT)
        self.today_search_var = tk.StringVar()
        self.today_search_var.trace_add("write", lambda *args: self._filter_today_table())
        ent_search = tk.Entry(toolbar, textvariable=self.today_search_var, bg="#1a1c29", fg="white", insertbackground="white", width=22)
        ent_search.pack(side=tk.LEFT, padx=6)

        self.lbl_today_count = tk.Label(toolbar, text="Total Present Today: 0", font=("Segoe UI", 9, "bold"), fg=self.ACCENT_SUCCESS, bg=self.BG_CARD)
        self.lbl_today_count.pack(side=tk.LEFT, padx=12)

        btn_refresh = ttk.Button(toolbar, text="⟳ Refresh", command=self.refresh_today_attendance)
        btn_refresh.pack(side=tk.RIGHT)

        # Treeview Table
        tree_frame = tk.Frame(self.tab_today, bg=self.BG_CARD)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("sno", "roll_no", "name", "time_in", "status")
        self.tree_today = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")
        
        self.tree_today.heading("sno", text="#")
        self.tree_today.heading("roll_no", text="Roll No")
        self.tree_today.heading("name", text="Student Name")
        self.tree_today.heading("time_in", text="Time In")
        self.tree_today.heading("status", text="Status")

        self.tree_today.column("sno", width=40, anchor="center")
        self.tree_today.column("roll_no", width=110, anchor="w")
        self.tree_today.column("name", width=180, anchor="w")
        self.tree_today.column("time_in", width=100, anchor="center")
        self.tree_today.column("status", width=90, anchor="center")

        scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree_today.yview)
        self.tree_today.configure(yscrollcommand=scroll_y.set)

        self.tree_today.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.all_today_rows = []

    def refresh_today_attendance(self):
        """Fetches today's attendance records from SQLite and updates the table."""
        today_records = get_today_attendance()
        self.all_today_rows = today_records
        self._filter_today_table()
        self.lbl_today_count.config(text=f"Total Present Today: {len(today_records)}")

        # Update registered students counter
        all_students = get_all_students()
        self.enrolled_count_label.config(text=f"Registered Students: {len(all_students)}")

    def _filter_today_table(self):
        """Filters today's table based on search input."""
        query = self.today_search_var.get().strip().lower()
        for item in self.tree_today.get_children():
            self.tree_today.delete(item)

        idx = 1
        for row in self.all_today_rows:
            if not query or query in row["name"].lower() or query in row["roll_no"].lower():
                self.tree_today.insert("", tk.END, values=(
                    idx,
                    row["roll_no"],
                    row["name"],
                    row["time_in"],
                    row["status"]
                ))
                idx += 1

    # -------------------------------------------------------------------------
    # Tab 2: Attendance Reports & Export
    # -------------------------------------------------------------------------
    def _build_reports_tab(self):
        filter_box = tk.LabelFrame(
            self.tab_reports,
            text=" Date Range Filter ",
            font=("Segoe UI", 9, "bold"),
            bg=self.BG_CARD,
            fg="white",
            padx=10,
            pady=8
        )
        filter_box.pack(fill=tk.X, pady=(0, 10))

        # Quick preset buttons
        preset_frame = tk.Frame(filter_box, bg=self.BG_CARD)
        preset_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(preset_frame, text="Quick Presets:", font=("Segoe UI", 9, "bold"), bg=self.BG_CARD, fg=self.TEXT_MUTED).pack(side=tk.LEFT)
        ttk.Button(preset_frame, text="Today", command=lambda: self._set_date_preset(0)).pack(side=tk.LEFT, padx=3)
        ttk.Button(preset_frame, text="Last 7 Days", command=lambda: self._set_date_preset(7)).pack(side=tk.LEFT, padx=3)
        ttk.Button(preset_frame, text="This Month", command=lambda: self._set_date_preset(30)).pack(side=tk.LEFT, padx=3)
        ttk.Button(preset_frame, text="All Time", command=self._set_date_preset_all).pack(side=tk.LEFT, padx=3)

        # Date input entries
        dates_row = tk.Frame(filter_box, bg=self.BG_CARD)
        dates_row.pack(fill=tk.X)

        tk.Label(dates_row, text="From Date (YYYY-MM-DD):", bg=self.BG_CARD, fg="white").pack(side=tk.LEFT)
        self.entry_from_date = tk.Entry(dates_row, bg="#1a1c29", fg="white", width=14, insertbackground="white")
        self.entry_from_date.insert(0, date.today().isoformat())
        self.entry_from_date.pack(side=tk.LEFT, padx=(5, 15))

        tk.Label(dates_row, text="To Date (YYYY-MM-DD):", bg=self.BG_CARD, fg="white").pack(side=tk.LEFT)
        self.entry_to_date = tk.Entry(dates_row, bg="#1a1c29", fg="white", width=14, insertbackground="white")
        self.entry_to_date.insert(0, date.today().isoformat())
        self.entry_to_date.pack(side=tk.LEFT, padx=(5, 15))

        btn_apply = ttk.Button(dates_row, text="Apply Filter", command=self.load_report_preview)
        btn_apply.pack(side=tk.LEFT, padx=5)

        # Export buttons
        btn_excel = ttk.Button(dates_row, text="📊 Export to Excel", style="Success.TButton", command=self.export_excel)
        btn_excel.pack(side=tk.RIGHT, padx=4)

        btn_csv = ttk.Button(dates_row, text="📄 Export to CSV", style="Primary.TButton", command=self.export_csv)
        btn_csv.pack(side=tk.RIGHT, padx=4)

        # Report preview table
        rep_frame = tk.Frame(self.tab_reports, bg=self.BG_CARD)
        rep_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("date", "time_in", "roll_no", "name", "status")
        self.tree_reports = ttk.Treeview(rep_frame, columns=cols, show="headings", selectmode="browse")
        
        self.tree_reports.heading("date", text="Date")
        self.tree_reports.heading("time_in", text="Time In")
        self.tree_reports.heading("roll_no", text="Roll Number")
        self.tree_reports.heading("name", text="Student Name")
        self.tree_reports.heading("status", text="Status")

        self.tree_reports.column("date", width=110, anchor="center")
        self.tree_reports.column("time_in", width=100, anchor="center")
        self.tree_reports.column("roll_no", width=120, anchor="w")
        self.tree_reports.column("name", width=200, anchor="w")
        self.tree_reports.column("status", width=90, anchor="center")

        scroll_y = ttk.Scrollbar(rep_frame, orient=tk.VERTICAL, command=self.tree_reports.yview)
        self.tree_reports.configure(yscrollcommand=scroll_y.set)

        self.tree_reports.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.current_report_df = pd.DataFrame()
        self.load_report_preview()

    def _set_date_preset(self, days: int):
        today = date.today()
        start = today - timedelta(days=days)
        self.entry_from_date.delete(0, tk.END)
        self.entry_from_date.insert(0, start.isoformat())
        self.entry_to_date.delete(0, tk.END)
        self.entry_to_date.insert(0, today.isoformat())
        self.load_report_preview()

    def _set_date_preset_all(self):
        self.entry_from_date.delete(0, tk.END)
        self.entry_to_date.delete(0, tk.END)
        self.load_report_preview()

    def load_report_preview(self):
        """Loads and previews attendance records matching current date filters."""
        start_date = self.entry_from_date.get().strip() or None
        end_date = self.entry_to_date.get().strip() or None

        self.current_report_df = get_attendance_report(start_date=start_date, end_date=end_date)
        
        for item in self.tree_reports.get_children():
            self.tree_reports.delete(item)

        for _, row in self.current_report_df.iterrows():
            self.tree_reports.insert("", tk.END, values=(
                row["Date"],
                row["Time In"],
                row["Roll Number"],
                row["Student Name"],
                row["Status"]
            ))

    def export_csv(self):
        """Exports the filtered attendance report to a CSV file."""
        if self.current_report_df.empty:
            messagebox.showwarning("Export Warning", "No records found to export for the selected date range.")
            return

        default_name = f"Attendance_Report_{date.today().isoformat()}.csv"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            self.current_report_df.to_csv(file_path, index=False)
            messagebox.showinfo("Export Successful", f"Successfully exported {len(self.current_report_df)} records to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save CSV file:\n{str(e)}")

    def export_excel(self):
        """Exports the filtered attendance report to an Excel (.xlsx) file."""
        if self.current_report_df.empty:
            messagebox.showwarning("Export Warning", "No records found to export for the selected date range.")
            return

        default_name = f"Attendance_Report_{date.today().isoformat()}.xlsx"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialfile=default_name,
            filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                self.current_report_df.to_excel(writer, sheet_name="Attendance", index=False)
            messagebox.showinfo("Export Successful", f"Successfully exported {len(self.current_report_df)} records to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save Excel file:\n{str(e)}")

    # -------------------------------------------------------------------------
    # Tab 3: Attendance Analytics & Summary (Stretch Goal)
    # -------------------------------------------------------------------------
    def _build_summary_tab(self):
        top_bar = tk.Frame(self.tab_summary, bg=self.BG_CARD)
        top_bar.pack(fill=tk.X, pady=(0, 8))

        self.lbl_summary_info = tk.Label(
            top_bar,
            text="Student Cumulative Attendance & Performance Summary",
            font=("Segoe UI", 10, "bold"),
            fg="white",
            bg=self.BG_CARD
        )
        self.lbl_summary_info.pack(side=tk.LEFT)

        btn_refresh_sum = ttk.Button(top_bar, text="⟳ Refresh Summary", command=self.refresh_summary)
        btn_refresh_sum.pack(side=tk.RIGHT)

        # Summary Treeview Table
        tree_frame = tk.Frame(self.tab_summary, bg=self.BG_CARD)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("roll_no", "name", "registered", "days_present", "total_days", "percentage", "status")
        self.tree_summary = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")

        self.tree_summary.heading("roll_no", text="Roll Number")
        self.tree_summary.heading("name", text="Student Name")
        self.tree_summary.heading("registered", text="Registered Date")
        self.tree_summary.heading("days_present", text="Days Present")
        self.tree_summary.heading("total_days", text="Total Days")
        self.tree_summary.heading("percentage", text="Attendance %")
        self.tree_summary.heading("status", text="Eligibility")

        self.tree_summary.column("roll_no", width=110, anchor="w")
        self.tree_summary.column("name", width=180, anchor="w")
        self.tree_summary.column("registered", width=130, anchor="center")
        self.tree_summary.column("days_present", width=100, anchor="center")
        self.tree_summary.column("total_days", width=100, anchor="center")
        self.tree_summary.column("percentage", width=110, anchor="center")
        self.tree_summary.column("status", width=110, anchor="center")

        scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree_summary.yview)
        self.tree_summary.configure(yscrollcommand=scroll_y.set)

        self.tree_summary.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

    def refresh_summary(self):
        """Calculates and refreshes the per-student attendance percentage summary."""
        df = get_attendance_summary()

        for item in self.tree_summary.get_children():
            self.tree_summary.delete(item)

        for _, row in df.iterrows():
            pct = row["Attendance %"]
            # Color-coded eligibility badge (> 75% Eligible, < 75% Low Attendance)
            eligibility = "Eligible (>=75%)" if pct >= 75.0 else "Warning (<75%)"

            self.tree_summary.insert("", tk.END, values=(
                row["Roll Number"],
                row["Student Name"],
                str(row["Registered Date"])[:10],
                row["Days Present"],
                row["Total Days"],
                f"{pct:.1f}%",
                eligibility
            ))

    # -------------------------------------------------------------------------
    # Status Bar & Real-Time Updates
    # -------------------------------------------------------------------------
    def _build_status_bar(self):
        self.status_bar = tk.Frame(self, bg="#161822", height=28, padx=15)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.lbl_status = tk.Label(
            self.status_bar,
            text="System ready.",
            font=("Segoe UI", 9),
            fg=self.TEXT_MUTED,
            bg="#161822"
        )
        self.lbl_status.pack(side=tk.LEFT)

    def _update_clock(self):
        """Updates the digital clock on header every second."""
        now_str = datetime.now().strftime("%A, %b %d %Y   %I:%M:%S %p")
        self.clock_label.config(text=now_str)
        self.after(1000, self._update_clock)

    # -------------------------------------------------------------------------
    # Live Camera Scanner Control
    # -------------------------------------------------------------------------
    def toggle_scanner(self):
        """Starts or stops the live webcam attendance scanner."""
        if not self.is_camera_running:
            self.start_scanner()
        else:
            self.stop_scanner()

    def start_scanner(self):
        """Initializes and runs the AttendanceRecognizer background thread."""
        self.is_camera_running = True
        self.btn_toggle_cam.config(text="⏹ Stop Scanner", style="Danger.TButton")
        self.cam_badge.config(text="● LIVE", fg=self.ACCENT_SUCCESS)
        self.lbl_status.config(text="Camera active. Scanning for faces in real-time...")

        self.recognizer.start(
            frame_callback=self._on_scanner_frame,
            attendance_callback=self._on_attendance_logged,
            status_callback=self._on_scanner_status
        )

    def stop_scanner(self):
        """Stops the camera scanner."""
        self.is_camera_running = False
        self.recognizer.stop()
        self.btn_toggle_cam.config(text="▶ Start Scanner", style="Success.TButton")
        self.cam_badge.config(text="● OFF", fg=self.ACCENT_DANGER)
        self.lbl_status.config(text="Camera stopped. Scanner is on standby.")
        self._draw_placeholder_canvas("Camera Standby\nClick 'Start Scanner' to Begin")

    def _on_scanner_frame(self, frame_bgr: np.ndarray):
        """Callback received from recognition thread with an annotated frame."""
        if not self.is_camera_running:
            return

        # Resize for preview canvas (540x405)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (540, 405), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(resized)

        # Thread-safe UI update
        self.after(0, self._render_frame_to_canvas, img)

    def _render_frame_to_canvas(self, pil_image: Image.Image):
        """Renders PIL image to Tkinter canvas."""
        self.current_tk_image = ImageTk.PhotoImage(image=pil_image)
        self.video_canvas.delete("all")
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.current_tk_image)

    def _on_attendance_logged(self, student_dict: dict):
        """Callback triggered when a student is recognized and attendance is recorded."""
        self.after(0, self._handle_attendance_event, student_dict)

    def _handle_attendance_event(self, s: dict):
        """Updates GUI when attendance is logged."""
        msg = f"✓ Verified: {s['name']} (Roll: {s['roll_no']}) at {s['time_in']}"
        self.lbl_alert_msg.config(text=msg, fg="#52b788")
        self.lbl_status.config(text=f"Logged attendance for {s['name']} at {s['time_in']}")
        # Refresh tables
        self.refresh_today_attendance()
        self.refresh_summary()

    def _on_scanner_status(self, status_msg: str, is_error: bool = False):
        """Status update callback from recognizer."""
        self.after(0, lambda: self._update_scanner_status(status_msg, is_error))

    def _update_scanner_status(self, msg: str, is_error: bool):
        self.lbl_status.config(text=msg)
        if is_error:
            messagebox.showerror("Camera Error", msg)
            self.stop_scanner()

    # -------------------------------------------------------------------------
    # Student Face Enrollment Wizard Modal
    # -------------------------------------------------------------------------
    def open_enrollment_dialog(self):
        """Opens interactive enrollment modal dialog."""
        # Temporarily pause scanner if it was running
        was_scanner_active = self.is_camera_running
        if was_scanner_active:
            self.stop_scanner()

        dialog = EnrollmentDialog(self)
        self.wait_window(dialog)

        # Reload recognizer known faces and update counts
        self.recognizer.reload_known_faces()
        self.refresh_today_attendance()
        self.refresh_summary()

        # Resume scanner if previously active
        if was_scanner_active:
            self.start_scanner()

    def on_close(self):
        """Cleanup resources on application exit."""
        self.stop_scanner()
        self.destroy()


class EnrollmentDialog(tk.Toplevel):
    """
    Modal window to capture student info and 5 face samples via webcam.
    """
    def __init__(self, parent: AttendanceDashboard):
        super().__init__(parent)
        self.parent = parent
        self.title("Enroll New Student")
        self.geometry("680x620")
        self.resizable(False, False)
        self.configure(bg="#2b2d42")
        self.transient(parent)
        self.grab_set()

        self.enrollment_manager = EnrollmentManager(camera_index=0, samples_needed=5)
        self.is_capturing = False
        self.capture_thread = None
        self.current_preview_tk = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_dialog_close)

    def _build_ui(self):
        header = tk.Frame(self, bg="#161822", pady=10, padx=15)
        header.pack(fill=tk.X)
        tk.Label(header, text="Student Registration Wizard", font=("Segoe UI", 13, "bold"), fg="white", bg="#161822").pack(anchor="w")
        tk.Label(header, text="Enter student details and capture 5 face samples via webcam.", font=("Segoe UI", 9), fg="#adb5bd", bg="#161822").pack(anchor="w")

        # Form Inputs
        form_frame = tk.Frame(self, bg="#2b2d42", padx=20, pady=12)
        form_frame.pack(fill=tk.X)

        tk.Label(form_frame, text="Full Name:", font=("Segoe UI", 10, "bold"), fg="white", bg="#2b2d42").grid(row=0, column=0, sticky="w", pady=4)
        self.ent_name = tk.Entry(form_frame, font=("Segoe UI", 10), bg="#1a1c29", fg="white", insertbackground="white", width=32)
        self.ent_name.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=4)

        tk.Label(form_frame, text="Roll Number:", font=("Segoe UI", 10, "bold"), fg="white", bg="#2b2d42").grid(row=1, column=0, sticky="w", pady=4)
        self.ent_roll = tk.Entry(form_frame, font=("Segoe UI", 10), bg="#1a1c29", fg="white", insertbackground="white", width=32)
        self.ent_roll.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=4)

        # Live Preview Canvas (480x360)
        self.preview_canvas = tk.Canvas(self, width=480, height=340, bg="#0d0e15", highlightthickness=1, highlightbackground="#3d405b")
        self.preview_canvas.pack(pady=8)
        self._draw_placeholder("Live Preview\nClick 'Start Capture' Below")

        # Progress bar
        self.progress_bar = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=480, mode="determinate", maximum=5)
        self.progress_bar.pack(pady=(4, 6))

        self.lbl_progress = tk.Label(self, text="0 / 5 Face Samples Captured", font=("Segoe UI", 9), fg="#94d2bd", bg="#2b2d42")
        self.lbl_progress.pack()

        # Action Buttons
        btn_bar = tk.Frame(self, bg="#2b2d42", pady=12)
        btn_bar.pack(fill=tk.X, padx=20)

        self.btn_start = ttk.Button(btn_bar, text="📸 Start Face Capture (5 Shots)", style="Success.TButton", command=self.start_enrollment_process)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        btn_cancel = ttk.Button(btn_bar, text="Cancel", style="Danger.TButton", command=self._on_dialog_close)
        btn_cancel.pack(side=tk.RIGHT, padx=5)

    def _draw_placeholder(self, text: str):
        self.preview_canvas.delete("all")
        w = int(self.preview_canvas.cget("width"))
        h = int(self.preview_canvas.cget("height"))
        self.preview_canvas.create_text(w // 2, h // 2, text=text, fill="#6c757d", font=("Segoe UI", 11, "bold"), justify="center")

    def start_enrollment_process(self):
        """Validates inputs and starts enrollment capture thread."""
        name = self.ent_name.get().strip()
        roll = self.ent_roll.get().strip()

        if not name:
            messagebox.showwarning("Validation Error", "Please enter the student's full name.", parent=self)
            return
        if not roll:
            messagebox.showwarning("Validation Error", "Please enter the student's roll number.", parent=self)
            return

        existing = get_student_by_roll_no(roll)
        if existing:
            messagebox.showerror("Duplicate Roll Number", f"Roll number '{roll}' is already assigned to '{existing['name']}'.", parent=self)
            return

        self.btn_start.config(state=tk.DISABLED)
        self.ent_name.config(state=tk.DISABLED)
        self.ent_roll.config(state=tk.DISABLED)
        self.is_capturing = True

        self.capture_thread = threading.Thread(target=self._capture_worker, args=(name, roll), daemon=True)
        self.capture_thread.start()

    def _capture_worker(self, name: str, roll: str):
        """Worker thread executing face enrollment."""
        def on_frame(frame, status_text):
            if not self.is_capturing:
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (480, 340), interpolation=cv2.INTER_AREA)
            img = Image.fromarray(resized)
            self.after(0, self._render_preview, img)

        def on_progress(current, total):
            self.after(0, lambda: self._update_progress(current, total))

        def on_status(msg):
            self.after(0, lambda: self.lbl_progress.config(text=msg))

        success, message = self.enrollment_manager.capture_and_enroll(
            name=name,
            roll_no=roll,
            camera_index=0,
            frame_callback=on_frame,
            progress_callback=on_progress,
            status_callback=on_status
        )

        self.after(0, lambda: self._handle_enrollment_result(success, message))

    def _render_preview(self, pil_img: Image.Image):
        self.current_preview_tk = ImageTk.PhotoImage(image=pil_img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self.current_preview_tk)

    def _update_progress(self, current: int, total: int):
        self.progress_bar["value"] = current
        self.lbl_progress.config(text=f"{current} / {total} Face Samples Captured")

    def _handle_enrollment_result(self, success: bool, message: str):
        self.is_capturing = False
        if success:
            messagebox.showinfo("Enrollment Success", message, parent=self)
            self.destroy()
        else:
            messagebox.showerror("Enrollment Failed", message, parent=self)
            self.btn_start.config(state=tk.NORMAL)
            self.ent_name.config(state=tk.NORMAL)
            self.ent_roll.config(state=tk.NORMAL)
            self._draw_placeholder("Capture Interrupted\nClick 'Start Capture' to Retry")

    def _on_dialog_close(self):
        self.is_capturing = False
        self.destroy()


def run_dashboard():
    """Initializes and runs the Face Recognition Attendance Desktop application."""
    app = AttendanceDashboard()
    app.mainloop()


if __name__ == "__main__":
    run_dashboard()
