# Face Recognition Attendance System

An automated, real-time facial recognition attendance system that captures webcam video, detects and recognizes student faces, and logs attendance into an SQLite database with duplicate prevention. Built with Python, OpenCV, `face_recognition` (dlib), Tkinter desktop GUI, and Pandas.

---

## 🌟 Key Features

1. **Real-Time Face Recognition**:
   - Detects and identifies faces continuously via webcam.
   - Calculates 128-dimensional deep facial metric embeddings.
   - Strict confidence threshold (distance $\le 0.6$) separating known students from **"Unknown"** visitors.
2. **Duplicate Attendance Prevention**:
   - Database-level composite unique constraint on `(student_id, date)`.
   - Prevents marking attendance multiple times for the same student on the same calendar day.
3. **Student Enrollment Wizard**:
   - Captures 5 high-quality face samples with visual feedback and angle variety.
   - Computes a normalized average 128-d encoding for high recognition robustness across varying lighting and head tilts.
4. **Desktop GUI Dashboard (Tkinter)**:
   - Live embedded webcam video feed with real-time bounding boxes and confidence score tags.
   - Live "Today's Attendance" table with instant auto-refresh and search filtering.
   - Export reports filtered by date range to **CSV** and **Excel (`.xlsx`)** using Pandas.
5. **Stretch Goals Implemented**:
   - 🛡️ **Liveness Detection (Anti-Spoofing)**: Facial landmark Eye Aspect Ratio (EAR) blink detection prevents attendance fraud using printed photos or phone screens.
   - 👥 **Multi-Face Recognition**: Detects, frames, and marks attendance for multiple faces in a single video frame simultaneously.
   - 📊 **Attendance Percentage & Analytics**: Computes overall attendance rate (`Days Present / Total Days * 100%`) with eligibility warnings (< 75%).
6. **Dual-Backend Resilience**:
   - Primary: `face_recognition` (dlib 128-d ResNet).
   - Fallback: OpenCV Smart Recognizer (spatial gradient 128-d embedding) for immediate zero-compile execution on Windows systems without C++ compilers.

---

## 🏗️ Project Architecture

```
d:/face_attendance_system/
├── database.py              # SQLite schema, student records, duplicate check, reports & summaries
├── face_engine.py           # Unified face detection, 128-d encoding, Euclidean distance, & EAR blink liveness
├── enroll.py                # 5-shot webcam student enrollment module (CLI + GUI wizard)
├── recognize_attendance.py  # Real-time multi-face recognition and attendance logging engine
├── dashboard.py             # Polished Tkinter desktop GUI application
├── main.py                  # Entry point: checks environment & initializes database
├── requirements.txt         # Project dependencies
├── test_system.py           # Comprehensive automated test suite
└── README.md                # Full documentation & setup guide
```

---

## 📐 Face Encoding and Matching Logic Explained

### 1. 128-Dimensional Face Embeddings
When a face is detected in an image frame:
- The face region is cropped, aligned, and passed through a deep convolutional neural network (ResNet trained on millions of faces).
- The network outputs a **128-dimensional unit vector** representing unique facial measurements (e.g. distance between eyes, nose bridge length, jawline curvature, chin contour).
$$\vec{v} = [x_1, x_2, \dots, x_{128}]$$

### 2. Multi-Shot Average Enrollment
During enrollment, 5 separate face frames are captured. The system calculates the mean vector and L2-normalizes it:
$$\vec{v}_{\text{enrolled}} = \frac{\sum_{i=1}^5 \vec{v}_i}{\|\sum_{i=1}^5 \vec{v}_i\|_2}$$
This eliminates noise from single-frame facial micro-expressions or uneven illumination.

### 3. Euclidean Distance & Threshold 0.6
To test whether a candidate face matches an enrolled student, we compute the Euclidean distance between candidate vector $\vec{c}$ and enrolled vector $\vec{e}$:
$$d(\vec{c}, \vec{e}) = \sqrt{\sum_{k=1}^{128} (c_k - e_k)^2}$$
- **$d \le 0.6$ (Match)**: High confidence match. The student is recognized, framed with a green box, and attendance is recorded.
- **$d > 0.6$ (Unknown)**: Candidate is not registered. Framed with a red box marked **"Unknown"** to prevent false matching.

### 4. Confidence Percentage Calculation
Distance is mapped to an intuitive percentage score:
$$\text{Confidence (\%)} = \begin{cases} 
100 - \left(\frac{d}{0.6}\right) \times 35 & \text{if } d \le 0.6 \\
\max\left(0, 65 - \left(\frac{d - 0.6}{0.48}\right) \times 65\right) & \text{if } d > 0.6 
\end{cases}$$

### 5. Eye Aspect Ratio (EAR) for Blink Detection
To verify that a real person is in front of the camera (anti-spoofing):
The system tracks 6 facial landmarks for each eye:
$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \times \|p_1 - p_4\|}$$
During a natural blink, the eyelid points close and EAR drops drastically ($< 0.22$) for 2-3 frames before returning to normal ($> 0.28$). Detecting this transition verifies liveness.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14.
- A functional webcam.

### 1. Install Base Dependencies
```bash
pip install opencv-python pandas openpyxl pillow
```

### 2. Installing `face_recognition` & `dlib` on Windows

`dlib` compiles C++ extensions. Follow these steps based on your environment:

#### Option A: Quick Pre-built Wheels (Recommended for Python 3.10 / 3.11)
If using Python 3.10 or 3.11, you can install pre-compiled dlib wheels directly:
```bash
pip install cmake
pip install face-recognition
```
Or download a community pre-built `.whl` for your Python version and run:
```bash
pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
pip install face-recognition
```

#### Option B: Build from Source with Visual Studio (For Python 3.12+)
1. Download and install **Visual Studio Community** or **Visual Studio C++ Build Tools** from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. In the installer, check:
   - **Desktop development with C++**
   - **MSVC v143 - VS 2022 C++ x64/x86 build tools**
   - **Windows 10/11 SDK**
   - **C++ CMake tools for Windows**
3. Open a terminal and run:
   ```bash
   pip install cmake
   pip install dlib
   pip install face-recognition
   ```

#### Option C: Zero-Compile Fallback Mode (Works out-of-the-box!)
If you do not have C++ compilers installed, **the system automatically activates its built-in OpenCV Smart Recognizer fallback engine**. All functions—live camera feeds, 5-shot enrollment, Euclidean distance matching, duplicate prevention, and CSV/Excel export—operate seamlessly out of the box!

---

## 💻 How to Run

### 1. Launch the Desktop GUI Dashboard
```bash
cd d:/face_attendance_system
python main.py
```
- Click **"▶ Start Scanner"** to activate the live webcam attendance feed.
- Click **"👤 Enroll New Student"** to open the registration wizard.
- View real-time logged attendance in the **Today's Attendance** tab.
- Filter and export records in the **Reports & Export** tab.
- View per-student attendance rates in the **Student Summary (%)** tab.

### 2. Standalone Face Enrollment (CLI)
```bash
python enroll.py
```
Or with arguments:
```bash
python enroll.py --name "Alice Johnson" --roll "CS-101" --camera 0
```

### 3. Standalone Live Recognition (CLI/OpenCV)
```bash
python recognize_attendance.py
```
- Press `q` or `ESC` to quit the live camera window.

---

## 🧪 Automated Testing

A complete automated test suite is provided to verify the database, face engine, enrollment, recognition logic, and Pandas export without needing a physical webcam:

```bash
python test_system.py
```

The test validates:
- [x] Database creation and SQLite schema integrity
- [x] Student registration with 128-d serialized BLOB
- [x] Prevention of duplicate roll numbers
- [x] Attendance logging and composite unique constraint `(student_id, date)`
- [x] Correct distance computation ($\le 0.6$ match vs $> 0.6$ unknown)
- [x] Multi-face recognition pipeline
- [x] Eye Aspect Ratio / blink liveness tracker
- [x] Export to CSV and Excel (`.xlsx`) via Pandas

---

## 📊 Database Schema Reference

### `students` Table
| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique student ID |
| `name` | TEXT | NOT NULL | Student full name |
| `roll_no` | TEXT | UNIQUE NOT NULL | Unique roll/ID number |
| `face_encoding`| BLOB | NOT NULL | Pickled 128-d float64 numpy array |
| `registered_on`| TIMESTAMP| DEFAULT CURRENT_TIMESTAMP | Registration timestamp |

### `attendance` Table
| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Attendance log ID |
| `student_id` | INTEGER | NOT NULL, FK -> students(id)| Reference to student |
| `date` | TEXT | NOT NULL (YYYY-MM-DD) | Date of attendance |
| `time_in` | TEXT | NOT NULL (HH:MM:SS) | Entry timestamp |
| `status` | TEXT | DEFAULT 'Present' | Status (Present/Late) |
| *Composite* | UNIQUE | `(student_id, date)` | Ensures 1 entry per day |

---

## 🛡️ Error Handling Details

- **Camera Not Available**: Clean error message prompting to verify webcam permissions or USB connection.
- **No Face in View**: Live HUD prompts "No face detected. Please look directly at camera."
- **Multiple Faces During Registration**: Enrollment alerts that only 1 person must be in view.
- **Duplicate Registration**: Caught at SQLite constraint level, informing user of existing student name.
- **Duplicate Attendance**: Silently verified in-memory and in SQLite; user gets "Already Marked" badge.
- **Exporting Empty Date Range**: Clear warning dialogue without corrupting files.
