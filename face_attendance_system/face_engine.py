"""
face_engine.py - Face Processing and Recognition Engine
Provides a unified interface for:
  1. Face detection in image/video frames
  2. Generating 128-dimensional face encodings
  3. Distance calculation and identity matching using Euclidean distance (threshold: 0.6)
  4. Liveness / Anti-spoofing verification via Eye Aspect Ratio (EAR) blink detection
  
Primary Backend: 'face_recognition' library (dlib 128-d ResNet embeddings).
Fallback Backend: High-fidelity OpenCV engine for environments where dlib is not yet compiled.
"""

import os
import math
import numpy as np
import cv2

# Flag to indicate whether the primary dlib-based face_recognition library is available
HAS_FACE_RECOGNITION = False

try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except (ImportError, Exception):
    HAS_FACE_RECOGNITION = False

# Safely check for Haar cascade classifiers if supported by the installed OpenCV version
HAAR_FACE = None
HAAR_EYE = None

try:
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        face_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        eye_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
        if os.path.exists(face_path):
            HAAR_FACE = cv2.CascadeClassifier(face_path)
        if os.path.exists(eye_path):
            HAAR_EYE = cv2.CascadeClassifier(eye_path)
except Exception:
    HAAR_FACE = None
    HAAR_EYE = None


def get_engine_info() -> dict:
    """Returns metadata about the active face recognition engine."""
    if HAS_FACE_RECOGNITION:
        return {
            "backend": "face_recognition (dlib)",
            "is_dlib": True,
            "description": "Dlib 128-d ResNet Deep Metric Embeddings (Primary Engine)"
        }
    return {
        "backend": "OpenCV Smart Recognizer (Fallback)",
        "is_dlib": False,
        "description": "OpenCV Spatial 128-d Normalized Embedding Engine (Zero-Compile Fallback)"
    }


def _detect_faces_fallback(frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Fallback face detector:
    Uses Haar cascades if available, or chromatic skin-segmentation + morphological
    filtering and contour geometry analysis.
    """
    h, w = frame_bgr.shape[:2]

    # 1. Try Haar Cascade if available
    if HAAR_FACE is not None and not HAAR_FACE.empty():
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = HAAR_FACE.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            locations = []
            for (x, y, fw, fh) in faces:
                locations.append((int(y), int(x + fw), int(y + fh), int(x)))
            if locations:
                return locations
        except Exception:
            pass

    # 2. Chromatic Skin Segmentation in YCrCb color space
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    # Human skin chrominance cluster: Cr in [133, 173], Cb in [77, 127]
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
    
    # Morphological opening and closing to reduce noise and close face holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (h * w) * 0.015  # At least 1.5% of frame area

    locations = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            x, y, cw, ch = cv2.boundingRect(c)
            aspect_ratio = float(ch) / float(cw)
            # Faces typically have an aspect ratio between 0.75 and 1.6
            if 0.7 <= aspect_ratio <= 1.8:
                locations.append((int(y), int(x + cw), int(y + ch), int(x)))

    if locations:
        return locations

    # 3. Intensity-contrast contour detector for non-skin / synthetic test faces
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edges, kernel_e, iterations=2)
    contours_e, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours_e:
        area = cv2.contourArea(c)
        if area >= min_area:
            x, y, cw, ch = cv2.boundingRect(c)
            aspect_ratio = float(ch) / float(cw)
            if 0.65 <= aspect_ratio <= 1.6:
                locations.append((int(y), int(x + cw), int(y + ch), int(x)))

    return locations


def detect_faces(frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detects all faces in a BGR frame.
    
    Returns:
        List of bounding boxes in (top, right, bottom, left) format,
        matching face_recognition library convention.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    if HAS_FACE_RECOGNITION:
        # face_recognition requires RGB format
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Model 'hog' is fast and CPU-friendly; 'cnn' is GPU accelerated
        locations = face_recognition.face_locations(rgb_frame, model="hog")
        return locations

    return _detect_faces_fallback(frame_bgr)


def _generate_opencv_128d_encoding(face_chip: np.ndarray) -> np.ndarray:
    """
    Fallback 128-dimensional face embedding generator using normalized spatial-frequency
    and texture feature projection.
    Produces a unit-norm 128-d vector where Euclidean distance reflects facial similarity.
    """
    # 1. Convert to canonical 96x96 grayscale face chip
    if len(face_chip.shape) == 3:
        gray = cv2.cvtColor(face_chip, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_chip.copy()

    resized = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    # Lighting normalization
    normalized = cv2.equalizeHist(resized)

    # 2. Divide into 4x4 spatial cells (each 24x24 px) -> 16 regions
    cells = []
    for r in range(4):
        for c in range(4):
            cell = normalized[r*24:(r+1)*24, c*24:(c+1)*24]
            # Mean and Std
            mean, std = cv2.meanStdDev(cell)
            # Sobel horizontal and vertical gradients
            gx = cv2.Sobel(cell, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(cell, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            grad_mean = float(np.mean(mag))
            grad_std = float(np.std(mag))
            
            # Simple 4-bin gradient orientation histogram
            angles = cv2.phase(gx, gy, angleInDegrees=True)
            hist, _ = np.histogram(angles, bins=4, range=(0, 360))
            hist_norm = hist / (np.sum(hist) + 1e-6)

            # 2 (mean, std) + 2 (grad_mean, grad_std) + 4 (orientations) = 8 features per cell
            cell_features = [float(mean[0][0]), float(std[0][0]), grad_mean, grad_std]
            cell_features.extend([float(h) for h in hist_norm])
            cells.extend(cell_features)

    # 16 cells * 8 features = 128-dimensional feature vector
    raw_vector = np.array(cells, dtype=np.float64)

    # L2 normalize so Euclidean distance d = sqrt(2 * (1 - cosine_similarity))
    norm = np.linalg.norm(raw_vector)
    if norm > 1e-6:
        return raw_vector / norm
    return raw_vector


def encode_faces(frame_bgr: np.ndarray, face_locations: list[tuple[int, int, int, int]] | None = None) -> list[np.ndarray]:
    """
    Computes 128-dimensional face encodings for detected faces.
    
    Args:
        frame_bgr: Source image in BGR format.
        face_locations: Optional list of (top, right, bottom, left) coordinates.
                        If None, faces will be detected automatically.
                        
    Returns:
        List of 128-element 1D numpy arrays (dtype float64).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    if face_locations is None:
        face_locations = detect_faces(frame_bgr)

    if not face_locations:
        return []

    if HAS_FACE_RECOGNITION:
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Returns 128-d encodings from dlib ResNet model
        encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations, num_jitters=1)
        return [enc.astype(np.float64) for enc in encodings]

    # Fallback encoding
    h, w = frame_bgr.shape[:2]
    encodings = []
    for (top, right, bottom, left) in face_locations:
        # Clamp bounds
        t = max(0, top)
        l = max(0, left)
        b = min(h, bottom)
        r = min(w, right)
        if b - t < 10 or r - l < 10:
            continue
        chip = frame_bgr[t:b, l:r]
        enc = _generate_opencv_128d_encoding(chip)
        encodings.append(enc)

    return encodings


def face_distance(known_encodings: list[np.ndarray] | np.ndarray, face_encoding: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean distance between a face encoding and a list of known face encodings.
    
    In a 128-dimensional Euclidean metric space:
      - Distance 0.0 means identical face
      - Distance <= 0.6 indicates a match (standard industry threshold)
      - Distance > 0.6 indicates an unknown face or non-match
      
    Args:
        known_encodings: List or array of known 128-d encodings.
        face_encoding: 128-d encoding of the candidate face.
        
    Returns:
        1D numpy array of Euclidean distances.
    """
    if len(known_encodings) == 0:
        return np.empty((0,))

    known_arr = np.array(known_encodings, dtype=np.float64)
    target = np.array(face_encoding, dtype=np.float64)

    # Euclidean distance formula: sqrt( sum( (a_i - b_i)^2 ) )
    return np.linalg.norm(known_arr - target, axis=1)


def compare_faces(known_encodings: list[np.ndarray], face_encoding: np.ndarray, tolerance: float = 0.6) -> list[bool]:
    """
    Compares a candidate face encoding against known encodings.
    
    Args:
        known_encodings: List of stored face encodings.
        face_encoding: Candidate face encoding to test.
        tolerance: Distance threshold (default 0.6).
        
    Returns:
        List of booleans: True if distance <= tolerance, False otherwise.
    """
    distances = face_distance(known_encodings, face_encoding)
    return list(distances <= tolerance)


def calculate_confidence(distance: float, threshold: float = 0.6) -> float:
    """
    Converts Euclidean distance into an intuitive confidence percentage [0.0 - 100.0%].
    
    When distance == 0.0 -> Confidence is 100%
    When distance == threshold (0.6) -> Confidence is 65% (matching boundary)
    When distance > threshold -> Confidence drops towards 0%
    """
    if distance <= threshold:
        # Linear interpolation from 100% at dist=0 to 65% at dist=threshold
        confidence = 100.0 - (distance / threshold) * 35.0
    else:
        # Drops from 65% to 0% as distance grows from threshold to threshold*1.8
        excess = distance - threshold
        decay_range = threshold * 0.8
        confidence = max(0.0, 65.0 - (excess / decay_range) * 65.0)

    return round(float(np.clip(confidence, 0.0, 100.0)), 1)


# ---------------------------------------------------------------------------
# Stretch Goal: Liveness / Anti-Spoofing Detection (Eye Blink / EAR)
# ---------------------------------------------------------------------------

def _calculate_ear_from_landmarks(eye_points: list[tuple[int, int]]) -> float:
    """
    Calculates Eye Aspect Ratio (EAR) from 6 landmark points:
      p1: outer corner, p4: inner corner
      p2, p3: top eyelid points
      p5, p6: bottom eyelid points
      
    EAR = (|p2 - p6| + |p3 - p5|) / (2.0 * |p1 - p4|)
    """
    def dist(p_a, p_b):
        return math.hypot(p_a[0] - p_b[0], p_a[1] - p_b[1])

    # Vertical eye distances
    a = dist(eye_points[1], eye_points[5])
    b = dist(eye_points[2], eye_points[4])
    # Horizontal eye distance
    c = dist(eye_points[0], eye_points[3])

    if c < 1e-5:
        return 0.0
    return (a + b) / (2.0 * c)


class LivenessTracker:
    """
    Tracks eye state over consecutive frames to detect natural human blinks,
    providing protection against photo/screen presentation attacks.
    """
    def __init__(self, ear_threshold: float = 0.22, consecutive_frames: int = 2):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.closed_frame_count = 0
        self.total_blinks = 0
        self.is_live = False

    def update(self, frame_bgr: np.ndarray, face_location: tuple[int, int, int, int]) -> tuple[bool, float]:
        """
        Updates liveness state with a new frame.
        
        Returns:
            (is_verified_live: bool, current_ear_or_openness: float)
        """
        top, right, bottom, left = face_location
        h, w = frame_bgr.shape[:2]
        t, b = max(0, top), min(h, bottom)
        l, r = max(0, left), min(w, right)
        
        if b - t < 20 or r - l < 20:
            return self.is_live, 0.0

        if HAS_FACE_RECOGNITION:
            rgb_face = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            landmarks_list = face_recognition.face_landmarks(rgb_face, face_locations=[face_location])
            if landmarks_list:
                landmarks = landmarks_list[0]
                left_eye = landmarks.get("left_eye", [])
                right_eye = landmarks.get("right_eye", [])
                if len(left_eye) == 6 and len(right_eye) == 6:
                    left_ear = _calculate_ear_from_landmarks(left_eye)
                    right_ear = _calculate_ear_from_landmarks(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0

                    if avg_ear < self.ear_threshold:
                        self.closed_frame_count += 1
                    else:
                        if self.closed_frame_count >= self.consecutive_frames:
                            self.total_blinks += 1
                            self.is_live = True
                        self.closed_frame_count = 0
                    return self.is_live, avg_ear

        # Fallback eye detector via Haar Cascade or ocular region variance
        if HAAR_EYE is not None and not HAAR_EYE.empty():
            try:
                face_gray = cv2.cvtColor(frame_bgr[t:b, l:r], cv2.COLOR_BGR2GRAY)
                upper_face = face_gray[0:int((b-t)*0.6), :]
                eyes = HAAR_EYE.detectMultiScale(upper_face, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
                if len(eyes) == 0:
                    self.closed_frame_count += 1
                else:
                    if self.closed_frame_count >= self.consecutive_frames:
                        self.total_blinks += 1
                        self.is_live = True
                    self.closed_frame_count = 0
                return self.is_live, float(len(eyes))
            except Exception:
                pass

        # Pixel variance tracker over eye region
        try:
            face_gray = cv2.cvtColor(frame_bgr[t:b, l:r], cv2.COLOR_BGR2GRAY)
            eye_region = face_gray[int((b-t)*0.2):int((b-t)*0.5), :]
            _, std_val = cv2.meanStdDev(eye_region)
            variance = float(std_val[0][0])
            # High variance = open eye with contrast; low variance = closed smooth eyelid
            if variance < 15.0:
                self.closed_frame_count += 1
            else:
                if self.closed_frame_count >= self.consecutive_frames:
                    self.total_blinks += 1
                    self.is_live = True
                self.closed_frame_count = 0
            return self.is_live, variance
        except Exception:
            return self.is_live, 0.0
