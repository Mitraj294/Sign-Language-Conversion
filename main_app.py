# main_app.py

import numpy as np
import math
import cv2
import os
import sys
import traceback
import pyttsx3
import enchant
import tkinter as tk
from PIL import Image, ImageTk
import argparse
from string import ascii_uppercase
import time # Added for data collection timing

# Attempt to import Keras components, handling potential TensorFlow 2+ structure
try:
    from tensorflow.keras.models import load_model
except ImportError:
    try:
        from keras.models import load_model
    except ImportError:
        print("ERROR: Keras or TensorFlow not found. Please install TensorFlow.")
        sys.exit(1)

try:
    from cvzone.HandTrackingModule import HandDetector
except ImportError:
    print("ERROR: cvzone not found. Please install cvzone (`pip install cvzone`).")
    sys.exit(1)

# --- Constants ---
OFFSET = 29
SKELETON_SIZE = 400 # Target size for the skeleton image fed to the model
DEFAULT_MODEL_PATH = 'cnn8grps_rad1_model.h5'
DEFAULT_BG_PATH = 'white.jpg'
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
# Define landmark connections for drawing skeleton
LANDMARK_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (5, 6), (6, 7), (7, 8),               # Index
    (9, 10), (10, 11), (11, 12),          # Middle
    (13, 14), (14, 15), (15, 16),         # Ring
    (17, 18), (18, 19), (19, 20),         # Pinky
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17) # Palm
]

# --- Configuration Class ---
class Config:
    """Holds application configuration."""
    def __init__(self, args):
        self.input = args.input
        self.mode = args.mode.lower()
        self.gui = args.gui.lower() == 'on'
        self.model_path = args.model
        self.bg_path = args.bg
        self.output_dir = args.output
        self.label = args.label.upper() if args.label else None

        if self.mode == 'collect' and not self.output_dir:
            print("ERROR: --output directory must be specified for collection mode.")
            sys.exit(1)
        if self.mode == 'collect' and not self.label:
            print("ERROR: --label must be specified for collection mode.")
            sys.exit(1)

# --- Helper Functions ---
def distance(p1, p2):
    """Calculates Euclidean distance between two 2D points."""
    if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2:
        # print(f"Warning: Invalid points for distance calculation: {p1}, {p2}")
        return float('inf')
    try:
        return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    except IndexError:
        # print(f"Warning: Index error during distance calculation: {p1}, {p2}")
        return float('inf')

# --- Main Application Class ---
class SignPredictorApp:
    """Handles Sign Language Prediction/Collection with optional GUI."""

    def __init__(self, config: Config):
        self.config = config
        self.is_image_mode = False
        self.static_image = None
        self.vs = None
        self.model = None
        self.white_bg = None
        self.speak_engine = None
        self.spell_checker = None
        self.root = None # Tkinter root

        # --- Initialize Core Components ---
        self._initialize_libraries()
        if not self._load_resources():
             # If essential resources failed, prevent GUI/Loop start
             self.root = None # Ensure mainloop isn't called
             return
        self._initialize_state()

        # --- Initialize Input Source ---
        if not self._initialize_input_source():
             self.root = None # Ensure mainloop isn't called
             return

        # --- Initialize GUI (if enabled) ---
        if self.config.gui:
            self._initialize_gui()
        else:
            # Create dummy panels if GUI is off, so updates don't crash
            # (though they won't be used)
            class DummyWidget:
                def config(self, *args, **kwargs): pass
            self.panel = DummyWidget()
            self.panel2 = DummyWidget()
            self.panel3 = DummyWidget()
            self.panel5 = DummyWidget()
            self.b1 = self.b2 = self.b3 = self.b4 = DummyWidget()


    def _initialize_libraries(self):
        """Initialize external libraries like pyttsx3 and enchant."""
        print("Initializing libraries...")
        # Spell Checker
        try:
            self.spell_checker = enchant.Dict("en-US")
        except enchant.errors.DictNotFoundError as e:
            print(f"Warning: Enchant dictionary 'en-US' not found: {e}")
            print("Spell checking/suggestions will be disabled.")
            print("Ensure enchant and dictionary files (e.g., aspell-en or hunspell-en-us) are installed.")
            self.spell_checker = None
        except Exception as e:
            print(f"Warning: Failed to initialize enchant: {e}")
            self.spell_checker = None

        # Text-to-Speech
        try:
            self.speak_engine = pyttsx3.init()
            self.speak_engine.setProperty("rate", 150)
            voices = self.speak_engine.getProperty("voices")
            if voices:
                self.speak_engine.setProperty("voice", voices[0].id)
            else:
                print("Warning: No TTS voices found.")
        except Exception as e:
            print(f"Warning: Failed to initialize pyttsx3: {e}")
            self.speak_engine = None

        # Hand Detector
        self.detector = HandDetector(maxHands=1) # Only need one detector

    def _load_resources(self):
        """Load model and background image."""
        print("Loading resources...")
        # Load Model
        if self.config.mode == 'predict': # Only load model in prediction mode
            try:
                self.model = load_model(self.config.model_path)
                print(f"Loaded model from {self.config.model_path}")
            except Exception as e:
                print(f"ERROR: Failed to load model '{self.config.model_path}': {e}")
                # If GUI is on, show error there too?
                return False # Critical error

        # Load White Background Image
        try:
            self.white_bg = cv2.imread(self.config.bg_path)
            if self.white_bg is None:
                print(f"Warning: Could not load background image '{self.config.bg_path}'. Creating default.")
                self.white_bg = np.ones((SKELETON_SIZE, SKELETON_SIZE, 3), np.uint8) * 255
            elif self.white_bg.shape[:2] != (SKELETON_SIZE, SKELETON_SIZE):
                print(f"Warning: Background image is not {SKELETON_SIZE}x{SKELETON_SIZE}. Resizing.")
                self.white_bg = cv2.resize(self.white_bg, (SKELETON_SIZE, SKELETON_SIZE))
        except Exception as e:
            print(f"Warning: Exception loading background image '{self.config.bg_path}': {e}. Creating default.")
            self.white_bg = np.ones((SKELETON_SIZE, SKELETON_SIZE, 3), np.uint8) * 255
        return True

    def _initialize_state(self):
        """Initialize state variables for prediction/collection."""
        self.pts = [] # Hand landmarks
        # Prediction state
        self.current_symbol = "?"
        self.sentence = " "
        self.current_word = " "
        self.suggestions = [" ", " ", " ", " "]
        self.prev_char = ""
        self.prediction_history = [" "] * 10
        self.history_idx = -1
        # Collection state
        self.collection_count = 0
        self.collection_label = self.config.label
        self.save_enabled = False
        self.frame_step = 0
        self.saved_count_session = 0
        if self.config.mode == 'collect':
            self._update_collection_count()

    def _initialize_input_source(self):
        """Initialize video capture or load static image."""
        print("Initializing input source...")
        source = self.config.input
        # Check if input is an image file
        if isinstance(source, str) and any(source.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            self.is_image_mode = True
            print(f"Input is an image: {source}")
            try:
                self.static_image = cv2.imread(source)
                if self.static_image is None:
                    print(f"ERROR: Could not read image file: {source}")
                    if self.config.gui: self._show_error_gui(f"Failed to read image:\n{source}")
                    return False
                else:
                    print(f"Successfully loaded image: {source}")
                    return True
            except Exception as e:
                print(f"ERROR: Exception loading image {source}: {e}")
                if self.config.gui: self._show_error_gui(f"Exception loading image:\n{e}")
                return False
        else:
            # Input is video source (webcam index or video file path)
            self.is_image_mode = False
            video_source_val = source
            if isinstance(source, str) and source.isdigit():
                try:
                    video_source_val = int(source)
                except ValueError:
                    print(f"Warning: Invalid camera index '{source}'. Using default 0.")
                    video_source_val = 0

            print(f"Input is a video source: {video_source_val}")
            try:
                self.vs = cv2.VideoCapture(video_source_val)
                if not self.vs.isOpened():
                    print(f"ERROR: Could not open video source: {video_source_val}")
                    self.vs = None
                    if self.config.gui: self._show_error_gui(f"Failed to open video source:\n{video_source_val}")
                    return False
                else:
                    print(f"Successfully opened video source: {video_source_val}")
                    return True
            except Exception as e:
                print(f"ERROR: Exception opening video source {video_source_val}: {e}")
                if self.config.gui: self._show_error_gui(f"Exception opening video source:\n{e}")
                return False

    def _initialize_gui(self):
        """Setup Tkinter GUI elements."""
        print("Initializing GUI...")
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1300x800") # Adjusted size

        # --- Panels ---
        # Main video/image panel
        self.panel = tk.Label(self.root)
        self.panel.place(x=50, y=90, width=640, height=480)
        # Skeleton panel
        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=750, y=130, width=SKELETON_SIZE, height=SKELETON_SIZE)

        # --- Labels ---
        tk.Label(self.root, text="Sign Language To Text Conversion", font=("Courier", 30, "bold")).place(x=10, y=10)
        tk.Label(self.root, text="Character:", font=("Courier", 24, "bold")).place(x=10, y=580)
        self.panel3 = tk.Label(self.root, text=self.current_symbol, font=("Courier", 30)) # Prediction display
        self.panel3.place(x=220, y=580)
        tk.Label(self.root, text="Sentence :", font=("Courier", 24, "bold")).place(x=10, y=630)
        self.panel5 = tk.Label(self.root, text=self.sentence, font=("Courier", 24), wraplength=1000, justify=tk.LEFT) # Sentence display
        self.panel5.place(x=220, y=630)
        tk.Label(self.root, text="Suggestions:", fg="blue", font=("Courier", 20, "bold")).place(x=10, y=700)

        # --- Buttons ---
        button_font = ("Courier", 16)
        self.b1 = tk.Button(self.root, text=self.suggestions[0], font=button_font, command=self.action1)
        self.b1.place(x=220, y=700, width=200, height=40)
        self.b2 = tk.Button(self.root, text=self.suggestions[1], font=button_font, command=self.action2)
        self.b2.place(x=440, y=700, width=200, height=40)
        self.b3 = tk.Button(self.root, text=self.suggestions[2], font=button_font, command=self.action3)
        self.b3.place(x=660, y=700, width=200, height=40)
        self.b4 = tk.Button(self.root, text=self.suggestions[3], font=button_font, command=self.action4)
        self.b4.place(x=880, y=700, width=200, height=40)

        action_button_font = ("Courier", 18)
        self.clear_btn = tk.Button(self.root, text="Clear", font=action_button_font, command=self.clear_sentence)
        self.clear_btn.place(x=1100, y=700, width=150, height=40)
        self.speak_btn = tk.Button(self.root, text="Speak", font=action_button_font, command=self.speak_sentence)
        self.speak_btn.place(x=1100, y=640, width=150, height=40)

        # Disable prediction-related buttons if in collection mode
        if self.config.mode == 'collect':
            self.b1.config(state=tk.DISABLED)
            self.b2.config(state=tk.DISABLED)
            self.b3.config(state=tk.DISABLED)
            self.b4.config(state=tk.DISABLED)
            self.speak_btn.config(state=tk.DISABLED)
            self.clear_btn.config(state=tk.DISABLED)
            self.panel3.config(text="-") # No prediction in collect mode
            self.panel5.config(text="-")

    def _show_error_gui(self, error_message):
        """Initializes a simple GUI window just to show an error message."""
        if not self.root: # Create root only if it doesn't exist
             self.root = tk.Tk()
        self.root.title("Error")
        # Clear existing widgets if any
        for widget in self.root.winfo_children():
            widget.destroy()
        error_label = tk.Label(self.root, text=error_message, font=("Courier", 16), fg="red", wraplength=400, justify=tk.LEFT)
        error_label.pack(pady=30, padx=30)

    # --- Core Processing ---
    def _draw_skeleton(self, image, landmarks, crop_w, crop_h):
        """Draws the hand skeleton on a white background."""
        skeleton_image = self.white_bg.copy()
        if not landmarks:
            return skeleton_image # Return blank background if no landmarks

        # Calculate offset to center the skeleton
        skel_os_x = max(0, (SKELETON_SIZE - crop_w) // 2)
        skel_os_y = max(0, (SKELETON_SIZE - crop_h) // 2)

        try:
            # Draw connections
            for start_idx, end_idx in LANDMARK_CONNECTIONS:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    p1 = landmarks[start_idx]
                    p2 = landmarks[end_idx]
                    pt1 = (p1[0] + skel_os_x, p1[1] + skel_os_y)
                    pt2 = (p2[0] + skel_os_x, p2[1] + skel_os_y)
                    # Check bounds before drawing
                    if 0 <= pt1[0] < SKELETON_SIZE and 0 <= pt1[1] < SKELETON_SIZE and \
                       0 <= pt2[0] < SKELETON_SIZE and 0 <= pt2[1] < SKELETON_SIZE:
                        cv2.line(skeleton_image, pt1, pt2, (0, 255, 0), 3)

            # Draw landmark points
            for i in range(len(landmarks)):
                pt = (landmarks[i][0] + skel_os_x, landmarks[i][1] + skel_os_y)
                if 0 <= pt[0] < SKELETON_SIZE and 0 <= pt[1] < SKELETON_SIZE:
                    cv2.circle(skeleton_image, pt, 3, (0, 0, 255), cv2.FILLED) # Filled circle

        except IndexError:
            print("Error: Landmark index out of range during drawing.")
        except Exception as e:
            print(f"Error drawing skeleton: {e}")

        return skeleton_image

    def _process_frame(self, frame):
        """Detects hand, creates skeleton, predicts (if mode is 'predict'), and updates state."""
        display_frame = cv2.flip(frame, 1)
        frame_copy = np.array(display_frame) # For potential cropping
        skeleton_image = None
        hand_found = False
        landmarks_for_prediction = []

        # --- Hand Detection ---
        # Use detector.findHands which returns [hands, image_with_drawing]
        hands_data = self.detector.findHands(display_frame, draw=False) # Don't draw on original
        processed_display_frame = hands_data[1] # This is the frame cvzone returns (might have drawings if draw=True)

        if hands_data[0]: # Check if the hands list is not empty
            hand = hands_data[0][0] # Get the first hand
            x, y, w, h = hand['bbox']
            landmarks_abs = hand['lmList'] # Landmarks relative to the full frame

            # --- Crop Hand Region (for potential secondary processing or display) ---
            y_start = max(0, y - OFFSET)
            y_end = min(frame_copy.shape[0], y + h + OFFSET)
            x_start = max(0, x - OFFSET)
            x_end = min(frame_copy.shape[1], x + w + OFFSET)
            img_cropped = frame_copy[y_start:y_end, x_start:x_end]

            if img_cropped.size > 0:
                hand_found = True
                crop_h, crop_w = img_cropped.shape[:2]

                # --- Get Landmarks Relative to Crop (for skeleton drawing) ---
                # Adjust absolute landmarks to be relative to the cropped region's top-left corner
                landmarks_rel_crop = []
                valid_landmarks = True
                for lm_abs in landmarks_abs:
                     lm_rel_x = lm_abs[0] - x_start
                     lm_rel_y = lm_abs[1] - y_start
                     # Basic check if relative landmarks are within crop bounds (can be slightly outside due to offset)
                     # if not (0 <= lm_rel_x < crop_w and 0 <= lm_rel_y < crop_h):
                     #     # This might happen if hand is near edge and offset goes out of bounds
                     #     # print("Warning: Landmark outside crop bounds.")
                     #     # valid_landmarks = False
                     #     # break
                     #     pass # Allow slightly out-of-bounds for drawing robustness
                     landmarks_rel_crop.append([lm_rel_x, lm_rel_y]) # Store only x, y

                if valid_landmarks:
                    landmarks_for_prediction = landmarks_rel_crop # Use these for prediction rules
                    # --- Create Skeleton Image ---
                    skeleton_image = self._draw_skeleton(self.white_bg, landmarks_rel_crop, crop_w, crop_h)

                    # --- Prediction (only in predict mode) ---
                    if self.config.mode == 'predict' and self.model:
                        self._predict_sign(skeleton_image, landmarks_for_prediction)
                    elif self.config.mode == 'predict' and not self.model:
                         self.current_symbol = "No Model"

        # --- Update GUI / Display ---
        self._update_display(display_frame, skeleton_image) # Pass original flipped frame

        return skeleton_image # Return skeleton for data collection

    def _predict_sign(self, skeleton_image, landmarks):
        """Predicts the sign based on the skeleton image and landmarks."""
        if skeleton_image is None or not landmarks:
            self.current_symbol = "-"
            return

        # Reshape image for the model
        img_for_model = skeleton_image.reshape(1, SKELETON_SIZE, SKELETON_SIZE, 3)

        # Get model prediction probabilities
        try:
            prob = np.array(self.model.predict(img_for_model)[0], dtype='float32')
        except Exception as e:
            print(f"Error during model prediction: {e}")
            self.current_symbol = "Pred Error"
            return

        # Find top predictions (indices)
        ch1_idx = np.argmax(prob)
        prob[ch1_idx] = 0
        ch2_idx = np.argmax(prob)
        pl = [ch1_idx, ch2_idx] # Top 2 prediction indices

        # Store original prediction before heuristics
        pred_group_idx = ch1_idx
        self.pts = landmarks # Store landmarks for heuristic rules

        # ======================================================================
        # --- Heuristic Rules (Copied and Corrected) ---
        # ======================================================================
        # condition for [Aemnst] group (index 0)
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and
                    self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                pred_group_idx = 0

        # condition for [o][s] (group 0 vs group 2)
        l = [[2, 2], [2, 1]]
        if pl in l:
            if self.pts[5][0] < self.pts[4][0]:
                pred_group_idx = 0

        # condition for [c0][aemnst] (group 2 vs group 0)
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and
                    self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and
                    self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                pred_group_idx = 2

        # condition for [c0][aemnst] (group 2 vs group 0) - different rule
        l = [[6, 0], [6, 6], [6, 2]]
        if pl in l:
            if distance(self.pts[8], self.pts[16]) < 52:
                pred_group_idx = 2

        # condition for [gh][bdfikruvw] (group 3 vs group 1)
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and
                    self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][0] and
                    self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and
                    self.pts[0][0] < self.pts[20][0]):
                pred_group_idx = 3

        # con for [gh][l] (group 3 vs group 4)
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                pred_group_idx = 3

        # con for [gh][pqz] (group 3 vs group 5)
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                pred_group_idx = 3

        # con for [l][x] (group 4 vs group 6)
        l = [[6, 4], [6, 1], [6, 2]]
        if pl in l:
            if distance(self.pts[4], self.pts[11]) > 55:
                pred_group_idx = 4

        # con for [l][d] (group 4 vs group 1)
        l = [[1, 4], [1, 6], [1, 1]]
        if pl in l:
            if (distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and
                    self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                pred_group_idx = 4

        # con for [l][gh] (group 4 vs group 3)
        l = [[3, 6], [3, 4]]
        if pl in l:
            if self.pts[4][0] < self.pts[0][0]:
                pred_group_idx = 4

        # con for [l][c0] (group 4 vs group 2)
        l = [[2, 2], [2, 5], [2, 4]]
        if pl in l:
            if self.pts[1][0] < self.pts[12][0]:
                pred_group_idx = 4

        # con for [gh][z] (group 5 vs group 3)
        l = [[3, 6], [3, 5], [3, 4]]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and
                    self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and \
                    self.pts[4][1] > self.pts[10][1]:
                pred_group_idx = 5

        # con for [gh][pq] (group 5 vs group 3)
        l = [[3, 2], [3, 1], [3, 6]]
        if pl in l:
            if (self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and
                    self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][1] + 17 > self.pts[20][1]):
                pred_group_idx = 5

        # con for [l][pqz] (group 5 vs group 4)
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                pred_group_idx = 5

        # con for [pqz][aemnst] (group 5 vs group 0)
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        if pl in l:
            if (self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and
                    self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]):
                pred_group_idx = 5

        # con for [pqz][yj] (group 7 vs group 5)
        l = [[5, 7], [5, 2], [5, 6]]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                pred_group_idx = 7

        # con for [l][yj] (group 7 vs group 4)
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                pred_group_idx = 7

        # con for [x][yj] (group 7 vs group 6)
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                pred_group_idx = 7

        # condition for [x][aemnst] (group 6 vs group 0)
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                pred_group_idx = 6

        # condition for [yj][x] (group 6 vs group 7)
        l = [[7, 2]]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                pred_group_idx = 6

        # condition for [c0][x] (group 6 vs group 2)
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        if pl in l:
            if distance(self.pts[8], self.pts[16]) > 50:
                pred_group_idx = 6

        # con for [l][x] (group 6 vs group 4)
        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        if pl in l:
            if distance(self.pts[4], self.pts[11]) < 60:
                pred_group_idx = 6

        # con for [x][d] (group 6 vs group 1)
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                pred_group_idx = 6

        # --- Group 1 (bfdikruvw) Conditions ---
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and
                    self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                pred_group_idx = 1

        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and
                    self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                pred_group_idx = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                pred_group_idx = 1

        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and
                 self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and
                    (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                pred_group_idx = 1

        l = [[4, 1], [4, 2], [4, 4]]
        if pl in l:
            if (distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and
                    self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                pred_group_idx = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and
                 self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and
                    (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                pred_group_idx = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                pred_group_idx = 1

        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and
                 self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                pred_group_idx = 1

        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
               (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and
                self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                pred_group_idx = 7 # This rule actually sets to group 7

        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and
                 self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1])) and \
                    self.pts[4][1] > self.pts[14][1]:
                pred_group_idx = 1

        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        if pl in l:
            wrist_far_left = (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and
                              self.pts[0][0] + fg < self.pts[16][0] and self.pts[0][0] + fg < self.pts[20][0])
            wrist_far_right = (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and
                               self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0])
            thumb_middle_close = distance(self.pts[4], self.pts[11]) < 50
            if not wrist_far_left and not wrist_far_right and thumb_middle_close:
                pred_group_idx = 1

        l = [[5, 0], [5, 5], [0, 1]]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and
                    self.pts[14][1] > self.pts[16][1]):
                pred_group_idx = 1
        # ======================================================================
        # --- Subgroup Classification (Refining pred_group_idx to a specific letter) ---
        # ======================================================================
        final_char = "?" # Default

        if pred_group_idx == 0: # Group [A, E, M, N, S, T]
            final_char = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]: final_char = 'A'
            elif self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]: final_char = 'T'
            elif self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]: final_char = 'E'
            elif self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]: final_char = 'M'
            elif self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]: final_char = 'N'

        elif pred_group_idx == 1: # Group [B, D, F, I, K, R, U, V, W]
            final_char = '?' # Default for group 1
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]): final_char = 'B'
            elif (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]): final_char = 'D'
            elif (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]): final_char = 'F'
            elif (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]): final_char = 'I'
            elif (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][1]): final_char = 'W'
            elif (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] < self.pts[9][1]: final_char = 'K'
            elif ((distance(self.pts[8], self.pts[12]) - distance(self.pts[6], self.pts[10])) < 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]): final_char = 'U'
            elif ((distance(self.pts[8], self.pts[12]) - distance(self.pts[6], self.pts[10])) >= 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]): final_char = 'V'
            elif (self.pts[8][0] > self.pts[12][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]): final_char = 'R'

        elif pred_group_idx == 2: final_char = 'C' if distance(self.pts[12], self.pts[4]) > 42 else 'O'
        elif pred_group_idx == 3: final_char = 'G' if distance(self.pts[8], self.pts[12]) > 72 else 'H'
        elif pred_group_idx == 4: final_char = 'L'
        elif pred_group_idx == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]: final_char = 'Z' if self.pts[8][1] < self.pts[5][1] else 'Q'
            else: final_char = 'P'
        elif pred_group_idx == 6: final_char = 'X'
        elif pred_group_idx == 7: final_char = 'Y' if distance(self.pts[8], self.pts[4]) > 42 else 'J'

        # --- Special Character Rules (Override final_char) ---
        # Store the potential letter before checking special chars
        potential_letter = final_char
        original_group = pred_group_idx # Group index after heuristics but before subgroup

        # Space condition
        # Check if original group was potentially one that could map to space gesture
        if original_group in [1, 0, 6, 7]:
             # Check for specific finger posture: Index down, Middle/Ring down, Pinky up
             if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and
                 self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                 final_char = " " # Space

        # Next condition
        # Check if the potential letter was E, Y, or B
        is_potentially_E = (potential_letter == 'E')
        is_potentially_Y = (potential_letter == 'Y')
        is_potentially_B = (potential_letter == 'B')
        if is_potentially_E or is_potentially_Y or is_potentially_B:
             # Check for specific gesture: Thumb tip left of thumb base AND all fingers down
             if (self.pts[4][0] < self.pts[5][0]) and \
                (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and
                 self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                 final_char = "next"

        # Backspace condition
        # Check if potential letter was B, C, H, F, X or if it was already 'next'
        is_potentially_C = (potential_letter == 'C')
        is_potentially_H = (potential_letter == 'H')
        is_potentially_F = (potential_letter == 'F')
        is_potentially_X = (potential_letter == 'X')
        is_potentially_Next = (final_char == "next") # Check if overridden by 'next' rule

        if is_potentially_Next or is_potentially_B or is_potentially_C or is_potentially_H or is_potentially_F or is_potentially_X:
             # Check for specific gesture: Wrist right of all fingertips AND Thumb tip below all fingertips AND Thumb tip below all knuckles
             if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and
                 self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and \
                (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and
                 self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and \
                (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and
                 self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                 final_char = 'Backspace'
        # ======================================================================

        # --- Update Sentence Logic ---
        self.current_symbol = final_char

        # Add character to sentence based on 'next' trigger
        if final_char == "next" and self.prev_char != "next":
            char_to_add = None
            for i in range(1, len(self.prediction_history)):
                idx = (self.history_idx - i) % len(self.prediction_history)
                buffered_char = self.prediction_history[idx]
                if buffered_char != "next":
                    char_to_add = buffered_char
                    break
            if char_to_add is not None and char_to_add != " ":
                if char_to_add == "Backspace":
                    if len(self.sentence) > 1: self.sentence = self.sentence[:-1]
                else:
                    self.sentence = self.sentence + char_to_add

        # Add space (only if not repeating space)
        if final_char == " " and self.prev_char != " ":
             if self.sentence and self.sentence[-1] != ' ':
                 self.sentence = self.sentence + " "

        # Update history buffer
        self.prev_char = final_char
        self.history_idx = (self.history_idx + 1) % len(self.prediction_history)
        self.prediction_history[self.history_idx] = final_char

        # --- Update Word Suggestions ---
        self._update_suggestions()

    def _update_suggestions(self):
        """Updates word suggestions based on the current word."""
        self.suggestions = [" ", " ", " ", " "] # Reset suggestions
        if not self.spell_checker: return # Skip if enchant failed

        if len(self.sentence.strip()) > 0:
            last_space_idx = self.sentence.rfind(" ")
            self.current_word = self.sentence[last_space_idx + 1:]

            if len(self.current_word.strip()) > 0:
                try:
                    raw_suggestions = self.spell_checker.suggest(self.current_word)
                    for i in range(min(len(raw_suggestions), 4)):
                        self.suggestions[i] = raw_suggestions[i]
                except Exception as e:
                    print(f"Error getting suggestions for '{self.current_word}': {e}")
        else:
             self.current_word = " "


    def _update_display(self, display_frame, skeleton_image):
        """Update GUI elements or cv2 windows."""
        # --- Update GUI ---
        if self.config.gui and self.root:
            # Update main panel
            panel_w, panel_h = 640, 480
            h_orig, w_orig = display_frame.shape[:2]
            if w_orig > 0 and h_orig > 0:
                ratio = min(panel_w / w_orig, panel_h / h_orig)
                new_w, new_h = int(w_orig * ratio), int(h_orig * ratio)
                if new_w > 0 and new_h > 0:
                    resized_image = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    img_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    imgtk = ImageTk.PhotoImage(image=img_pil)
                    self.panel.imgtk = imgtk
                    self.panel.config(image=imgtk)

            # Update skeleton panel
            if skeleton_image is not None:
                img_pil_skel = Image.fromarray(skeleton_image) # Already RGB
                imgtk_skel = ImageTk.PhotoImage(image=img_pil_skel)
                self.panel2.imgtk = imgtk_skel
                self.panel2.config(image=imgtk_skel)
            else:
                # Clear skeleton panel if no hand detected (optional)
                # self.panel2.config(image='')
                pass

            # Update text labels/buttons
            self.panel3.config(text=self.current_symbol)
            self.panel5.config(text=self.sentence)
            self.b1.config(text=self.suggestions[0])
            self.b2.config(text=self.suggestions[1])
            self.b3.config(text=self.suggestions[2])
            self.b4.config(text=self.suggestions[3])

        # --- Update cv2 windows (if GUI is off) ---
        elif not self.config.gui:
            cv2.imshow("Frame", display_frame)
            if skeleton_image is not None:
                cv2.imshow("Skeleton", skeleton_image)
            elif cv2.getWindowProperty("Skeleton", cv2.WND_PROP_VISIBLE) >= 1:
                 # Close skeleton window if no hand is detected anymore
                 cv2.destroyWindow("Skeleton")

    # --- Button Actions ---
    def _apply_suggestion(self, suggested_word):
        """Helper to replace the last word part with a suggestion."""
        if suggested_word and suggested_word.strip() and self.current_word and self.current_word != " ":
            idx_space = self.sentence.rfind(" ")
            idx_word_start = self.sentence.find(self.current_word, idx_space if idx_space != -1 else 0)

            if idx_word_start != -1:
                self.sentence = self.sentence[:idx_word_start] + suggested_word.upper() + " "
                self.current_word = " " # Reset current word
                self._update_suggestions() # Clear suggestions display

    def action1(self): self._apply_suggestion(self.suggestions[0])
    def action2(self): self._apply_suggestion(self.suggestions[1])
    def action3(self): self._apply_suggestion(self.suggestions[2])
    def action4(self): self._apply_suggestion(self.suggestions[3])

    def speak_sentence(self):
        """Speaks the current sentence text."""
        text_to_speak = self.sentence.strip()
        if text_to_speak and self.speak_engine:
            try:
                print(f"Speaking: {text_to_speak}")
                self.speak_engine.say(text_to_speak)
                self.speak_engine.runAndWait()
            except Exception as e: print(f"Error during text-to-speech: {e}")
        elif not self.speak_engine: print("TTS engine not available.")
        else: print("Nothing to speak.")

    def clear_sentence(self):
        """Clears the sentence and suggestions."""
        print("Clearing text.")
        self.sentence = " "
        self.current_word = " "
        self._update_suggestions() # Clear suggestions display

    # --- Data Collection ---
    def _update_collection_count(self):
        """Updates the count of existing files for the current label."""
        label_dir = os.path.join(self.config.output_dir, self.config.label)
        os.makedirs(label_dir, exist_ok=True)
        self.collection_count = len(os.listdir(label_dir))
        print(f"Current label: {self.config.label}, Starting count: {self.collection_count}")

    def _handle_collection_keys(self, key, skeleton_image):
        """Handles key presses during data collection mode."""
        if key == 27: # ESC
            return False # Signal to exit loop

        if key == ord('n'): # Next label
            current_ord = ord(self.config.label)
            next_ord = current_ord + 1
            if next_ord > ord('Z'):
                next_ord = ord('A')
            self.config.label = chr(next_ord)
            self.collection_label = self.config.label
            self._update_collection_count()
            self.save_enabled = False # Disable saving when changing label
            self.saved_count_session = 0
            print(f"Switched to label: {self.config.label}")

        if key == ord('a'): # Toggle saving
            self.save_enabled = not self.save_enabled
            if self.save_enabled:
                print("--- STARTING SAVE ---")
                self.saved_count_session = 0 # Reset session count
            else:
                print("--- STOPPING SAVE ---")

        # Save image if enabled and skeleton exists
        if self.save_enabled and skeleton_image is not None:
            # Save every N frames (e.g., every 3rd frame like in data_collection_final)
            self.frame_step += 1
            if self.frame_step % 3 == 0:
                save_dir = os.path.join(self.config.output_dir, self.config.label)
                filename = f"{self.collection_count}.jpg" # Simple count-based filename
                save_path = os.path.join(save_dir, filename)
                try:
                    cv2.imwrite(save_path, skeleton_image)
                    # print(f"Saved: {save_path}")
                    self.collection_count += 1
                    self.saved_count_session += 1
                    # Optional: Stop saving after a certain number per session
                    # if self.saved_count_session >= 180:
                    #     self.save_enabled = False
                    #     print("--- Reached save limit for session ---")

                except Exception as e:
                    print(f"ERROR: Failed to save image {save_path}: {e}")
        return True # Continue loop

    # --- Main Loop Control ---
    def run(self):
        """Starts the main application loop based on configuration."""
        if self.config.mode == 'predict':
            if self.is_image_mode:
                # Process static image once and start GUI if enabled
                print("Processing static image...")
                self._process_frame(self.static_image)
                if self.config.gui and self.root:
                    self.root.mainloop()
                elif not self.config.gui:
                     # Keep window open until key press in non-GUI image mode
                     print("Press any key in an OpenCV window to exit.")
                     cv2.waitKey(0)
                     self.destructor()

            elif self.config.gui and self.root:
                # Start Tkinter loop for video/webcam with GUI
                print("Starting GUI video loop...")
                self.video_loop_gui()
                self.root.mainloop()
            elif not self.config.gui:
                # Start OpenCV loop for video/webcam without GUI
                print("Starting non-GUI video loop...")
                self.video_loop_nogui()

        elif self.config.mode == 'collect':
            # Start data collection loop
            print("Starting data collection loop...")
            self.collection_loop()

    def video_loop_gui(self):
        """Recursive loop for GUI mode (video/webcam)."""
        if self.vs is None or not self.vs.isOpened(): return
        ok, frame = self.vs.read()
        if not ok:
            print("Cannot read frame (GUI loop). End of video or error.")
            # Optionally close window automatically
            # self.destructor()
            return

        try:
            self._process_frame(frame)
        except Exception:
            print("Error in GUI video loop processing:", traceback.format_exc())
        finally:
            # Schedule next frame
            if self.vs and self.vs.isOpened(): # Check if vs still valid
                 self.root.after(20, self.video_loop_gui) # Adjust timing as needed

    def video_loop_nogui(self):
        """Standard while loop for non-GUI prediction mode (video/webcam)."""
        while True:
            if self.vs is None or not self.vs.isOpened(): break
            ok, frame = self.vs.read()
            if not ok:
                print("Cannot read frame (non-GUI loop). End of video or error.")
                break

            try:
                self._process_frame(frame)
            except Exception:
                print("Error in non-GUI video loop processing:", traceback.format_exc())

            # Check for exit key
            interrupt = cv2.waitKey(1) # Use waitKey(1) for video
            if interrupt & 0xFF == 27: # ESC key
                print("ESC pressed, exiting.")
                break
        self.destructor() # Clean up when loop finishes

    def collection_loop(self):
        """Loop for data collection mode."""
        while True:
            if self.vs is None or not self.vs.isOpened(): break
            ok, frame = self.vs.read()
            if not ok:
                print("Cannot read frame (collection loop). End of video or error.")
                break

            try:
                # Process frame to get skeleton (prediction logic is skipped)
                skeleton_img = self._process_frame(frame)

                # Display info on the frame
                info_text = f"Mode: Collect | Label: {self.config.label} | Count: {self.collection_count} | Saving: {'ON' if self.save_enabled else 'OFF'}"
                cv2.putText(self.display_frame_with_info, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                if not self.config.gui:
                    cv2.imshow("Frame", self.display_frame_with_info) # Show frame with info if no GUI

            except Exception:
                print("Error in collection loop processing:", traceback.format_exc())

            # Handle key presses
            interrupt = cv2.waitKey(1) # Crucial for cv2 windows and key handling
            if not self._handle_collection_keys(interrupt, skeleton_img):
                break # Exit if ESC was pressed

        self.destructor()

    @property
    def display_frame_with_info(self):
        """Returns the latest frame used for display, potentially with overlays."""
        # This assumes _process_frame updates a member variable like self.latest_display_frame
        # For simplicity, let's just return a placeholder or the raw frame if needed
        # A more robust implementation would store the frame shown in the GUI/cv2 window
        if hasattr(self, 'latest_processed_display_frame'):
             return self.latest_processed_display_frame.copy()
        else:
             # Fallback - create a black frame with text
             fallback = np.zeros((480, 640, 3), dtype=np.uint8)
             cv2.putText(fallback, "No frame available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
             return fallback


    def destructor(self):
        """Closes the application and releases resources."""
        print("Closing Application...")
        if self.config.mode == 'predict':
             print("Prediction history:", self.prediction_history)
        if hasattr(self, 'root') and self.root:
            try:
                self.root.destroy()
            except tk.TclError:
                pass # Ignore error if root already destroyed
        if self.vs is not None and self.vs.isOpened():
            self.vs.release()
            print("Video source released.")
        cv2.destroyAllWindows()
        print("OpenCV windows closed.")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Recognition/Collection Tool")
    parser.add_argument("-i", "--input", default='0',
                        help="Input source: webcam index ('0', '1', ...), video file path, or image file path.")
    parser.add_argument("-m", "--mode", default='predict', choices=['predict', 'collect'],
                        help="Operating mode: 'predict' or 'collect'.")
    parser.add_argument("-g", "--gui", default='on', choices=['on', 'off'],
                        help="Enable or disable the Tkinter GUI ('on' or 'off').")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help=f"Path to the Keras model file (default: {DEFAULT_MODEL_PATH}).")
    parser.add_argument("--bg", default=DEFAULT_BG_PATH,
                        help=f"Path to the white background image (default: {DEFAULT_BG_PATH}).")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory for data collection mode.")
    parser.add_argument("-l", "--label", default=None,
                        help="Starting label (e.g., 'A') for data collection mode.")

    args = parser.parse_args()
    config = Config(args)

    print(f"--- Configuration ---")
    print(f"Input: {config.input}")
    print(f"Mode: {config.mode}")
    print(f"GUI: {'Enabled' if config.gui else 'Disabled'}")
    if config.mode == 'predict':
        print(f"Model: {config.model_path}")
    print(f"Background: {config.bg_path}")
    if config.mode == 'collect':
        print(f"Output Dir: {config.output_dir}")
        print(f"Start Label: {config.label}")
    print(f"---------------------")


    app = SignPredictorApp(config)

    # Start the application loop if initialization was successful
    if (config.gui and hasattr(app, 'root') and app.root) or \
       (not config.gui and (app.vs is not None or app.is_image_mode)):
        app.run()
    else:
        print("Application failed to initialize properly. Exiting.")
        # Attempt to show error GUI if it was created
        if config.gui and hasattr(app, 'root') and app.root:
             app.root.mainloop()

