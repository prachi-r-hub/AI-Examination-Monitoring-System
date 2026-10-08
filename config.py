import os

# ==========================================
# CENTRALIZED CONFIGURATION FOR EXAM MONITORING
# ==========================================

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Paths
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "runs", "classroom_detector", "weights", "best.pt")
COCO_PERSON_MODEL_PATH = os.path.join(BASE_DIR, "yolov8s.pt")

# Default Detection Parameters
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45

# Consecutive Frame & Duration Parameters
DEFAULT_REQUIRED_CONSECUTIVE_FRAMES = 3
DEFAULT_MISSED_FRAME_TOLERANCE = 1
SPATIAL_MATCH_IOU_THRESHOLD = 0.15
SPATIAL_MATCH_CENTER_DISTANCE = 150.0

# Person Tracking & Association Parameters
ENABLE_PERSON_TRACKING = True
PERSON_PROXIMITY_BUFFER_PX = 80.0  # Pixel expansion buffer for associating objects with a person

# Object/Paper Passing Detection Parameters
ENABLE_OBJECT_PASSING_DETECTION = True
PASSING_MIN_DISPLACEMENT_PX = 100.0
PASSING_MAX_DURATION_SECONDS = 3.0
PASSING_MIN_CONFIDENCE = 0.30

# Duration Significance Thresholds (in seconds)
SHORT_DURATION_THRESHOLD = 2.0  # < 2.0s -> LOW significance
LONG_DURATION_THRESHOLD = 5.0   # >= 5.0s -> HIGH significance

# Head Movement Detection Parameters
REQUIRED_HEAD_TURNS = 3                 # Required head turns within time window
HEAD_MOVEMENT_WINDOW_SECONDS = 4.0      # Time window in seconds
HEAD_OFFSET_THRESHOLD = 0.10            # Center ratio offset threshold for LEFT / RIGHT classification

# Voice Alert Parameters
ENABLE_VOICE_ALERTS = True
VOICE_ALERT_COOLDOWN_SECONDS = 10.0
MSG_VOICE_RESTRICTED = "Warning. Restricted object detected."
MSG_VOICE_SUSPICIOUS = "Suspicious activity detected. Please review."

# Transparent Rule-Based Monitoring Risk Score Values (0-100)
# NOTE: It is a risk indicator, NOT cheating probability.
RISK_PROHIBITED_OBJECT = 25
RISK_PERSISTENT_PROHIBITED_OBJECT = 15
RISK_OBJECT_HANDLING = 20
RISK_REPEATED_HEAD_MOVEMENT = 10
RISK_POSSIBLE_OBJECT_PASSING = 25

# Class ID & Category Mappings
CLASS_MAPPING = {
    0: "notebook",
    1: "paper",
    2: "pen",
    3: "pencil",
    4: "phone",
    5: "smartwatch",
    6: "earbuds"
}

ALLOWED_CLASSES = {0: "notebook", 1: "paper", 2: "pen", 3: "pencil"}
RESTRICTED_CLASSES = {4: "phone", 5: "smartwatch", 6: "earbuds"}
PROHIBITED_CLASSES = RESTRICTED_CLASSES

# Visual Styling / Color Definitions (BGR for OpenCV)
COLOR_ALLOWED = (0, 255, 0)      # Green (BGR)
COLOR_RESTRICTED = (0, 0, 255)   # Red (BGR)
COLOR_PROHIBITED = (0, 0, 255)   # Red (BGR)
COLOR_PERSON = (255, 191, 0)     # Deep Cyan/Blue for Person
COLOR_UNKNOWN = (0, 165, 255)    # Orange/Yellow
COLOR_TEXT = (255, 255, 255)     # White

# Cautious Terminology Constants (No accusatory language)
MSG_SUSPICIOUS_BEHAVIOUR = "Possible suspicious behaviour"
MSG_REPEATED_HEAD_MOVEMENT = "Possible repeated head movement"
MSG_RESTRICTED_HANDLING = "Restricted-object handling detected"
MSG_OBJECT_PASSING = "Possible object passing — review recommended"
MSG_NORMAL = "Normal monitoring status"
MSG_BRIEF_PRESENCE = "Brief restricted object presence"
MSG_EXTENDED_HANDLING = "Extended restricted-object handling detected — review recommended"

# Evidence output paths
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
LIVE_INCIDENTS_DIR = os.path.join(EVIDENCE_DIR, "live_incidents")
VIDEO_INCIDENTS_DIR = os.path.join(EVIDENCE_DIR, "video_incidents")

# Ensure required directories exist
for path in [EVIDENCE_DIR, LIVE_INCIDENTS_DIR, VIDEO_INCIDENTS_DIR]:
    os.makedirs(path, exist_ok=True)
