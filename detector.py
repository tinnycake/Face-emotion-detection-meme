import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from deepface import DeepFace
import time
import threading 

EMOTION_IMAGES = {
    "happy"   : "smile.jpg",
    "surprise": "surprised.png",
    "neutral"     : "normal.jpeg",
    "disgust" : "disgust.png",
    "sad" : "disgust.png",
    "angry" : "disgust.png",
}

CONFIDENCE_THRESHOLD = 70   
DISPLAY_SECONDS = 1.0     
loaded_images = {}
for emotion, path in EMOTION_IMAGES.items():
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Warning: Could not load image for '{emotion}': {path}")
    else:
        loaded_images[emotion] = cv2.resize(img, (250, 250))
        print(f"Loaded image for '{emotion}': {path}")


# --- SHARED STATE between main thread and AI thread ---
# These variables are shared between the video loop and the AI analysis
current_emotion = "neutral"   # what emotion is detected right now
current_confidence = 0        # how confident the AI is
last_detected_time = 0        # when we last detected the trigger emotion
analysis_running = False      # is the AI currently busy analysing?


# --- BACKGROUND AI FUNCTION ---
# This function runs in a separate "thread" (like a parallel process)
# So the video feed stays smooth while the AI thinks in the background
def analyze_emotion(frame):
    global current_emotion, current_confidence, last_detected_time, analysis_running

    try:
        # model_name="Facenet" is lighter and faster than the default
        # detector_backend="opencv" is the fastest face finder
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",  # fastest detector
        )

        emotion = result[0]["dominant_emotion"]
        confidence = result[0]["emotion"][emotion]

        current_emotion = emotion
        current_confidence = confidence

        # If a trigger emotion is detected with enough confidence, record the time
        if emotion in EMOTION_IMAGES and confidence > CONFIDENCE_THRESHOLD:
            last_detected_time = time.time()

    except Exception:
        pass

    # Mark analysis as done so the next one can start
    analysis_running = False


# --- START WEBCAM ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # set resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Camera started! Press Q to quit.")

last_analysis_time = 0   # when we last started an AI analysis

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    # --- TRIGGER BACKGROUND ANALYSIS ---
    # Start a new AI analysis every 0.4 seconds, but only if previous one finished
    now = time.time()
    if not analysis_running and (now - last_analysis_time > 0.4):
        analysis_running = True
        last_analysis_time = now
        # threading.Thread runs analyze_emotion() in the background
        # daemon=True means it auto-closes when the main program closes
        t = threading.Thread(target=analyze_emotion, args=(frame.copy(),), daemon=True)
        t.start()

    # --- SHOW CURRENT EMOTION TEXT ---
    cv2.putText(display_frame,
                f"{current_emotion} ({current_confidence:.0f}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    # --- SHOW OVERLAY IMAGE ---
    # Check if a trigger emotion was detected recently
    if now - last_detected_time < DISPLAY_SECONDS:
        # Figure out WHICH emotion was last triggered
        if current_emotion in loaded_images:
            overlay = loaded_images[current_emotion]
            h, w = overlay.shape[:2]
            fh, fw = display_frame.shape[:2]
            x_offset = fw - w - 20
            y_offset = 20
            display_frame[y_offset:y_offset+h, x_offset:x_offset+w] = overlay[:, :, :3]

    cv2.imshow("Expression Detector", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()