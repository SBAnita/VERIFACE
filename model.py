import os
import cv2
import numpy as np
import pickle
import traceback
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "model.pkl"

# -------------------------------------------------
# Utility: crop face + generate simple embedding
# -------------------------------------------------
def crop_face_and_embed(bgr_image, detection):
    h, w = bgr_image.shape[:2]
    bbox = detection.location_data.relative_bounding_box

    x1 = int(max(0, bbox.xmin * w))
    y1 = int(max(0, bbox.ymin * h))
    x2 = int(min(w, (bbox.xmin + bbox.width) * w))
    y2 = int(min(h, (bbox.ymin + bbox.height) * h))

    if x2 <= x1 or y2 <= y1:
        return None

    face = bgr_image[y1:y2, x1:x2]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (32, 32), interpolation=cv2.INTER_AREA)

    emb = face.flatten().astype(np.float32) / 255.0
    return emb


# -------------------------------------------------
# Extract embedding from uploaded image
# -------------------------------------------------
def extract_embedding_for_image(stream):
    data = stream.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return None

    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as face_detector:

        results = face_detector.process(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )

        if not results.detections:
            return None

        return crop_face_and_embed(img, results.detections[0])


# -------------------------------------------------
# Model helpers
# -------------------------------------------------
def load_model_if_exists():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_with_model(clf, emb):
    proba = clf.predict_proba([emb])[0]
    idx = np.argmax(proba)
    return clf.classes_[idx], float(proba[idx])


# -------------------------------------------------
# Background training function
# -------------------------------------------------
def train_model_background(dataset_dir, progress_callback=None):
    from app import write_train_status

    try:
        print("TRAINING THREAD STARTED")

        write_train_status({
            "running": True,
            "progress": 0,
            "message": "Initializing training"
        })

        X, y = [], []

        student_dirs = [
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ]

        if not student_dirs:
            write_train_status({
                "running": False,
                "progress": 0,
                "message": "No students found"
            })
            return

        total = len(student_dirs)

        with mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        ) as face_detector:

            for idx, sid in enumerate(student_dirs):
                folder = os.path.join(dataset_dir, sid)
                images = [
                    f for f in os.listdir(folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                for img_name in images:
                    img = cv2.imread(os.path.join(folder, img_name))
                    if img is None:
                        continue

                    results = face_detector.process(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    )

                    if not results.detections:
                        continue

                    emb = crop_face_and_embed(img, results.detections[0])
                    if emb is None:
                        continue

                    X.append(emb)
                    y.append(int(sid))

                progress = int((idx + 1) / total * 80)
                write_train_status({
                    "running": True,
                    "progress": progress,
                    "message": f"Processed {idx + 1}/{total} students"
                })

        if not X:
            write_train_status({
                "running": False,
                "progress": 0,
                "message": "No valid face images found"
            })
            return

        X = np.stack(X)
        y = np.array(y)

        write_train_status({
            "running": True,
            "progress": 85,
            "message": "Training RandomForest model"
        })

        clf = RandomForestClassifier(
            n_estimators=150,
            n_jobs=-1,
            random_state=42
        )
        clf.fit(X, y)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)

        write_train_status({
            "running": False,
            "progress": 100,
            "message": "Training completed successfully"
        })

    except Exception as e:
        traceback.print_exc()
        write_train_status({
            "running": False,
            "progress": 0,
            "message": f"Training failed: {e}"
        })
