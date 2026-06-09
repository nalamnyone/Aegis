# Importing Libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import PIL.Image
import io
import streamlit as st

# Using st.cache_resource so it only loads once per app session


@st.cache_resource
def load_fight_model():
    model_path = hf_hub_download(
        repo_id="Musawer14/fight_detection_yolov8",
        filename="Yolo_nano_weights.pt"
    )
    return YOLO(model_path)


# The function that runs fight detection on an image
def analyze_fight(image_bytes, person_count):

    # Rule — fight detection only runs if 2 or more people are detected
    # A single person on the ground is never a fight
    if person_count < 2:
        return "FIGHT DETECTION: Skipped — only one person detected. Not a fight scenario.", False, 0.0

    # Opening the image
    image = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Running the fight detection model
    fight_model = load_fight_model()
    results = fight_model(image, verbose=False)

    # Processing the results
    violence_detected = False
    violence_confidence = 0.0
    non_violence_confidence = 0.0
    detections = []

    for result in results:
        for box in result.boxes:
            label = fight_model.names[int(box.cls)]
            conf = float(box.conf)
            detections.append({"label": label, "confidence": conf})

            if label == "violence" and conf > violence_confidence:
                violence_confidence = conf
                violence_detected = True
            elif label == "non_violence" and conf > non_violence_confidence:
                non_violence_confidence = conf

    if violence_detected:
        fight_summary = f"FIGHT DETECTION: Violence detected ({violence_confidence:.0%} confidence). Physical altercation likely present."
        is_fight = True
    elif detections:
        fight_summary = f"FIGHT DETECTION: No violence detected. Scene appears non-violent ({non_violence_confidence:.0%} confidence)."
        is_fight = False
    else:
        fight_summary = "FIGHT DETECTION: No clear detection — scene ambiguous. Claude to make final judgment."
        is_fight = False

    return fight_summary, is_fight, violence_confidence
