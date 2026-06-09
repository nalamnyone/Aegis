# Importing the necessary libraries
import os
import time
from datetime import datetime

import cv2
import PIL.Image
from dotenv import load_dotenv
import anthropic
import base64
from ultralytics import YOLO
from action_recognition import analyze_fight

# Loading environment variables from the .env file
load_dotenv()

# Connecting to Claude using the API key
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

_yolo_model = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model

RECENT_PERSON_WINDOW_SEC = 4

# Only flagging objects that are actually relevant to campus safety
RELEVANT_CLASSES = {
    "person", "backpack", "handbag", "knife", "scissors",
    "cell phone", "bottle", "chair", "bench", "car",
    "motorcycle", "bicycle", "truck", "bus"
}


def severity_score(severity: str) -> int:
    return {"CLEAR": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(severity, 0)


def extract_severity(alert_text: str) -> str:
    if "SEVERITY: HIGH" in alert_text:
        return "HIGH"
    elif "SEVERITY: MEDIUM" in alert_text:
        return "MEDIUM"
    elif "SEVERITY: LOW" in alert_text:
        return "LOW"
    return "CLEAR"


def classify_incident_type(alert_text: str) -> str:
    text = alert_text.lower()

    if any(word in text for word in ["fight", "altercation", "violence", "shoving", "pushing"]):
        return "Fight / Physical Altercation"
    elif any(word in text for word in ["fallen", "collapsed", "unconscious", "on the ground", "lying", "stairs"]):
        return "Person Fallen / Unconscious"
    elif any(word in text for word in ["medical", "distress", "convulsing", "unresponsive"]):
        return "Medical Emergency"
    elif any(word in text for word in ["harassment", "bullying", "cornered", "intimidated"]):
        return "Harassment / Bullying"
    elif any(word in text for word in ["loitering", "lingering"]):
        return "Loitering / Suspicious Presence"
    elif any(word in text for word in ["vandalism", "damaging property"]):
        return "Vandalism / Property Damage"
    elif any(word in text for word in ["crowd", "gathering", "panic", "running"]):
        return "Unusual Crowd / Panic"
    else:
        return "Suspicious Behaviour"


# Merging logic for nearby incidents of the same type
def should_merge(last_incident: dict, new_incident: dict, merge_window_sec: float) -> bool:
    same_type = last_incident["incident_type"] == new_incident["incident_type"]
    close_in_time = (
        new_incident["timestamp"] - last_incident["end_timestamp"]) <= merge_window_sec
    return same_type and close_in_time


# Saving annotated frames to disk for incident records
def save_annotated_frame(frame_bgr, output_path: str) -> None:
    cv2.imwrite(output_path, frame_bgr)


def quick_yolo_scan(frame_bgr):
    """
    Runs a lightweight YOLO pass on a frame before deciding to call Claude.
    Returns (person_found, list_of_detected_labels).
    Keeps Claude API calls to a minimum by pre-filtering empty frames.
    """
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_pil = PIL.Image.fromarray(image_rgb)
    _yolo = _get_yolo()
    results = _yolo(image_pil, verbose=False)

    person_found = False
    detected_labels = []

    for result in results:
        for box in result.boxes:
            label = _yolo.names[int(box.cls)]
            conf = float(box.conf)

            # Only tracking relevant classes above 25% confidence
            if conf > 0.25 and label in RELEVANT_CLASSES:
                detected_labels.append(label)
                if label == "person":
                    person_found = True

    return person_found, detected_labels


# The main function that analyzes a single video frame with YOLO, fight detection, and Claude
def analyze_frame(frame_bgr, location: str, timestamp_sec: float):
    """
    Full analysis of a single video frame.

    Steps:
    1. Run YOLO — detect relevant objects only, draw bounding boxes
    2. Run fight detection if 2+ people are present
    3. Build a detection summary string
    4. Send the frame and summary to Claude for threat assessment
    5. Return detection summary, alert, annotated frame, severity, and fight info

    Returns: (detection_summary, alert, annotated_frame_bgr, severity,
              fight_summary, is_fight, fight_confidence)
    """
    time_now = datetime.now().strftime("%I:%M %p")

    # Making a copy so we can draw on the frame without affecting the original
    image_cv = frame_bgr.copy()

    # Converting from OpenCV BGR format to RGB for PIL
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_pil = PIL.Image.fromarray(image_rgb)

    # Running YOLO on the frame with confidence and IOU thresholds to reduce duplicate boxes
    # conf=0.25 matches the loop filter below — using 0.40 here would silently discard
    # occluded people and weapons (like knives held close to the body) before we even see them
    _yolo = _get_yolo()
    results = _yolo(image_pil, conf=0.25, iou=0.45, verbose=False)
    detections = []
    person_count = 0

    # Looping through YOLO detections and drawing bounding boxes
    for result in results:
        for box in result.boxes:
            label = _yolo.names[int(box.cls)]
            conf = float(box.conf)

            # Skipping irrelevant objects
            if label not in RELEVANT_CLASSES:
                continue

            # Only keeping detections above 25% confidence
            if conf > 0.25:
                detections.append(f"{label} ({conf:.0%})")

                if label == "person":
                    person_count += 1

                # Getting bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Red for people, yellow for other objects
                color = (0, 0, 255) if label == "person" else (0, 255, 255)

                # Dynamically scaling the box thickness and text size based on image resolution
                img_height, img_width = image_cv.shape[:2]
                box_thickness = max(2, int(min(img_width, img_height) / 200))
                font_scale = max(0.6, min(img_width, img_height) / 800)
                font_thickness = max(2, int(min(img_width, img_height) / 400))

                # Drawing the bounding box
                cv2.rectangle(image_cv, (x1, y1), (x2, y2),
                              color, box_thickness)

                # Writing the label and confidence score
                cv2.putText(
                    image_cv,
                    f"{label} {conf:.0%}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    font_thickness
                )

    # Building the detection summary
    if detections:
        detection_summary = f"{person_count} person(s) detected. Objects: {', '.join(detections)}"
    else:
        detection_summary = "No objects detected by YOLO — Claude visual assessment only."

    # Running fight detection
    image_bytes = cv2.imencode('.jpg', frame_bgr)[1].tobytes()
    fight_summary, is_fight, fight_confidence = analyze_fight(
        image_bytes, person_count)

    prompt = f"""
    You are Aegis, an AI campus safety assistant at Taylor's University Malaysia.

    Camera location: {location}
    Current time: {time_now}
    Video timestamp: {timestamp_sec:.1f} seconds

    YOLO computer vision analysis:
    {detection_summary}

    Fight detection analysis:
    {fight_summary}

    NOTE: YOLO may undercount people in close-contact or dark scenes. Trust your
    own visual assessment over the YOLO count — if you see people, they are there.

    Carefully inspect this frame for REAL campus safety threats only.

    Check for:
    1. FIGHT or VIOLENCE — physical aggression, grappling, assault, pushing, shoving,
       one person restraining another, body contact that looks forceful or non-consensual
    2. WEAPON — any knife, blade, sharp object, or improvised weapon visible in anyone's hand
    3. PERSON FALLEN or COLLAPSED — someone on the ground, on stairs, injured, unconscious
    4. MEDICAL DISTRESS — clear signs of physical emergency or unresponsiveness
    5. UNUSUAL CROWD / PANIC — chaotic gathering, running, surrounding a person
    6. HARASSMENT or BULLYING — visible intimidation or cornering
    7. LOITERING — suspicious lingering in a way that may require security attention
    8. VANDALISM — visible property damage

    IMPORTANT RULES:
    - Do NOT flag normal walking, standing, carrying a bag, or using stairs as suspicious.
    - Do NOT invent danger if the frame looks normal.
    - Only raise an alert if there is clear visual evidence that security should act.
    - If uncertain, return CLEAR.
    - If you see a weapon in anyone's hand, always flag HIGH regardless of other context.
    - If two or more people appear to be physically struggling, always flag at least MEDIUM.

    If ANY real threat is present, respond in this exact format:

    SEVERITY: [HIGH / MEDIUM / LOW]

    [What is happening and exactly where]

    [Brief context of what security will find]

    ACTION: [One clear instruction]

    If NO real threat is present, respond with:
    CLEAR — No threats detected at this location.

    Under 60 words. Write like a human security officer.
    """

    image_b64 = base64.standard_b64encode(cv2.imencode('.jpg', frame_bgr)[1].tobytes()).decode("utf-8")
    alert = None
    for attempt in range(3):
        try:
            response = client.messages.create(
                model="claude-opus-4-8",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            alert = response.content[0].text
            break  # success — stop retrying

        except anthropic.RateLimitError:
            if attempt < 2:
                print(
                    f"[AEGIS] Rate limit hit at {timestamp_sec:.1f}s — waiting 15s before retry {attempt + 2}/3")
                time.sleep(15)
            else:
                alert = "CLAUDE_ERROR: Rate limit exhausted"
                print(f"[AEGIS] Claude rate limit exhausted at {timestamp_sec:.1f}s")
                break

        except Exception as e:
            alert = f"CLAUDE_ERROR: {str(e)}"
            print(f"[AEGIS] Claude failed at {timestamp_sec:.1f}s: {str(e)[:100]}")
            break

    if alert is None:
        alert = "CLAUDE_ERROR: No response returned"

    if alert and alert.startswith("CLAUDE_ERROR"):
        severity = "CLEAR"
    else:
        severity = extract_severity(alert)

    return detection_summary, alert, image_cv, severity, fight_summary, is_fight, fight_confidence


def analyze_video(
    video_path: str,
    location: str,
    sample_every_sec: float = 1.0,
    merge_window_sec: float = 3,
    ai_cooldown_sec: float = 2.0
):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Getting the video FPS and total frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    # Getting the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Converting seconds into a frame interval for sampling
    frame_interval = max(1, int(fps * sample_every_sec))

    incidents = []
    frame_index = 0

    # Tracking when Claude was last called to respect the cooldown
    last_ai_call_time = -999

    # Tracking when a person was last seen by YOLO
    last_person_seen_time = -999

    # Counts Claude calls — forces at least one call per video
    # even if YOLO misses everyone (dark scenes, close-contact grappling, fast motion)
    ai_call_count = 0

    # Creating a folder for saving incident frames
    os.makedirs("incident_frames", exist_ok=True)

    while True:
        ret, frame = cap.read()

        # Stop when the video ends
        if not ret:
            break

        # Only analyzing frames at the chosen time interval
        if frame_index % frame_interval == 0:
            timestamp_sec = frame_index / fps

            # Running a quick YOLO check first before deciding to call Claude
            person_found, _ = quick_yolo_scan(frame)

            # Updating the last time a person was seen
            if person_found:
                last_person_seen_time = timestamp_sec

            # Deciding whether Claude should be called on this frame
            cooldown_ready = (
                timestamp_sec - last_ai_call_time) >= ai_cooldown_sec
            recent_person_context = (
                timestamp_sec - last_person_seen_time) <= RECENT_PERSON_WINDOW_SEC

            # For short videos (under 10s): analyze every sampled frame — a 3-second
            # fight clip cannot afford to skip frames.
            # For longer videos: use the YOLO pre-filter and cooldown to save API calls.
            video_duration = total_frames / fps
            is_short_video = video_duration <= 10.0
            is_first_frame = (ai_call_count == 0)

            should_analyze = (
                is_short_video  # always analyze every sampled frame in short clips
                or is_first_frame  # always check at least the first frame of any video
                or (cooldown_ready and (person_found or recent_person_context))
            )

            if should_analyze:

                # Running the full analysis pipeline on this frame
                detection_summary, alert, annotated_frame, severity, fight_summary, is_fight, fight_confidence = analyze_frame(
                    frame, location, timestamp_sec
                )

                # Updating the last Claude call time
                last_ai_call_time = timestamp_sec
                ai_call_count += 1

                # Only logging non-clear results
                if severity != "CLEAR":
                    incident_type = classify_incident_type(alert)

                    # Saving the annotated frame that represents this incident moment
                    frame_filename = f"incident_frames/frame_{timestamp_sec:.1f}s.jpg"
                    save_annotated_frame(annotated_frame, frame_filename)

                    # Creating the new incident record
                    new_incident = {
                        "start_timestamp": timestamp_sec,
                        "end_timestamp": timestamp_sec,
                        "timestamp": timestamp_sec,
                        "Severity": severity,
                        "incident_type": incident_type,
                        "Detection": detection_summary,
                        "fight_summary": fight_summary,
                        "is_fight": is_fight,
                        "fight_confidence": fight_confidence,
                        "Alert Summary": alert[:120] + "..." if len(alert) > 120 else alert,
                        "alert": alert,
                        "frame_path": frame_filename,
                        # stores up to 3 frames for timeline display
                        "all_frame_paths": [frame_filename]
                    }

                    # Grouping incidents that are likely part of the same ongoing event
                    if incidents and should_merge(incidents[-1], new_incident, merge_window_sec):
                        # Merging with the previous incident
                        incidents[-1]["end_timestamp"] = timestamp_sec
                        incidents[-1]["timestamp"] = timestamp_sec
                        incidents[-1]["Detection"] = detection_summary
                        incidents[-1]["alert"] = alert
                        incidents[-1]["frame_path"] = frame_filename
                        incidents[-1]["fight_summary"] = fight_summary

                        # Keeping up to 3 frames from different moments for timeline display
                        if len(incidents[-1]["all_frame_paths"]) < 3:
                            incidents[-1]["all_frame_paths"].append(
                                frame_filename)

                        # Keeping the higher severity if the situation escalated
                        if severity_score(new_incident["Severity"]) > severity_score(incidents[-1]["Severity"]):
                            incidents[-1]["Severity"] = new_incident["Severity"]
                    else:
                        # Adding as a new separate incident
                        incidents.append(new_incident)

        frame_index += 1

    cap.release()
    return incidents, total_frames, fps
