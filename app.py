# Importing Libraries
import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime
import anthropic
import PIL.Image
import io
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from video_analysis import analyze_video
import tempfile
import base64
import streamlit.components.v1 as components
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from action_recognition import analyze_fight
import json

LOG_FILE = "incident_log.json"

# Functions to load and save the incident log as a JSON file


def load_log():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # Log file is corrupted or unreadable then start afresh
            return []
    return []

# Save the log to a JSON file


def save_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f)


# Function to generate a PDF report of the incident log
def generate_incident_report(incident_log):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Report Header
    elements.append(Paragraph("AEGIS Incident Report", styles['Title']))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", styles['Normal']))
    elements.append(
        Paragraph("Taylor's University Lakeside Campus", styles['Normal']))
    elements.append(Spacer(1, 20))

    if not incident_log:
        elements.append(Paragraph("No incidents recorded.", styles['Normal']))
    else:
        # Table headers
        data = [["Time", "Camera", "Severity", "Detection", "Alert Summary"]]

        for incident in incident_log:
            data.append([
                incident.get("Time", ""),
                Paragraph(incident.get("Camera", ""), styles['Normal']),
                incident.get("Severity", ""),
                Paragraph(incident.get("Detection", ""), styles['Normal']),
                Paragraph(incident.get("Alert Summary", "").replace("SEVERITY: HIGH", "")
                          .replace("SEVERITY: MEDIUM", "").replace("SEVERITY: LOW", "").strip(), styles['Normal']),
            ])

        table = Table(data, colWidths=[50, 100, 50, 120, 160])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.whitesmoke, colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer


load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Loading the YOLO model once when the app starts


@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")


yolo = load_yolo()


@st.cache_resource
def load_pose_model():
    return YOLO("yolo11n-pose.pt")


pose_model = load_pose_model()

# Initializing session state so nothing gets lost between interactions
if 'incident_log' not in st.session_state:
    st.session_state.incident_log = load_log()
if 'annotated_image' not in st.session_state:
    st.session_state.annotated_image = None
if 'alert' not in st.session_state:
    st.session_state.alert = None
if 'play_sound' not in st.session_state:
    st.session_state.play_sound = False
if 'detection_summary' not in st.session_state:
    st.session_state.detection_summary = None
if 'video_incidents' not in st.session_state:
    st.session_state.video_incidents = []
if 'video_result_frame' not in st.session_state:
    st.session_state.video_result_frame = None
if 'video_alert_text' not in st.session_state:
    st.session_state.video_alert_text = None
if 'video_alert_severity' not in st.session_state:
    st.session_state.video_alert_severity = None
if 'video_alert_timestamp' not in st.session_state:
    st.session_state.video_alert_timestamp = None
if 'video_alert_camera' not in st.session_state:
    st.session_state.video_alert_camera = None
if 'video_fight_summary' not in st.session_state:
    st.session_state.video_fight_summary = None
if 'video_all_frames' not in st.session_state:
    st.session_state.video_all_frames = []
if 'video_alert_type' not in st.session_state:
    st.session_state.video_alert_type = None
if 'pose_summary' not in st.session_state:
    st.session_state.pose_summary = None
if 'fight_summary' not in st.session_state:
    st.session_state.fight_summary = None
# New keys for Feature 1 (multi-camera grid) and Feature 2 (video timeline)
if 'multi_camera_results' not in st.session_state:
    st.session_state.multi_camera_results = []
if 'video_all_incidents' not in st.session_state:
    st.session_state.video_all_incidents = []

# Real Taylor's University Lakeside Campus camera locations based on floor plans
CAMPUS_CAMERAS = [
    # Ground Floor
    "Block A — Ground Floor, Main Entrance Corridor",
    "Block B — Ground Floor, Corridor",
    "Block C — Ground Floor, Truffles Restaurant Area",
    "Block D — Ground Floor, Corridor near Lift Lobby",
    "Block E — Ground Floor, Corridor near Lift Lobby",
    "Syopz Mall — Ground Floor Entrance",
    "University Square — Outdoor Area",
    "Drop Off Zone — Main Entrance",
    "Outdoor Amphitheatre — Main Area",
    "Lake Area — Outdoor Walkway",

    # First Floor
    "Block A — Level 1, Enrollment and Admissions Corridor",
    "Block B — Level 1, Corridor near Lift Lobby",
    "Block C — Level 1, Lecture Theatre C1.01 Corridor",
    "Block C — Level 1, Lecture Theatre C1.02 Corridor",
    "Block C — Level 1, Lecture Theatres C1.03 to C1.05 Corridor",
    "Block C — Level 1, Lecture Theatre C1.06 Corridor",
    "Block C — Level 1, Lecture Theatres C1.08 to C1.10 Corridor",
    "Block D — Level 1, Corridor near Lift Lobby",
    "Block D — Level 1, Bellevue Concourse",
    "Block E — Level 1, Corridor near Lift Lobby",
    "Crescent Walkway — Level 1",

    # Second Floor
    "Block A — Level 2, Campus Central Corridor",
    "Block B — Level 2, Grand Hall (TGH) Entrance",
    "Block C — Level 2, Library C2.01 Entrance",
    "Block C — Level 2, Library C2.01 Interior",
    "Block C — Level 2, Arcadia Area",
    "Block D — Level 2, Corridor near Lift Lobby",
    "Block E — Level 2, Corridor near Lift Lobby",
    "Block C — Level 2, Roof Garden",
    "Block E — Level 2, Roof Garden",

    # Third Floor
    "Block C — Level 3, Library Corridor",
    "Block C — Level 3, Library Interior",
    "Block D — Level 3, Corridor near Lift Lobby",
    "Block D — Level 3, Hive Room D3.04 Area",
    "Block D — Level 3, Audio Recording Studio Corridor",
    "Block E — Level 3, Corridor near Lift Lobby",

    # Fourth Floor
    "Block C — Level 4, Library Corridor",
    "Block C — Level 4, Library Interior",
    "Block D — Level 4, Corridor near Lift Lobby",
    "Block D — Level 4, Physics Lab D4.06 Corridor",
    "Block D — Level 4, Chemistry Lab D4.07 Corridor",
    "Block D — Level 4, Final Year Project Labs Corridor",
    "Block E — Level 4, Corridor near Lift Lobby",
    "Block E — Level 4, Architecture Studios Corridor",

    # Fifth Floor
    "Block C — Level 5, Library C5.01 Entrance",
    "Block C — Level 5, Library C5.01 Interior",
    "Block D — Level 5, Corridor near Lift Lobby",
    "Block D — Level 5, Pharmaceutical Labs Corridor",
    "Block D — Level 5, Microbiology Lab D5.09 Corridor",
    "Block E — Level 5, Corridor near Lift Lobby",
    "Block E — Level 5, Design Studios Corridor",
    "Block D — Level 5, Roof Terrace",

    # Sixth Floor
    "Block C — Level 6, Seminar Room C6.06 Corridor",
    "Block C — Level 6, Roof Terrace",
    "Block D — Level 6, Chemistry Lab D6.01 Corridor",
    "Block D — Level 6, Biology Lab D6.03 Corridor",
    "Block D — Level 6, Anatomy Lab D6.07 Interior",
    "Block D — Level 6, Research Lab D6.05 Corridor",
    "Block E — Level 6, Corridor near Lift Lobby",

    # Seventh Floor
    "Block C — Level 7, Computer Lab C7.01 Interior",
    "Block C — Level 7, Computer Lab C7.02 Interior",
    "Block C — Level 7, Computer Lab C7.03 Interior",
    "Block C — Level 7, Computer Lab C7.04 Interior",
    "Block C — Level 7, Cyber Security Lab C7.05 Interior",
    "Block C — Level 7, Huawei Lab C7.06 Interior",
    "Block C — Level 7, Computer Lab C7.07 Interior",
    "Block C — Level 7, Computer Labs Corridor",
    "Block D — Level 7, Data Centre D7.07 Corridor",
    "Block D — Level 7, Network Operations Centre Corridor",
    "Block D — Level 7, MAC Labs Corridor",
    "Block E — Level 7, Corridor near Lift Lobby",
    "Block E — Level 7, Roof Terrace",

    # Eighth Floor
    "Block C — Level 8, Big Data Lab C8.03 Interior",
    "Block C — Level 8, Computer Labs Corridor",
    "Block D — Level 8, Corridor near Lift Lobby",
    "Block E — Level 8, Corridor near Lift Lobby",
    "Block E — Level 8, Pharmaceutical Technology Lab Corridor",

    # Security Posts
    "Main Security Gate — Entrance",
    "Syopz Mall — Parking Area",
    "Block A — Parking Area Entrance",
]


# The full Aegis analysis pipeline

# A function to classify the incident type
def classify_incident_type(alert_text: str) -> str:
    """Classifies alert text into a human-readable incident category."""
    text = alert_text.lower()

    if any(w in text for w in ["fight", "altercation", "violence", "shoving", "pushing", "weapon", "knife", "blade", "assault", "grappling", "restraining"]):
        return "Fight / Physical Altercation"
    elif any(w in text for w in ["fallen", "collapsed", "unconscious", "on the ground", "lying"]):
        return "Person Fallen / Unconscious"
    elif any(w in text for w in ["medical", "distress", "convulsing", "unresponsive"]):
        return "Medical Emergency"
    elif any(w in text for w in ["harassment", "bullying", "cornered", "intimidated"]):
        return "Harassment / Bullying"
    elif any(w in text for w in ["loitering", "lingering"]):
        return "Loitering / Suspicious Presence"
    elif any(w in text for w in ["vandalism", "damaging property"]):
        return "Vandalism / Property Damage"
    elif any(w in text for w in ["crowd", "gathering", "panic", "running"]):
        return "Unusual Crowd / Panic"
    elif "clear" in text:
        return "Clear"
    else:
        return "Suspicious Behaviour"

# Using pose estimation to detect possible fallen people


def analyze_pose(image_bytes):
    image = PIL.Image.open(io.BytesIO(image_bytes))
    results = pose_model(image, verbose=False)

    fallen_detected = False
    people_count = 0
    fallen_details = []

    for result in results:
        people_count = len(result.boxes)

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1

            if height == 0:
                continue

            ratio = width / height
            is_fallen = ratio > 1.5

            keypoint_note = ""
            if result.keypoints is not None and is_fallen:
                kpts = result.keypoints.xy[i].tolist()
                scores = result.keypoints.conf[i].tolist() \
                    if result.keypoints.conf is not None else []

                shoulder_y, hip_y, ankle_y = [], [], []
                for idx, (kpt, score) in enumerate(zip(kpts, scores)):
                    if score > 0.5:
                        if idx in [5, 6]:
                            shoulder_y.append(kpt[1])
                        elif idx in [11, 12]:
                            hip_y.append(kpt[1])
                        elif idx in [15, 16]:
                            ankle_y.append(kpt[1])

                if shoulder_y and ankle_y:
                    vertical_spread = abs(
                        np.mean(ankle_y) - np.mean(shoulder_y))
                    if vertical_spread < (height * 0.4):
                        keypoint_note = "keypoints confirm horizontal body"
                    else:
                        is_fallen = False

            if is_fallen:
                fallen_detected = True
                fallen_details.append(
                    f"Person {i+1}: ratio {ratio:.2f} {keypoint_note}"
                )

    lines = [f"Pose analysis — people detected: {people_count}"]
    if fallen_detected:
        lines.append(
            f"FALLEN PERSON DETECTED — {len(fallen_details)} person(s) horizontal"
        )
        lines.extend(fallen_details)
    else:
        lines.append("No fallen persons — all individuals appear upright")

    return "\n".join(lines), fallen_detected

# The function that runs the entire analysis pipeline for an uploaded image


def analyze_image(image_bytes, location):
    time_now = datetime.now().strftime("%I:%M %p")

    # Running YOLO on the uploaded image
    # conf=0.25 — lower threshold catches occluded people and partially visible weapons
    # (knives held close to the body, people grappling) that 0.40 would silently discard
    image = PIL.Image.open(io.BytesIO(image_bytes))
    results = yolo(image, conf=0.25, iou=0.45)
    pose_summary, fallen_detected = analyze_pose(image_bytes)

    # Building the detection summary and drawing bounding boxes
    detections = []
    person_count = 0

    # Converting the image to numpy array so OpenCV can draw on it
    image_array = np.array(image)
    image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Defining classes that are relevant to campus safety monitoring
    RELEVANT_CLASSES = {
        "person",
        # Threat indicators
        "knife", "scissors",
        # Suspicious belongings
        "backpack", "handbag", "suitcase",
        # Outdoor safety risks
        "car", "motorcycle", "bicycle", "bus", "truck"
    }

    for result in results:
        for box in result.boxes:
            label = yolo.names[int(box.cls)]
            conf = float(box.conf)

            # Skip irrelevant objects — no box, no label, no summary entry
            if label not in RELEVANT_CLASSES:
                continue

            # Only consider detections above 25% confidence for the summary and bounding boxes
            # Matches the conf=0.25 model call — no detections are silently discarded
            if conf > 0.25:
                detections.append(f"{label} ({conf:.0%})")
                if label == "person":
                    person_count += 1

                # Getting the bounding box coordinates from YOLO
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Red for people, orange if fallen, yellow for objects
                if label == "person":
                    color = (0, 165, 255) if fallen_detected else (0, 0, 255)
                else:
                    color = (0, 255, 255)

                # Drawing the bounding box and confidence label on the image
                img_height, img_width = image_cv.shape[:2]
                box_thickness = max(2, int(min(img_width, img_height) / 200))
                font_scale = max(0.6, min(img_width, img_height) / 800)
                font_thickness = max(2, int(min(img_width, img_height) / 400))

                cv2.rectangle(image_cv, (x1, y1), (x2, y2),
                              color, box_thickness)

                # Prevent label from going off the top or left edge
                label_y = y1 - 10 if y1 - 10 > 20 else y1 + 25
                label_x = max(5, x1)
                display_label = f"FALLEN {conf:.0%}" if (
                    label == "person" and fallen_detected) else f"{label} {conf:.0%}"
                cv2.putText(image_cv, display_label,
                            (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, color, font_thickness)

    # Converting the annotated image back to PIL format for Streamlit
    annotated_image = PIL.Image.fromarray(
        cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    # Running fight detection AFTER person_count is fully calculated
    fight_summary, is_fight, fight_confidence = analyze_fight(
        image_bytes, person_count)

    # Building the detection summary text
    if detections:
        detection_summary = f"{person_count} person(s) detected. Objects: {', '.join(detections)}"
    else:
        detection_summary = "No people or objects detected."

    pose_display = "Fallen person detected" if fallen_detected else "No fallen person detected"
    fight_display = f"Violence detected ({fight_confidence:.0%})" if is_fight else f"No violence detected ({fight_confidence:.0%})"

    prompt = f"""
    You are Aegis, an AI campus safety assistant at Taylor's University Malaysia.

    Camera location: {location}
    Current time: {time_now}

    Computer vision analysis:
    {detection_summary}

    Pose estimation analysis:
    {pose_summary}

    Fight detection analysis:
    {fight_summary}

    NOTE: YOLO may undercount people in close-contact or occluded scenes. Trust your
    own visual assessment — if you see people, they are there regardless of YOLO count.

    Now look at this image carefully and check for ANY of the following:
    1. FIGHT or VIOLENCE — physical aggression, grappling, assault, pushing, shoving,
       one person restraining another, body contact that looks forceful or non-consensual
    2. WEAPON — any knife, blade, sharp object, or improvised weapon visible in anyone's hand
    3. PERSON FALLEN or COLLAPSED — someone on the ground, injured or unconscious
    4. MEDICAL DISTRESS — someone clutching chest, convulsing, unresponsive
    5. UNUSUAL CROWD — sudden gathering, panic, running, people surrounding someone
    6. HARASSMENT or BULLYING — someone being cornered or intimidated
    7. PERSON IN DISTRESS — someone sitting alone looking distressed or upset
    8. LOITERING — someone lingering suspiciously in one spot
    9. VANDALISM — someone visibly damaging property

    IMPORTANT RULES:
    - Do NOT flag normal walking, standing, carrying a bag, or using stairs as suspicious.
    - Do NOT invent danger if the scene looks normal.
    - Only raise an alert if there is clear visual evidence that security should act.
    - If uncertain, return CLEAR.
    - If you see a weapon in anyone's hand, always flag HIGH regardless of other context.
    - If two or more people appear to be physically struggling, always flag at least MEDIUM.

    If ANY threat is present respond in this exact format:

    SEVERITY: [HIGH / MEDIUM / LOW]

    [What is happening and exactly where]

    [Brief context of what the guard will find]

    ACTION: [One clear instruction]

    If NO threat is present respond with:
    CLEAR — No threats detected at this location.

    Under 60 words. Write like a human security officer.
    Write as if a life depends on reading this fast.
    """

    # Getting the response from Claude — retries once on rate limit errors
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    alert = None
    for attempt in range(2):
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
            break
        except anthropic.RateLimitError:
            if attempt == 0:
                import time as _time
                _time.sleep(5)
            else:
                alert = (
                    f"AI ANALYSIS UNAVAILABLE\n\n"
                    f"YOLO detected: {detection_summary}\n"
                    f"Pose: {pose_display}\n"
                    f"Fight detection: {fight_display}\n\n"
                    f"Claude AI is temporarily unavailable (rate limit). Please review this feed manually."
                )
                break
        except Exception:
            alert = (
                f"AI ANALYSIS UNAVAILABLE\n\n"
                f"YOLO detected: {detection_summary}\n"
                f"Pose: {pose_display}\n"
                f"Fight detection: {fight_display}\n\n"
                f"Claude AI could not complete analysis. Please review this feed manually."
            )
            break
    if alert is None:
        alert = "AI ANALYSIS UNAVAILABLE — Please review this camera feed manually."

    # Extracting the severity level for the incident log
    severity = "CLEAR"
    if "SEVERITY: HIGH" in alert:
        severity = "HIGH"
    elif "SEVERITY: MEDIUM" in alert:
        severity = "MEDIUM"
    elif "SEVERITY: LOW" in alert:
        severity = "LOW"

    return detection_summary, pose_display, fight_display, alert, annotated_image, severity

# Alert Sound Function


def play_alert_sound():
    sound_path = "Alert_Sound/alert.mp3"
    if not os.path.exists(sound_path):
        return
    with open(sound_path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    components.html(f"""
        <audio autoplay style="display:none">
            <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
        </audio>
        <script>
            var audio = new Audio("data:audio/mpeg;base64,{b64}");
            audio.play();
        </script>
    """, height=0)


# Building the Streamlit Dashboard
st.set_page_config(page_title="Aegis — Campus AI Safety", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
/* ══════════════════════ AEGIS THEME ══════════════════════ */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

.stApp { background-color: #0a0e1a; }
.main .block-container { padding-top: 1.5rem; }

[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #1e293b;
}

/* Buttons */
.stButton > button {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #21262d !important;
    border-color: #58a6ff !important;
    color: #58a6ff !important;
    box-shadow: 0 0 12px rgba(88,166,255,0.15) !important;
}
[data-testid="stDownloadButton"] > button {
    background: rgba(13,61,102,0.6) !important;
    border-color: #1a6699 !important;
    color: #7dd3fc !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(14,77,128,0.8) !important;
    border-color: #58a6ff !important;
    color: #93c5fd !important;
}

/* File uploader */
[data-testid="stFileUploader"] section {
    background: #0d1117;
    border: 1px dashed #30363d;
    border-radius: 10px;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #58a6ff;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #161b22;
    border-color: #30363d;
}

hr { border-color: #1e293b !important; }
[data-testid="stAlertContainer"] { border-radius: 10px !important; }
video { border-radius: 10px; }

@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,68,68,0.35); }
    50%       { box-shadow: 0 0 0 8px rgba(255,68,68,0); }
}
</style>
""", unsafe_allow_html=True)

# Sidebar — Campus Camera Selection and Summary
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 8px 0;">
        <div style="font-size:1.4rem; font-weight:800; letter-spacing:0.1em; color:#e2e8f0;">🛡️ AEGIS</div>
        <div style="font-size:0.7rem; color:#64748b; letter-spacing:0.08em; margin-top:2px; text-transform:uppercase;">Campus AI Safety System</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<p style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:4px;">Camera Location</p>', unsafe_allow_html=True)
    selected_camera = st.selectbox(
        "Select Camera Location",
        CAMPUS_CAMERAS,
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown('<p style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">Today\'s Summary</p>', unsafe_allow_html=True)

    total = len(st.session_state.incident_log)
    high = len(
        [i for i in st.session_state.incident_log if i['Severity'] == 'HIGH'])
    medium = len(
        [i for i in st.session_state.incident_log if i['Severity'] == 'MEDIUM'])
    low = len([i for i in st.session_state.incident_log if i['Severity'] == 'LOW'])
    cleared = len(
        [i for i in st.session_state.incident_log if i['Severity'] == 'CLEAR'])

    st.markdown(f"""
    <div style="margin:4px 0 12px 0;">
        <div style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:12px 16px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em;">Total Incidents</span>
            <span style="font-size:1.6rem; font-weight:700; color:#e2e8f0; line-height:1;">{total}</span>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">
            <div style="background:#161b22; border:1px solid rgba(255,68,68,0.35); border-radius:10px; padding:10px; text-align:center;">
                <div style="font-size:0.6rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:3px;">High</div>
                <div style="font-size:1.4rem; font-weight:700; color:#ff4444; line-height:1;">{high}</div>
            </div>
            <div style="background:#161b22; border:1px solid rgba(255,170,0,0.35); border-radius:10px; padding:10px; text-align:center;">
                <div style="font-size:0.6rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:3px;">Medium</div>
                <div style="font-size:1.4rem; font-weight:700; color:#ffaa00; line-height:1;">{medium}</div>
            </div>
            <div style="background:#161b22; border:1px solid rgba(52,211,153,0.35); border-radius:10px; padding:10px; text-align:center;">
                <div style="font-size:0.6rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:3px;">Low</div>
                <div style="font-size:1.4rem; font-weight:700; color:#34d399; line-height:1;">{low}</div>
            </div>
            <div style="background:#161b22; border:1px solid rgba(100,116,139,0.3); border-radius:10px; padding:10px; text-align:center;">
                <div style="font-size:0.6rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:3px;">Clear</div>
                <div style="font-size:1.4rem; font-weight:700; color:#64748b; line-height:1;">{cleared}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Main Dashboard Header
st.markdown(f"""
<div style="display:flex; align-items:center; gap:16px; padding:8px 0 20px 0; border-bottom:1px solid #1e293b; margin-bottom:24px;">
    <div style="font-size:2.4rem; line-height:1;">🛡️</div>
    <div>
        <div style="font-size:2rem; font-weight:800; letter-spacing:0.06em; color:#e2e8f0; line-height:1.1;">AEGIS</div>
        <div style="font-size:0.72rem; color:#64748b; letter-spacing:0.14em; text-transform:uppercase; margin-top:2px;">Taylor's University Lakeside Campus</div>
    </div>
    <div style="margin-left:auto; display:flex; align-items:center; gap:8px; background:rgba(52,211,153,0.08); border:1px solid rgba(52,211,153,0.25); border-radius:100px; padding:6px 14px;">
        <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#34d399;"></span>
        <span style="font-size:0.72rem; font-weight:600; color:#34d399; letter-spacing:0.08em;">SYSTEM LIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Two columns — left for camera feed, right for alert
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
        <span style="font-size:0.65rem; padding:3px 9px; background:rgba(88,166,255,0.12); color:#58a6ff; border:1px solid rgba(88,166,255,0.25); border-radius:100px; letter-spacing:0.06em; text-transform:uppercase; font-weight:600;">CAM</span>
        <span style="font-size:0.88rem; color:#e2e8f0; font-weight:500;">{selected_camera}</span>
    </div>
    """, unsafe_allow_html=True)

    # Image uploader for the camera feed
    uploaded_image = st.file_uploader(
        "Upload camera footage",
        type=["jpg", "jpeg", "png"]
    )

    # Showing annotated image after analysis or original before analysis
    if uploaded_image is not None:
        if st.session_state.annotated_image is not None:
            st.image(st.session_state.annotated_image,
                     caption="Aegis Detection Feed",
                     use_container_width=False,
                     width=500)
        else:
            st.image(uploaded_image,
                     caption="Live Camera Feed",
                     use_container_width=False,
                     width=500)

with col2:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
        <span style="font-size:0.65rem; padding:3px 9px; background:rgba(255,68,68,0.1); color:#ff6b6b; border:1px solid rgba(255,68,68,0.25); border-radius:100px; letter-spacing:0.06em; text-transform:uppercase; font-weight:600;">ALERT</span>
        <span style="font-size:0.88rem; color:#e2e8f0; font-weight:500;">Aegis Analysis</span>
    </div>
    """, unsafe_allow_html=True)

    # Analyze button — triggers the full YOLO and Claude pipeline
    if st.button("⚡ Analyze with Aegis", use_container_width=True):
        if uploaded_image is not None:
            with st.spinner("Aegis is analyzing the camera feed..."):
                image_bytes = uploaded_image.read()
                detection_summary, pose_summary, fight_summary, alert, annotated_image, severity = analyze_image(
                    image_bytes, selected_camera)

                # Saving results to session state so they persist
                st.session_state.annotated_image = annotated_image
                st.session_state.alert = alert
                st.session_state.detection_summary = detection_summary
                st.session_state.pose_summary = pose_summary
                st.session_state.fight_summary = fight_summary

                # Adding the incident to the log
                st.session_state.incident_log.append({
                    "Time": datetime.now().strftime("%I:%M %p"),
                    "Camera": selected_camera,
                    "Type": classify_incident_type(alert),
                    "Severity": severity,
                    "Detection": detection_summary,
                    "Alert Summary": alert[:120] + "..." if len(alert) > 120 else alert,
                    "Full Alert": alert
                })

                save_log(st.session_state.incident_log)

            # Sound plays AFTER spinner, BEFORE rerun
            if severity in ["HIGH", "MEDIUM"]:
                st.session_state.play_sound = True

            st.rerun()
        else:
            st.warning("Please upload a camera image first.")

    # Detection summary cards
    if st.session_state.detection_summary:
        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #1e293b; border-left:3px solid #58a6ff; border-radius:8px; padding:10px 14px; margin:6px 0;">
            <div style="font-size:0.65rem; color:#58a6ff; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:3px;">YOLO Detection</div>
            <div style="font-size:0.85rem; color:#cbd5e1;">{st.session_state.detection_summary}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.fight_summary:
        is_fight = "Violence detected" in st.session_state.fight_summary
        fight_color = "#ff4444" if is_fight else "#58a6ff"
        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #1e293b; border-left:3px solid {fight_color}; border-radius:8px; padding:10px 14px; margin:6px 0;">
            <div style="font-size:0.65rem; color:{fight_color}; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:3px;">Fight Detection</div>
            <div style="font-size:0.85rem; color:#cbd5e1;">{st.session_state.fight_summary}</div>
            <div style="font-size:0.7rem; color:#475569; margin-top:4px;">Validated through multimodal reasoning.</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.pose_summary:
        pose_color = "#ffaa00" if "FALLEN" in st.session_state.pose_summary.upper() else "#58a6ff"
        pose_text = st.session_state.pose_summary.replace("\n", " · ")
        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #1e293b; border-left:3px solid {pose_color}; border-radius:8px; padding:10px 14px; margin:6px 0;">
            <div style="font-size:0.65rem; color:{pose_color}; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:3px;">Pose Analysis</div>
            <div style="font-size:0.85rem; color:#cbd5e1;">{pose_text}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.detection_summary or st.session_state.fight_summary or st.session_state.pose_summary:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Alert display
    if st.session_state.alert:
        alert_text = st.session_state.alert
        if "CLEAR" in alert_text:
            st.markdown(f"""
            <div style="background:rgba(34,197,94,0.07); border:1px solid rgba(34,197,94,0.35); border-radius:12px; padding:16px 20px; margin-top:8px;">
                <div style="font-size:0.65rem; font-weight:700; color:#22c55e; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:6px;">✓ All Clear</div>
                <div style="font-size:0.88rem; color:#86efac;">{alert_text}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            sev = "HIGH" if "SEVERITY: HIGH" in alert_text else "MEDIUM" if "SEVERITY: MEDIUM" in alert_text else "LOW"
            _colors = {"HIGH": ("#ff4444", "rgba(255,68,68,0.08)", "rgba(255,68,68,0.45)"),
                       "MEDIUM": ("#ffaa00", "rgba(255,170,0,0.08)", "rgba(255,170,0,0.45)"),
                       "LOW": ("#34d399", "rgba(52,211,153,0.08)", "rgba(52,211,153,0.45)")}
            sev_color, sev_bg, sev_border = _colors[sev]
            pulse_style = "animation: pulse-red 2s infinite;" if sev == "HIGH" else ""
            formatted = alert_text.replace("\n", "<br>")
            st.markdown(f"""
            <div style="background:{sev_bg}; border:1px solid {sev_border}; border-radius:12px; padding:20px; margin-top:8px; {pulse_style}">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                    <span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:{sev_color}; flex-shrink:0;"></span>
                    <span style="font-size:0.65rem; font-weight:700; letter-spacing:0.12em; color:{sev_color}; text-transform:uppercase;">⚠ Aegis Alert — Severity {sev}</span>
                </div>
                <div style="color:#e2e8f0; line-height:1.75; font-size:0.88rem;">{formatted}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.session_state.get("play_sound"):
                play_alert_sound()
                st.session_state.play_sound = False

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 1 — MULTI-CAMERA GRID ANALYSIS
# Lets the guard upload 2–5 feeds at once and analyze them all in one click.
# Results are laid out in a grid so multiple locations can be reviewed together,
# which matches how a real CCTV control room works.
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="border-top:1px solid #1e293b; padding-top:28px; margin-top:12px; margin-bottom:16px; display:flex; align-items:center; gap:10px;">
    <span style="font-size:0.65rem; padding:3px 9px; background:rgba(148,163,184,0.1); color:#94a3b8; border:1px solid rgba(148,163,184,0.2); border-radius:100px; letter-spacing:0.06em; text-transform:uppercase; font-weight:600;">Grid</span>
    <span style="font-size:1.1rem; font-weight:600; color:#e2e8f0;">Multi-Camera Grid Analysis</span>
</div>
""", unsafe_allow_html=True)

# Accept 2–5 images at once — each image represents a separate camera feed
multi_uploaded = st.file_uploader(
    "Upload 2–5 camera feeds for simultaneous analysis",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="multi_cam_uploader"
)

if multi_uploaded and len(multi_uploaded) > 5:
    st.markdown('<p style="font-size:0.82rem; color:#ff4444; margin:6px 0;">Maximum 5 feeds at once — please remove some files.</p>', unsafe_allow_html=True)
elif multi_uploaded and len(multi_uploaded) == 1:
    st.markdown('<p style="font-size:0.82rem; color:#94a3b8; margin:6px 0;">Upload at least 2 feeds to use grid mode. For a single feed use the section above.</p>', unsafe_allow_html=True)
elif multi_uploaded and 2 <= len(multi_uploaded) <= 5:
    st.markdown(f'<p style="font-size:0.8rem; color:#94a3b8; margin-bottom:10px;">{len(multi_uploaded)} feeds uploaded — assign a location to each one.</p>', unsafe_allow_html=True)

    # Each feed gets its own dropdown so the guard can assign a different location to each camera
    multi_locations = []
    loc_cols = st.columns(len(multi_uploaded))
    for _mi, _mc in enumerate(loc_cols):
        with _mc:
            _loc = st.selectbox(
                f"Feed {_mi + 1}",
                CAMPUS_CAMERAS,
                key=f"multi_cam_loc_{_mi}"
            )
            multi_locations.append(_loc)

    if st.button("⚡ Analyze All Feeds", use_container_width=True, key="multi_analyze_btn"):
        _grid_results = []
        for _mi, _uf in enumerate(multi_uploaded):
            _cam = multi_locations[_mi]
            # Show the current camera name in the spinner so the guard knows what is processing
            with st.spinner(f"Analyzing: {_cam}..."):
                _img_bytes = _uf.read()
                _det, _pose, _fight, _alert, _ann_img, _sev = analyze_image(_img_bytes, _cam)
            _grid_results.append({
                "camera": _cam,
                "alert": _alert,
                "severity": _sev,
                "annotated_image": _ann_img,
                "detection": _det,
            })
            # Add to the main incident log exactly as the single-image section does
            st.session_state.incident_log.append({
                "Time": datetime.now().strftime("%I:%M %p"),
                "Camera": _cam,
                "Type": classify_incident_type(_alert),
                "Severity": _sev,
                "Detection": _det,
                "Alert Summary": _alert[:120] + "..." if len(_alert) > 120 else _alert,
                "Full Alert": _alert
            })
        save_log(st.session_state.incident_log)
        st.session_state.multi_camera_results = _grid_results
        # Trigger the alert sound if any feed came back HIGH or MEDIUM
        if any(r["severity"] in ["HIGH", "MEDIUM"] for r in _grid_results):
            st.session_state.play_sound = True
        st.rerun()

# Show grid results stored from the most recent analysis run
if st.session_state.multi_camera_results:
    _mc_count = len(st.session_state.multi_camera_results)
    # 2 feeds → 2 columns, 3–5 feeds → 3 columns so cards stay readable
    _mc_ncols = 2 if _mc_count == 2 else 3
    _mc_cols = st.columns(_mc_ncols)

    # Severity color lookups — same values used throughout the rest of the dashboard
    _mc_c  = {"HIGH": "#ff4444", "MEDIUM": "#ffaa00", "LOW": "#34d399", "CLEAR": "#22c55e"}
    _mc_bg = {"HIGH": "rgba(255,68,68,0.08)", "MEDIUM": "rgba(255,170,0,0.08)", "LOW": "rgba(52,211,153,0.08)", "CLEAR": "rgba(34,197,94,0.07)"}
    _mc_bd = {"HIGH": "rgba(255,68,68,0.45)", "MEDIUM": "rgba(255,170,0,0.45)", "LOW": "rgba(52,211,153,0.45)", "CLEAR": "rgba(34,197,94,0.35)"}

    for _i, _res in enumerate(st.session_state.multi_camera_results):
        _sev = _res["severity"]
        _c   = _mc_c.get(_sev, "#64748b")
        _bg  = _mc_bg.get(_sev, "rgba(100,116,139,0.08)")
        _bd  = _mc_bd.get(_sev, "rgba(100,116,139,0.3)")
        _pulse = "animation: pulse-red 2s infinite;" if _sev == "HIGH" else ""
        _fmt = _res["alert"].replace("\n", "<br>")

        with _mc_cols[_i % _mc_ncols]:
            # Annotated image first so the guard sees the visual immediately
            st.image(_res["annotated_image"], use_container_width=True)
            # Severity card beneath each image — camera label, badge, full alert text
            st.markdown(f"""
            <div style="background:{_bg}; border:1px solid {_bd}; border-radius:10px; padding:14px; margin-top:6px; {_pulse}">
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; flex-wrap:wrap; gap:4px;">
                    <span style="font-size:0.75rem; color:#e2e8f0; font-weight:500;">{_res['camera']}</span>
                    <span style="background:{_bg}; color:{_c}; border:1px solid {_bd}; font-size:0.6rem; font-weight:700; padding:2px 8px; border-radius:100px; letter-spacing:0.1em; text-transform:uppercase;">{_sev}</span>
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.65;">{_fmt}</div>
            </div>
            """, unsafe_allow_html=True)

    # Play sound once after all cards are rendered (flag was set during the analysis loop)
    if st.session_state.get("play_sound"):
        play_alert_sound()
        st.session_state.play_sound = False

# Video Analysis
st.markdown("""
<div style="border-top:1px solid #1e293b; padding-top:28px; margin-top:12px; margin-bottom:16px; display:flex; align-items:center; gap:10px;">
    <span style="font-size:0.65rem; padding:3px 9px; background:rgba(88,166,255,0.1); color:#58a6ff; border:1px solid rgba(88,166,255,0.25); border-radius:100px; letter-spacing:0.06em; text-transform:uppercase; font-weight:600;">Video</span>
    <span style="font-size:1.1rem; font-weight:600; color:#e2e8f0;">Video Analysis</span>
</div>
""", unsafe_allow_html=True)

uploaded_video = st.file_uploader(
    "Upload video clip from camera",
    type=["mp4", "avi", "mov", "mkv"],
    key="video_uploader"
)

if uploaded_video is not None:
    st.video(uploaded_video)

    if st.button("Analyze Video with Aegis", use_container_width=True):
        tmp_video_path = None
        try:
            # Write video to a temp file so OpenCV can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                tmp_video_path = tmp_file.name

            with st.spinner("Aegis is analyzing the video feed..."):
                # Get video duration to adjust sampling and cooldown
                _cap = cv2.VideoCapture(tmp_video_path)
                _fps = _cap.get(cv2.CAP_PROP_FPS) or 25
                _total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                _cap.release()
                _duration = _total / _fps

                # For very short videos sample every second with tight cooldown.
                # For longer videos open the cooldown to avoid excessive API calls.
                _sample = 1.0
                if _duration <= 10:
                    _cooldown = 1.0
                elif _duration <= 60:
                    _cooldown = 2.0
                else:
                    _cooldown = 4.0

                # Run the full video analysis pipeline
                incidents, total_frames, fps = analyze_video(
                    tmp_video_path,
                    selected_camera,
                    sample_every_sec=_sample,
                    merge_window_sec=3,
                    ai_cooldown_sec=_cooldown
                )

        except Exception as e:
            st.markdown(f"""
            <div style="background:rgba(255,68,68,0.08); border:1px solid rgba(255,68,68,0.4); border-radius:10px; padding:14px 18px; margin:8px 0; font-size:0.88rem; color:#fca5a5;">
                <strong>Video analysis failed:</strong> {str(e)}
            </div>
            """, unsafe_allow_html=True)
            incidents = []

        finally:
            # Always clean up the temp file even if analysis crashed
            if tmp_video_path and os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)

        if incidents:
            # Pick the highest severity incident to show at the top
            top_incident = max(incidents, key=lambda x: [
                               "CLEAR", "LOW", "MEDIUM", "HIGH"].index(x["Severity"]))

            st.session_state.video_result_frame = top_incident["frame_path"]
            st.session_state.video_alert_text = top_incident["alert"]
            st.session_state.video_alert_severity = top_incident["Severity"]
            st.session_state.video_alert_timestamp = top_incident["start_timestamp"]
            st.session_state.video_alert_camera = selected_camera
            st.session_state.video_alert_type = top_incident.get(
                "incident_type", "Suspicious Behaviour")
            # Save all frame paths for the timeline display
            st.session_state.video_all_frames = top_incident.get(
                "all_frame_paths", [top_incident["frame_path"]])
            st.session_state.video_fight_summary = top_incident.get(
                "fight_summary", "")

            # Push all incidents into the main incident log
            for incident in incidents:
                st.session_state.incident_log.append({
                    "Time": datetime.now().strftime("%I:%M %p"),
                    "Camera": selected_camera,
                    "Type": incident.get("incident_type", "Unknown"),
                    "Severity": incident["Severity"],
                    "Detection": incident["Detection"],
                    "Alert Summary": incident["Alert Summary"],
                    "Full Alert": incident["alert"]
                })
            save_log(st.session_state.incident_log)

            if top_incident["Severity"] in ["HIGH", "MEDIUM"]:
                st.session_state.play_sound = True

            # Save every incident found in the video so the timeline view can show all of them
            st.session_state.video_all_incidents = incidents

            st.rerun()
        else:
            st.markdown("""
            <div style="background:rgba(34,197,94,0.07); border:1px solid rgba(34,197,94,0.35); border-radius:10px; padding:14px 18px; margin:8px 0; font-size:0.88rem; color:#86efac;">
                ✓ No threats detected in this video clip.
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 2 — VIDEO INCIDENT TIMELINE
# Replaces the old single-incident display. Shows all incidents found in the
# video in chronological order so the guard reads the event as a story from
# first detection to the end of the clip.
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.video_all_incidents:
    _tl_incidents = st.session_state.video_all_incidents
    _tl_camera = st.session_state.video_alert_camera or selected_camera

    # Severity rank used to find the most critical incident to emphasize
    _tl_rank  = {"CLEAR": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    _tl_c     = {"HIGH": "#ff4444", "MEDIUM": "#ffaa00", "LOW": "#34d399", "CLEAR": "#22c55e"}
    _tl_bg    = {"HIGH": "rgba(255,68,68,0.08)", "MEDIUM": "rgba(255,170,0,0.08)", "LOW": "rgba(52,211,153,0.08)", "CLEAR": "rgba(34,197,94,0.07)"}
    _tl_bd    = {"HIGH": "rgba(255,68,68,0.45)", "MEDIUM": "rgba(255,170,0,0.45)", "LOW": "rgba(52,211,153,0.45)", "CLEAR": "rgba(34,197,94,0.35)"}

    _tl_top = max(_tl_incidents, key=lambda x: _tl_rank.get(x["Severity"], 0))

    if len(_tl_incidents) > 1:
        # Only render the "Timeline" header when there are multiple incidents to navigate.
        # A single incident doesn't need a timeline — just show the card.
        _tl_count = len(_tl_incidents)
        _tl_plural = "s" if _tl_count != 1 else ""
        st.markdown(f"""
        <div style="margin:20px 0 16px 0; display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
            <span style="font-size:0.65rem; padding:3px 9px; background:rgba(88,166,255,0.1); color:#58a6ff; border:1px solid rgba(88,166,255,0.25); border-radius:100px; letter-spacing:0.06em; text-transform:uppercase; font-weight:600;">Timeline</span>
            <span style="font-size:1rem; font-weight:600; color:#e2e8f0;">Incident Timeline — {_tl_camera}</span>
            <span style="font-size:0.75rem; color:#64748b;">{_tl_count} event{_tl_plural} detected</span>
        </div>
        """, unsafe_allow_html=True)

    # Sort by start timestamp so the guard reads events in the order they happened
    _tl_sorted = sorted(_tl_incidents, key=lambda x: x.get("start_timestamp", 0))

    for _inc in _tl_sorted:
        _isev = _inc["Severity"]
        _ic   = _tl_c.get(_isev, "#64748b")
        _ibg  = _tl_bg.get(_isev, "rgba(100,116,139,0.08)")
        _ibd  = _tl_bd.get(_isev, "rgba(100,116,139,0.3)")
        _ipulse = "animation: pulse-red 2s infinite;" if _isev == "HIGH" else ""

        _is_top = (_inc is _tl_top) and (len(_tl_incidents) > 1)
        if _is_top:
            # Brightest border and a "Most Critical" label to help the guard triage fast
            _card_border = f"border:2px solid {_ic};"
            _crit_badge = f'<span style="font-size:0.6rem; color:{_ic}; font-weight:700; padding:2px 8px; border-radius:100px; background:{_ibg}; border:1px solid {_ic}; margin-left:6px;">Most Critical</span>'
        else:
            _card_border = f"border:1px solid {_ibd};"
            _crit_badge = ""

        _start = _inc.get("start_timestamp", 0)
        _end   = _inc.get("end_timestamp", _start)
        # Show a range like "4.0s — 12.5s" if the incident spanned more than half a second
        _ts_label = f"{_start:.1f}s — {_end:.1f}s" if abs(_end - _start) >= 0.5 else f"{_start:.1f}s"

        _itype = _inc.get("incident_type", "Suspicious Behaviour")
        _ifmt  = _inc["alert"].replace("\n", "<br>")
        _ifight = _inc.get("fight_summary", "")
        _iframes = _inc.get("all_frame_paths") or ([_inc["frame_path"]] if _inc.get("frame_path") else [])

        # Build the entire header card as one string so there is no split across
        # multiple render calls and no leading whitespace that could trick the
        # Markdown parser into treating the HTML as a code block.
        _header_html = (
            f'<div style="background:{_ibg}; {_card_border} border-radius:12px; padding:18px 20px; margin:12px 0; {_ipulse}">'
            f'<div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;">'
            f'<div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">'
            f'<span style="font-size:0.72rem; color:#94a3b8; font-family:monospace;">⏱ {_ts_label}</span>'
            f'<span style="font-size:0.82rem; color:#e2e8f0; font-weight:500;">{_itype}</span>'
            f'{_crit_badge}'
            f'</div>'
            f'<span style="background:{_ibg}; color:{_ic}; border:1px solid {_ic}; font-size:0.62rem; font-weight:700; padding:3px 10px; border-radius:100px; letter-spacing:0.1em; text-transform:uppercase;">{_isev}</span>'
            f'</div>'
            f'</div>'
        )
        st.markdown(_header_html, unsafe_allow_html=True)

        # Annotated frames — columns match how many frames were saved for this incident
        if len(_iframes) >= 3:
            _fi1, _fi2, _fi3 = st.columns(3)
            _fi1.image(_iframes[0], caption="Initial detection",   use_container_width=True)
            _fi2.image(_iframes[1], caption="Incident developing", use_container_width=True)
            _fi3.image(_iframes[2], caption="Peak moment",         use_container_width=True)
        elif len(_iframes) == 2:
            _fi1, _fi2 = st.columns(2)
            _fi1.image(_iframes[0], caption=f"Detected at {_start:.1f}s", use_container_width=True)
            _fi2.image(_iframes[1], caption="Escalation frame",           use_container_width=True)
        elif len(_iframes) == 1:
            st.image(_iframes[0], caption=f"Detected at {_start:.1f}s", use_container_width=False, width=500)

        if _ifight:
            _ifc = "#ff4444" if "Violence detected" in _ifight else "#58a6ff"
            st.markdown(f"""
            <div style="background:#161b22; border:1px solid #1e293b; border-left:3px solid {_ifc}; border-radius:8px; padding:10px 14px; margin:8px 0;">
                <div style="font-size:0.65rem; color:{_ifc}; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:3px;">Fight Detection</div>
                <div style="font-size:0.85rem; color:#cbd5e1;">{_ifight}</div>
            </div>
            """, unsafe_allow_html=True)

        # Full alert text with newlines converted to HTML breaks
        st.markdown(f"""
        <div style="background:#0d1117; border:1px solid #1e293b; border-radius:10px; padding:16px 20px; margin-top:8px; color:#e2e8f0; font-size:0.88rem; line-height:1.75;">{_ifmt}</div>
        """, unsafe_allow_html=True)

    if st.session_state.get("play_sound"):
        play_alert_sound()
        st.session_state.play_sound = False

# Incident Log Section at the bottom of the dashboard
st.markdown("""
<div style="border-top:1px solid #1e293b; padding-top:28px; margin-top:12px; margin-bottom:16px; display:flex; align-items:center; gap:10px;">
    <span style="font-size:0.65rem; padding:3px 9px; background:rgba(148,163,184,0.1); color:#94a3b8; border:1px solid rgba(148,163,184,0.2); border-radius:100px; letter-spacing:0.06em; text-transform:uppercase; font-weight:600;">Log</span>
    <span style="font-size:1.1rem; font-weight:600; color:#e2e8f0;">Incident Log</span>
</div>
""", unsafe_allow_html=True)

if st.session_state.incident_log:

    # Converting the log to a dataframe for display
    df = pd.DataFrame(st.session_state.incident_log)

    # Coloring rows based on severity
    def color_severity(val):
        if val == "HIGH":
            return "background-color: #ffcccc; color: #cc0000; font-weight: bold"
        elif val == "MEDIUM":
            return "background-color: #fff3cc; color: #cc8800; font-weight: bold"
        elif val == "LOW":
            return "background-color: #ccffcc; color: #006600; font-weight: bold"
        elif val == "CLEAR":
            return "background-color: #e8f5e9; color: #2e7d32"
        return ""

    # Applying the color styling to the severity column
    styled_df = df.style.map(color_severity, subset=["Severity"])
    st.dataframe(styled_df, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # FEATURE 3 — INCIDENT LOG DETAIL VIEW
    # The dataframe shows truncated alert summaries. This panel lets the guard
    # pick any logged incident and read the full Claude analysis in a styled card.
    # ─────────────────────────────────────────────────────────────────────────────

    st.markdown("""
    <div style="margin:20px 0 8px 0; display:flex; align-items:center; gap:8px;">
        <span style="font-size:0.65rem; padding:3px 9px; background:rgba(148,163,184,0.1); color:#94a3b8; border:1px solid rgba(148,163,184,0.2); border-radius:100px; letter-spacing:0.06em; text-transform:uppercase; font-weight:600;">Detail</span>
        <span style="font-size:0.9rem; font-weight:600; color:#e2e8f0;">Full Incident Detail</span>
    </div>
    """, unsafe_allow_html=True)

    # Build labels from index + time + camera so the guard can identify incidents quickly
    _dl_labels = [
        f"#{i + 1} — {inc.get('Time', 'N/A')} — {inc.get('Camera', 'Unknown')[:45]}"
        for i, inc in enumerate(st.session_state.incident_log)
    ]

    _dl_idx = st.selectbox(
        "Select incident",
        range(len(_dl_labels)),
        format_func=lambda i: _dl_labels[i],
        key="incident_detail_selector",
        label_visibility="collapsed"
    )

    if _dl_idx is not None:
        _dl_inc = st.session_state.incident_log[_dl_idx]
        # Use Full Alert when available (both single-image and video save it).
        # Fall back to Alert Summary for any older log entries written before Full Alert was added.
        _dl_full = _dl_inc.get("Full Alert", _dl_inc.get("Alert Summary", "No detail available."))
        _dl_sev  = _dl_inc.get("Severity", "CLEAR")

        _dl_c  = {"HIGH": "#ff4444", "MEDIUM": "#ffaa00", "LOW": "#34d399", "CLEAR": "#22c55e"}.get(_dl_sev, "#64748b")
        _dl_bg = {"HIGH": "rgba(255,68,68,0.08)", "MEDIUM": "rgba(255,170,0,0.08)", "LOW": "rgba(52,211,153,0.08)", "CLEAR": "rgba(34,197,94,0.07)"}.get(_dl_sev, "rgba(100,116,139,0.08)")
        _dl_bd = {"HIGH": "rgba(255,68,68,0.45)", "MEDIUM": "rgba(255,170,0,0.45)", "LOW": "rgba(52,211,153,0.45)", "CLEAR": "rgba(34,197,94,0.35)"}.get(_dl_sev, "rgba(100,116,139,0.3)")
        _dl_pulse = "animation: pulse-red 2s infinite;" if _dl_sev == "HIGH" else ""
        _dl_fmt = _dl_full.replace("\n", "<br>")

        st.markdown(f"""
        <div style="background:{_dl_bg}; border:1px solid {_dl_bd}; border-radius:12px; padding:20px; margin-top:8px; {_dl_pulse}">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px; margin-bottom:12px;">
                <div>
                    <div style="font-size:0.88rem; color:#e2e8f0; font-weight:500;">📍 {_dl_inc.get('Camera', '')}</div>
                    <div style="font-size:0.72rem; color:#94a3b8; margin-top:3px;">{_dl_inc.get('Time', '')} · {_dl_inc.get('Type', '')}</div>
                </div>
                <span style="background:{_dl_bg}; color:{_dl_c}; border:1px solid {_dl_c}; font-size:0.65rem; font-weight:700; padding:3px 10px; border-radius:100px; letter-spacing:0.1em; text-transform:uppercase;">{_dl_sev}</span>
            </div>
            <div style="color:#e2e8f0; line-height:1.75; font-size:0.88rem;">{_dl_fmt}</div>
        </div>
        """, unsafe_allow_html=True)

    # Generating and offering the incident report for download
    report_buffer = generate_incident_report(st.session_state.incident_log)
    st.download_button(
        label="Download Incident Report",
        data=report_buffer,
        file_name=f"aegis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    # Button to clear the incident log
    if st.button("Clear Incident Log"):
        st.session_state.incident_log = []
        st.session_state.annotated_image = None
        st.session_state.alert = None
        st.session_state.detection_summary = None
        st.session_state.video_result_frame = None
        st.session_state.video_alert_text = None
        st.session_state.video_all_frames = []
        st.rerun()

else:
    st.markdown("""
    <div style="background:#0d1117; border:1px dashed #1e293b; border-radius:12px; padding:40px 20px; text-align:center; margin:16px 0;">
        <div style="font-size:2rem; margin-bottom:8px;">📋</div>
        <div style="font-size:0.9rem; font-weight:500; color:#64748b; margin-bottom:4px;">No incidents logged yet</div>
        <div style="font-size:0.8rem; color:#475569;">Select a camera, upload footage, and click Analyze to begin monitoring.</div>
    </div>
    """, unsafe_allow_html=True)
