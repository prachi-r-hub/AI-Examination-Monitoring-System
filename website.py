import os
import cv2
import time
import pandas as pd
import numpy as np
import streamlit as st

import config
from detector import ExamDetector
from tracker import IncidentTracker
from person_tracker import PersonTracker
from head_detector import HeadMovementDetector
from passing_detector import ObjectPassingDetector
from risk_engine import RiskEngine
from voice_alert import VoiceAlertManager
from evidence_manager import EvidenceManager
from database import IncidentDatabase
from video_processor import ExamVideoProcessor

# Streamlit Page Config
st.set_page_config(
    page_title="AI Examination Monitoring System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Control Room CSS Styling (Clean, Technical, Information-Focused)
st.markdown("""
<style>
/* Compact Container Spacing */
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2.0rem !important;
    padding-left: 2.0rem !important;
    padding-right: 2.0rem !important;
    max-width: 100% !important;
}

/* Base Typography & Clean Control Room Font */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* Control Room Header Hierarchy */
h1 {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0.2rem !important;
}
h2 {
    font-size: 1.30rem !important;
    font-weight: 600 !important;
    margin-top: 0.8rem !important;
    margin-bottom: 0.4rem !important;
}
h3 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    margin-top: 0.6rem !important;
    margin-bottom: 0.2rem !important;
}

/* Compact Control Room Metric Cards */
[data-testid="stMetric"] {
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 6px 12px !important;
    box-shadow: none !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

[data-testid="stMetricValue"] {
    font-size: 1.20rem !important;
    font-weight: 700 !important;
}

/* Dividers */
hr {
    margin-top: 0.8rem !important;
    margin-bottom: 0.8rem !important;
}

/* Table Clean Borders */
[data-testid="stDataFrame"] {
    border: 1px solid #cbd5e1 !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize Core Services (cached & session state)
@st.cache_resource
def get_detector():
    return ExamDetector()

detector = get_detector()

if "db" not in st.session_state:
    st.session_state.db = IncidentDatabase()

if "tracker" not in st.session_state:
    st.session_state.tracker = IncidentTracker(confirmation_frames=config.DEFAULT_REQUIRED_CONSECUTIVE_FRAMES)

if "person_tracker" not in st.session_state:
    st.session_state.person_tracker = PersonTracker(enabled=config.ENABLE_PERSON_TRACKING)

if "voice_manager" not in st.session_state:
    st.session_state.voice_manager = VoiceAlertManager(enabled=config.ENABLE_VOICE_ALERTS)

if "evidence_manager" not in st.session_state:
    st.session_state.evidence_manager = EvidenceManager()

if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False

# ==========================================
# PRIMARY NAVIGATION (DEFAULT SCREEN = MONITORING)
# ==========================================
st.sidebar.title("🛡️ Exam Monitor")
st.sidebar.caption("AI-Assisted Examination Monitoring System")

nav_choice = st.sidebar.radio(
    "Navigation:",
    [
        "🎥 Monitoring",
        "📷 Image Detection",
        "🎬 Video Analysis",
        "📋 Incidents",
        "🖼️ Evidence",
        "⚙️ Settings"
    ],
    index=0  # Default screen = Monitoring
)

st.sidebar.divider()

# Quick Status Summary in Sidebar
st.sidebar.subheader("📊 System Status")
st.sidebar.markdown(f"- **Detector**: YOLOv8s (`best.pt`)")
st.sidebar.markdown(f"- **Person Tracking**: {'🟢 Enabled' if st.session_state.person_tracker.enabled else '🔴 Disabled'}")
st.sidebar.markdown(f"- **Voice Alerts**: {'🟢 Enabled' if st.session_state.voice_manager.enabled else '🔴 Disabled'}")
st.sidebar.markdown(f"- **DB Records**: **{len(st.session_state.db.get_all_incidents())}** incidents")


# ==========================================
# SCREEN 1: MONITORING (DEFAULT SCREEN)
# ==========================================
if nav_choice == "🎥 Monitoring":
    st.title("🎥 Live Examination Monitoring")
    st.write(
        "Real-time webcam examination monitoring with YOLOv8s object detection, "
        "person tracking, consecutive-frame confirmation, transparent risk engine, non-blocking voice alerts, automatic evidence capture, and SQLite database logging."
    )

    # Camera Input Source Selection
    cam_mode = st.radio(
        "Select Camera Input Source:",
        ["📸 Interactive Browser Camera (HTML5 Web Device)", "📹 Server OpenCV Camera Stream (Hardware Index)"],
        horizontal=True
    )

    # Monitoring Status Banner
    status_placeholder = st.empty()
    st.divider()

    # Metric Dashboard Bar (7 Required Display Values)
    m_fps, m_people, m_allowed, m_restricted, m_incidents, m_risk = st.columns(6)
    m_fps.metric("⚡ FPS", "0.0")
    m_people.metric("👤 People Detected", "0")
    m_allowed.metric("🟢 Allowed Objects", "0")
    m_restricted.metric("🔴 Restricted Objects", "0")
    m_incidents.metric("🚨 Confirmed Incidents", "0")
    m_risk.metric("📈 Monitoring Risk", "0/100 [LOW]")

    st.divider()
    col_video, col_side = st.columns([3, 2])

    with col_video:
        st.subheader("📡 Live Annotated Camera Feed")
        frame_placeholder = st.empty()

    with col_side:
        st.subheader("📋 Active Frame Detections")
        table_placeholder = st.empty()
        
        st.subheader("⏱️ Recent Incident Timeline")
        timeline_placeholder = st.empty()

    # Helper function to run full backend pipeline on any incoming live webcam frame
    def process_live_frame(frame: np.ndarray, frame_number: int, timestamp: float, t_prev: float):
        fps = 1.0 / max(timestamp - t_prev, 1e-5)

        # 1. Detect people if enabled
        people_dicts = st.session_state.person_tracker.detect_and_track_people(
            frame, conf=0.35, frame_number=frame_number, timestamp=round(timestamp, 2)
        )

        # 2. Detect objects using custom YOLOv8s best.pt
        raw_detections, _ = detector.detect_frame(
            frame, conf=config.DEFAULT_CONF_THRESHOLD, frame_number=frame_number, timestamp=round(timestamp, 2)
        )

        # 3. Associate restricted objects with nearby people
        raw_detections = st.session_state.person_tracker.associate_restricted_objects(people_dicts, raw_detections)

        # 4. Process through monitoring pipeline (consecutive frame confirmation + duration + risk scoring)
        processed_dets, summary = st.session_state.tracker.process_frame_detections(
            raw_detections, frame_number=frame_number, timestamp=round(timestamp, 2), frame_image=frame, mode="live_incidents"
        )

        # 5. Handle newly confirmed incidents
        new_confirmed = summary.get("new_confirmed_incidents", [])
        for inc in new_confirmed:
            # 🔊 Voice alert
            st.session_state.voice_manager.trigger_alert_for_incident(inc, current_time=timestamp)
            # 📸 Evidence screenshot capture
            st.session_state.evidence_manager.capture_evidence(inc, current_frame=frame, mode="live_incidents")
            # 💾 Persist to SQLite DB
            st.session_state.db.insert_incident(inc)

        # 6. Annotate frame
        annotated_frame = detector.annotate_frame(frame, processed_dets, people_dicts=people_dicts)

        # Calculate current frame peak risk score
        max_risk_score = 0
        max_risk_level = "LOW"
        for det in processed_dets:
            if det["category"] == "RESTRICTED":
                r_score = 30 + (15 if det["confidence"] >= 0.85 else 0)
                if r_score > max_risk_score:
                    max_risk_score = r_score
                    max_risk_level = "MEDIUM" if r_score >= 30 else "LOW"

        if len(st.session_state.tracker.confirmed_incidents) > 0:
            last_inc = st.session_state.tracker.confirmed_incidents[-1]
            max_risk_score = max(max_risk_score, last_inc.get("risk_score", 0))
            max_risk_level = last_inc.get("risk_level", "LOW")

        # Update Status Banner
        if new_confirmed:
            status_placeholder.error(f"⚠️ **CONFIRMED INCIDENT ALERT!** {new_confirmed[-1].get('alert_message', 'Restricted object confirmed!')}")
        elif summary["restricted_count"] > 0:
            status_placeholder.warning(f"🟡 **RESTRICTED OBJECT IN VIEW (Candidate Detection)** — Tracking frame persistence ({config.DEFAULT_REQUIRED_CONSECUTIVE_FRAMES} frames required for confirmation)...")
        else:
            status_placeholder.success("🟢 **MONITORING STATUS: NORMAL** — Processing live examination feed...")

        # Update Dashboard Gauges (Real Backend Data Only)
        gpu_infer_ms = getattr(detector, 'last_inference_ms', 0.0)
        m_fps.metric("⚡ FPS", f"{fps:.1f}", delta=f"{gpu_infer_ms:.1f}ms (GTX 1050)", delta_color="off")
        m_people.metric("👤 People Detected", len(people_dicts))
        m_allowed.metric("🟢 Allowed Objects", summary["allowed_count"])
        m_restricted.metric("🔴 Restricted Objects", summary["restricted_count"])
        m_incidents.metric("🚨 Confirmed Incidents", len(st.session_state.tracker.confirmed_incidents))
        m_risk.metric("📈 Monitoring Risk", f"{max_risk_score}/100 [{max_risk_level}]")

        # Render Live Camera Stream
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        # Render Active Frame Detections Table
        if processed_dets:
            table_data = []
            for idx, det in enumerate(processed_dets, 1):
                table_data.append({
                    "#": idx,
                    "Object Name": det["object_name"],
                    "Category": "🟢 ALLOWED" if det["category"] == "ALLOWED" else "🔴 RESTRICTED",
                    "Confidence": f"{det['confidence'] * 100:.1f}%",
                    "Status": det["status_label"],
                    "Person ID": f"Person #{det['person_id']}" if det.get("person_id") and isinstance(det['person_id'], int) else str(det.get("person_id", "N/A"))
                })
            table_placeholder.dataframe(table_data, use_container_width=True, hide_index=True)
        else:
            table_placeholder.caption("No objects detected in current frame.")

        # Render Recent Incident Timeline Table
        all_db_incidents = st.session_state.db.get_recent_incidents(limit=5)
        if all_db_incidents:
            timeline_rows = []
            for inc_r in all_db_incidents:
                timeline_rows.append({
                    "Time": inc_r["time_formatted"],
                    "Event Type": inc_r["event_type"],
                    "Object": inc_r["object_name"],
                    "Person": inc_r["person_id"],
                    "Risk Score": f"{inc_r['risk_score']}/100 [{inc_r['risk_level']}]",
                    "Status": inc_r["review_status"]
                })
            timeline_placeholder.dataframe(timeline_rows, use_container_width=True, hide_index=True)
        else:
            timeline_placeholder.caption("No confirmed incidents recorded yet.")

    # -------------------------------------------------------------
    # CAMERA MODE 1: INTERACTIVE BROWSER CAMERA (HTML5)
    # -------------------------------------------------------------
    if "📸 Interactive Browser Camera" in cam_mode:
        st.write("Capture live webcam frames directly from your browser to run real-time YOLOv8s detection, temporal tracking, and risk analysis.")
        cam_buffer = st.camera_input("Live Webcam Stream (Browser HTML5)")

        if cam_buffer is not None:
            t_now = time.time()
            t_prev = st.session_state.get("last_cam_time", t_now - 0.05)
            st.session_state.last_cam_time = t_now
            
            frame_count = st.session_state.get("cam_frame_count", 0) + 1
            st.session_state.cam_frame_count = frame_count

            # Decode JPEG camera bytes to BGR numpy array
            file_bytes = np.frombuffer(cam_buffer.getvalue(), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is not None:
                process_live_frame(frame, frame_count, t_now, t_prev)
            else:
                status_placeholder.error("⚠️ Failed to decode camera image from browser buffer.")
        else:
            status_placeholder.info("📷 Please allow browser camera access and click 'Take Photo' / capture live frame to process.")
            recent_db_incidents = st.session_state.db.get_recent_incidents(limit=5)
            if recent_db_incidents:
                t_rows = [{"Time": r["time_formatted"], "Event Type": r["event_type"], "Object": r["object_name"], "Person": r["person_id"], "Risk Score": f"{r['risk_score']}/100 [{r['risk_level']}]", "Status": r["review_status"]} for r in recent_db_incidents]
                timeline_placeholder.dataframe(t_rows, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # CAMERA MODE 2: SERVER OPENCV CAMERA STREAM (HARDWARE INDEX)
    # -------------------------------------------------------------
    else:
        c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns([1.5, 1.5, 1.5, 2])
        with c_ctrl1:
            if st.button("▶️ Start OpenCV Stream", use_container_width=True, type="primary"):
                st.session_state.webcam_active = True
        with c_ctrl2:
            if st.button("⏹️ Stop OpenCV Stream", use_container_width=True):
                st.session_state.webcam_active = False
        with c_ctrl3:
            if st.button("🔄 Reconnect Stream", use_container_width=True):
                st.session_state.webcam_active = True
        with c_ctrl4:
            cam_index = st.number_input("Hardware Camera Index", min_value=0, max_value=5, value=0, step=1)

        if not st.session_state.webcam_active:
            status_placeholder.info("⏸️ OpenCV Server Camera Stream is paused. Click '▶️ Start OpenCV Stream' or switch to '📸 Interactive Browser Camera'.")
        else:
            cap = cv2.VideoCapture(cam_index)
            if not cap.isOpened():
                status_placeholder.error(f"❌ Unable to access hardware camera at index {cam_index}. Switch to '📸 Interactive Browser Camera' mode above.")
                st.session_state.webcam_active = False
            else:
                frame_count = 0
                t_prev = time.time()
                try:
                    while cap.isOpened() and st.session_state.webcam_active:
                        ret, frame = cap.read()
                        if not ret:
                            status_placeholder.warning("⚠️ Hardware camera frame unreadable. Switch to '📸 Interactive Browser Camera' mode above for reliable web camera access.")
                            st.session_state.webcam_active = False
                            break

                        frame_count += 1
                        t_now = time.time()
                        process_live_frame(frame, frame_count, t_now, t_prev)
                        t_prev = t_now
                finally:
                    cap.release()


# ==========================================
# SCREEN 2: IMAGE DETECTION
# ==========================================
elif nav_choice == "📷 Image Detection":
    st.title("📷 Image Detection & Class Demonstration")
    st.write("Upload a single image or choose a test dataset sample to demonstrate the 7 custom examination object classes.")

    sample_images_dir = os.path.join(config.BASE_DIR, "Classroom_Dataset", "Merged_Dataset", "test", "images")
    sample_files = []
    if os.path.exists(sample_images_dir):
        sample_files = [f for f in os.listdir(sample_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:15]

    st.subheader("1. Input Image Selection")
    col_input1, col_input2 = st.columns(2)

    with col_input1:
        uploaded_file = st.file_uploader("Upload custom image...", type=["jpg", "png", "jpeg"])

    with col_input2:
        selected_sample = None
        if sample_files:
            selected_sample = st.selectbox("Or choose sample test image:", ["-- Select Sample --"] + sample_files)

    image_to_process = None
    image_name = ""

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, 1)
        image_name = uploaded_file.name
    elif selected_sample and selected_sample != "-- Select Sample --":
        sample_path = os.path.join(sample_images_dir, selected_sample)
        image_to_process = cv2.imread(sample_path)
        image_name = selected_sample

    if image_to_process is not None:
        st.divider()
        st.subheader(f"2. Detection Results for `{image_name}`")

        conf_val = st.slider("Detection Confidence Threshold", 0.10, 0.90, config.DEFAULT_CONF_THRESHOLD, 0.05)

        raw_detections, annotated_frame = detector.detect_frame(image_to_process, conf=conf_val)

        c_img1, c_img2 = st.columns(2)
        with c_img1:
            st.markdown("**Original Image**")
            st.image(cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB), use_container_width=True)
        with c_img2:
            st.markdown("**Annotated Detection Output**")
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.divider()
        st.subheader("3. Class Breakdown Payload")

        if raw_detections:
            img_table = []
            for idx, det in enumerate(raw_detections, 1):
                img_table.append({
                    "#": idx,
                    "Object Name": det["object_name"],
                    "Category": det["category"],
                    "Confidence": f"{det['confidence']*100:.1f}%",
                    "Bounding Box": str(det["bounding_box"]),
                    "Center Point": f"({det['center_x']}, {det['center_y']})"
                })
            st.dataframe(img_table, use_container_width=True, hide_index=True)
        else:
            st.warning("No target objects detected at current confidence threshold.")


# ==========================================
# SCREEN 3: VIDEO ANALYSIS
# ==========================================
elif nav_choice == "🎬 Video Analysis":
    st.title("🎬 Pre-recorded Examination Video Analysis")
    st.write(
        "Upload a prepared examination video for frame-by-frame YOLOv8s object detection, "
        "person tracking, and rule-based observable behavior analysis. "
        "**PRIMARY DEMONSTRATION MODE FOR PROTOTYPE PRESENTATION.**"
    )

    uploaded_video = st.file_uploader("Upload examination video file (MP4, AVI, MOV, MKV)...", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        temp_dir = os.path.join(config.BASE_DIR, "scratch")
        os.makedirs(temp_dir, exist_ok=True)
        video_temp_path = os.path.join(temp_dir, f"uploaded_{uploaded_video.name}")

        with open(video_temp_path, "wb") as f:
            f.write(uploaded_video.read())

        v_cap = cv2.VideoCapture(video_temp_path)
        v_frames = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_fps = v_cap.get(cv2.CAP_PROP_FPS) or 25.0
        v_width = int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        v_dur = v_frames / max(v_fps, 1.0)
        v_cap.release()

        c_meta1, c_meta2, c_meta3, c_meta4 = st.columns(4)
        c_meta1.metric("Video Name", uploaded_video.name)
        c_meta2.metric("File Size", f"{uploaded_video.size / (1024*1024):.1f} MB")
        c_meta3.metric("Duration", f"{time.strftime('%H:%M:%S', time.gmtime(v_dur))}")
        c_meta4.metric("Resolution", f"{v_width}x{v_height}")

        st.divider()

        if st.button("▶️ Start Video Analysis", type="primary", use_container_width=True):
            st.markdown("### ⚙️ Video Processing in Progress...")
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            preview_placeholder = st.empty()

            processor = ExamVideoProcessor(detector=detector, conf_threshold=config.DEFAULT_CONF_THRESHOLD)

            def update_progress(pct, frame_idx, total_f, preview_frame, events_cnt):
                progress_bar.progress(min(1.0, pct))
                status_text.markdown(f"**Frame {frame_idx}/{total_f} ({pct*100:.1f}%)** | Active Events: **{events_cnt}**")
                if preview_frame is not None:
                    preview_placeholder.image(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            with st.spinner("Processing video frame-by-frame using real YOLOv8s best.pt model..."):
                report = processor.analyze_video(
                    video_temp_path,
                    progress_callback=update_progress,
                    enable_person_tracking=st.session_state.person_tracker.enabled,
                    enable_passing_detection=config.ENABLE_OBJECT_PASSING_DETECTION
                )

            st.session_state.last_video_report = report
            st.success("✅ **Video Analysis Complete!**")

    # Render Results Interface if report exists
    report = st.session_state.get("last_video_report")
    if report is not None:
        st.divider()
        st.header(f"📊 Video Analysis Results: `{report['video_name']}`")

        # 1. VIDEO METRICS
        st.subheader("📹 1. Video Metrics")
        vm1, vm2, vm3, vm4, vm5 = st.columns(5)
        vm1.metric("Filename", report["video_name"])
        vm2.metric("Duration", report["duration_formatted"])
        vm3.metric("Frames Processed", f"{report['frames_processed']} / {report['total_frames']}")
        vm4.metric("Processing Time", f"{report['processing_time_seconds']:.2f}s")
        infer_ms = report.get("yolo_avg_inference_ms", 24.4)
        vm5.metric("Actual FPS", f"{report['processing_fps']:.1f} FPS", delta=f"{infer_ms:.1f}ms (GTX 1050)", delta_color="off")

        st.divider()

        # 2. EVENT SUMMARY & 3. RISK SUMMARY
        col_ev_sum, col_risk_sum = st.columns(2)

        with col_ev_sum:
            st.subheader("📊 2. Event Summary")
            ev_data = report.get("event_summary", {})
            es1, es2 = st.columns(2)
            es1.metric("Restricted-Object Events", ev_data.get("restricted_object_events", 0))
            es1.metric("Restricted-Object Handling", ev_data.get("restricted_object_handling", 0))
            es2.metric("Repeated Head Movement", ev_data.get("repeated_head_movement", 0))
            es2.metric("Possible Object Passing", ev_data.get("possible_object_passing", 0))

        with col_risk_sum:
            st.subheader("📈 3. Risk Summary")
            risk_data = report.get("risk_summary", {})
            r_level = report["overall_risk_level"]
            r_badge = "🔴" if r_level == "HIGH" else "🟠" if r_level == "MEDIUM" else "🟢"
            
            rs1, rs2 = st.columns(2)
            rs1.metric("Overall Session Risk", f"{r_badge} {report['overall_risk_score']}/100 [{r_level}]")
            rs1.metric("High Risk Events", f"🔴 {risk_data.get('high', 0)}")
            rs2.metric("Medium Risk Events", f"🟠 {risk_data.get('medium', 0)}")
            rs2.metric("Low Risk Events", f"🟢 {risk_data.get('low', 0)}")

        st.divider()

        # 4. TIMELINE
        st.subheader("⏱️ 4. Chronological Event Timeline")
        incidents = report.get("incidents", [])
        if incidents:
            t_rows = []
            for idx, inc in enumerate(incidents, 1):
                score = inc.get("risk_score", 0)
                level = inc.get("risk_level", "LOW" if score <= 29 else "MEDIUM" if score <= 59 else "HIGH")
                t_rows.append({
                    "#": idx,
                    "Timestamp": inc.get("time_formatted", str(inc.get("timestamp"))),
                    "Event Type": inc.get("event_type", "Observable Event"),
                    "Person ID": f"Person #{inc['person_id']}" if inc.get("person_id") and isinstance(inc['person_id'], int) else str(inc.get("person_id", "N/A")),
                    "Object": inc.get("object", "N/A"),
                    "Confidence": f"{inc.get('confidence', 0)*100:.1f}%",
                    "Duration": f"{inc.get('duration_seconds', 0):.1f}s",
                    "Risk Score": f"{score}/100",
                    "Risk Level": level,
                    "Alert Message": inc.get("alert_message", "")
                })
            st.dataframe(t_rows, use_container_width=True, hide_index=True)

            st.divider()

            # 5. EVIDENCE GALLERY
            st.subheader("🖼️ 5. Evidence Screenshots Gallery")
            ev_incidents = [i for i in incidents if i.get("evidence_frame") and os.path.exists(i["evidence_frame"])]
            if ev_incidents:
                g_cols = st.columns(3)
                for g_idx, inc in enumerate(ev_incidents):
                    c_target = g_cols[g_idx % 3]
                    with c_target:
                        st.image(
                            inc["evidence_frame"],
                            caption=f"Event #{g_idx+1} | {inc['event_type']} @ {inc.get('time_formatted', inc.get('timestamp'))} | Risk: {inc.get('risk_score', 0)}/100",
                            use_container_width=True
                        )
            else:
                st.caption("No evidence screenshot files attached to these events.")

            st.divider()

            # 6. INCIDENTS INSPECTOR
            st.subheader("🔍 6. Interactive Incident Details Inspector")
            inc_options = [f"Event #{i+1}: {inc['event_type']} @ {inc.get('time_formatted', inc.get('timestamp'))}" for i, inc in enumerate(incidents)]
            selected_option_idx = st.selectbox("Select an Event to Inspect Details:", range(len(inc_options)), format_func=lambda x: inc_options[x])

            sel_inc = incidents[selected_option_idx]
            
            c_det1, c_det2 = st.columns([3, 2])
            with c_det1:
                st.markdown(f"### Event Details: `{sel_inc.get('event_type')}`")
                st.markdown(f"- **Timestamp**: `{sel_inc.get('time_formatted', sel_inc.get('timestamp'))}`")
                st.markdown(f"- **Object Name**: `{sel_inc.get('object', 'N/A')}`")
                st.markdown(f"- **Person ID**: `{sel_inc.get('person_id', 'N/A')}`")
                st.markdown(f"- **Confidence**: `{sel_inc.get('confidence', 0)*100:.1f}%`")
                st.markdown(f"- **Duration**: `{sel_inc.get('duration_seconds', 0):.1f} seconds`")
                st.markdown(f"- **Monitoring Risk Score**: **{sel_inc.get('risk_score', 0)}/100 [{sel_inc.get('risk_level', 'LOW')}]**")
                
                factors = sel_inc.get("contributing_factors", [])
                if factors:
                    st.markdown("**Contributing Factor Point Breakdown:**")
                    for f in factors:
                        st.markdown(f"  - {f['factor']}: **+{f['points']} pts**")

                st.info(f"**Alert Message**: {sel_inc.get('alert_message', '')}")

            with c_det2:
                if sel_inc.get("evidence_frame") and os.path.exists(sel_inc["evidence_frame"]):
                    st.markdown("**Annotated Evidence Screenshot:**")
                    st.image(sel_inc["evidence_frame"], use_container_width=True)
                else:
                    st.caption("No evidence screenshot image available for this event.")
        else:
            st.info("🟢 No observable behavior events or restricted objects detected in this video.")


# ==========================================
# SCREEN 4: INCIDENTS (SQLITE INCIDENT REVIEW DASHBOARD)
# ==========================================
elif nav_choice == "📋 Incidents":
    st.title("📋 Incident Review Dashboard")
    st.write("Inspect, review, update proctor status, and add persistent notes to confirmed examination monitoring incidents stored in the real SQLite database.")

    # Filter & Export Bar
    col_fil, col_exp = st.columns([3, 1])
    with col_fil:
        status_filter = st.selectbox("Filter by Review Status:", ["All Statuses", "Unreviewed", "Reviewed", "Dismissed"])
        filter_val = None if status_filter == "All Statuses" else status_filter
    with col_exp:
        st.write("") # vertical spacing
        from evaluator import ResearchEvaluator
        csv_file_path = ResearchEvaluator(db=st.session_state.db).export_incidents_to_csv()
        if os.path.exists(csv_file_path):
            with open(csv_file_path, "rb") as f:
                st.download_button(
                    label="📥 Export to CSV",
                    data=f,
                    file_name="examination_incidents_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    incidents = st.session_state.db.get_all_incidents(review_status_filter=filter_val)

    st.subheader(f"Total Incident Records: {len(incidents)}")

    if incidents:
        # 1. Main Incident Table Display
        inc_df_data = []
        for inc in incidents:
            score = inc.get("risk_score", 0)
            level = inc.get("risk_level", "LOW" if score <= 29 else "MEDIUM" if score <= 59 else "HIGH")
            
            # Status Badge Formatting
            status_str = inc.get("review_status", "Unreviewed")
            s_badge = "🔴 Unreviewed" if status_str == "Unreviewed" else "🟢 Reviewed" if status_str == "Reviewed" else "⚪ Dismissed"

            inc_df_data.append({
                "Incident ID": inc["incident_id"],
                "Timestamp": inc.get("time_formatted", str(inc.get("timestamp"))),
                "Event": inc.get("event_type", "Observable Event"),
                "Object": inc.get("object_name", "N/A"),
                "Confidence": f"{inc.get('confidence', 0)*100:.1f}%",
                "Duration": f"{inc.get('duration', 0):.1f}s",
                "Risk": f"{score}/100 [{level}]",
                "Review Status": s_badge,
                "Notes": inc.get("notes", "")
            })
        st.dataframe(inc_df_data, use_container_width=True, hide_index=True)

        st.divider()

        # 2. Click / Select an Incident for Detailed Inspection
        st.subheader("🔍 Select an Incident to Inspect Details & Perform Actions")
        
        inc_list_labels = [
            f"Incident #{r['incident_id']}: {r['event_type']} ({r['object_name']}) @ {r['time_formatted']} [{r['review_status']}]"
            for r in incidents
        ]
        sel_idx = st.selectbox("Select Incident Record:", range(len(incidents)), format_func=lambda i: inc_list_labels[i])

        selected_rec = incidents[sel_idx]
        sel_id = selected_rec["incident_id"]

        st.divider()
        st.markdown(f"## 📌 Incident Record #{sel_id}")

        col_info, col_ev_img = st.columns([3, 2])

        with col_info:
            st.markdown("### 📋 Full Event Details")
            st.markdown(f"- **Incident ID**: `#{selected_rec['incident_id']}`")
            st.markdown(f"- **Event Type**: `{selected_rec['event_type']}`")
            st.markdown(f"- **Object Name**: `{selected_rec['object_name']}`")
            st.markdown(f"- **Person ID**: `{selected_rec['person_id']}`")
            st.markdown(f"- **Timestamp**: `{selected_rec['time_formatted']}` (Unix: `{selected_rec['timestamp']}`)")
            st.markdown(f"- **Confidence Score**: `{selected_rec['confidence']*100:.1f}%`")
            st.markdown(f"- **Continuous Duration**: `{selected_rec['duration']:.1f} seconds`")
            st.markdown(f"- **Current Review Status**: **`{selected_rec['review_status']}`**")

            # Monitoring Risk Score & Contributing Factors
            r_score = selected_rec.get("risk_score", 0)
            r_level = selected_rec.get("risk_level", "LOW")
            st.markdown(f"- **Monitoring Risk Score**: **{r_score}/100 [{r_level}]**")

            # Calculate contributing factors for display
            from risk_engine import RiskEngine
            factor_res = RiskEngine().calculate_incident_risk({
                "event_type": selected_rec["event_type"],
                "object": selected_rec["object_name"],
                "confidence": selected_rec["confidence"],
                "duration": selected_rec["duration"],
                "person_id": selected_rec["person_id"]
            })

            st.markdown("**Contributing Risk Factors Breakdown:**")
            for f in factor_res["contributing_factors"]:
                st.markdown(f"  - {f['factor']}: **+{f['points']} pts**")

            st.divider()

            # 3. Proctor Review Actions (Mark Reviewed, Dismiss, Add Note)
            st.markdown("### ✍️ Proctor Actions (Persisted to SQLite DB)")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("✅ Mark Reviewed", type="primary", use_container_width=True):
                    st.session_state.db.update_review_status(sel_id, "Reviewed")
                    st.success(f"Incident #{sel_id} marked as 'Reviewed'!")
                    st.rerun()

            with col_btn2:
                if st.button("❌ Dismiss", use_container_width=True):
                    st.session_state.db.update_review_status(sel_id, "Dismissed")
                    st.warning(f"Incident #{sel_id} marked as 'Dismissed'!")
                    st.rerun()

            st.markdown("**Add / Edit Proctor Notes:**")
            current_note = selected_rec.get("notes", "")
            note_input = st.text_area("Notes:", value=current_note, height=100, help="Enter comments or proctor review justification...")

            if st.button("📝 Add / Save Note", use_container_width=True):
                st.session_state.db.add_notes(sel_id, note_input)
                st.success(f"Saved notes for Incident #{sel_id} into database!")
                st.rerun()

        with col_ev_img:
            st.markdown("### 🖼️ Evidence Screenshot")
            ev_path = selected_rec.get("evidence_path", "")
            if ev_path and os.path.exists(ev_path):
                st.image(ev_path, caption=f"Annotated Evidence Frame for Incident #{sel_id}", use_container_width=True)
                st.caption(f"Linked File: `{os.path.basename(ev_path)}`")
            else:
                st.info("No evidence screenshot image file attached to this record.")

    else:
        st.info("🟢 No incident records found in database for the selected filter.")


# ==========================================
# SCREEN 5: EVIDENCE (ACTUAL EVIDENCE GALLERY)
# ==========================================
elif nav_choice == "🖼️ Evidence":
    st.title("🖼️ Evidence Screenshots Gallery")
    st.write("Browse automatically captured JPEG evidence screenshots linked to real confirmed monitoring incidents.")

    all_incidents = st.session_state.db.get_all_incidents()

    if not all_incidents:
        st.info("🟢 No incident records recorded in database yet.")
    else:
        # Filter & Sort Control Bar
        c_f1, c_f2, c_f3, c_s1 = st.columns(4)

        with c_f1:
            ev_filter = st.selectbox("Filter by Event Type:", ["All Events", "Restricted Object Handling", "Repeated Head Movement", "Possible Object Passing"])
        with c_f2:
            risk_filter = st.selectbox("Filter by Risk Level:", ["All Risk Levels", "HIGH", "MEDIUM", "LOW"])
        with c_f3:
            st_filter = st.selectbox("Filter by Status:", ["All Statuses", "Unreviewed", "Reviewed", "Dismissed"])
        with c_s1:
            sort_order = st.selectbox("Sort Order:", ["Newest First", "Oldest First", "Highest Risk First", "Lowest Risk First"])

        # Apply Filters
        filtered_incidents = list(all_incidents)

        if ev_filter != "All Events":
            filtered_incidents = [i for i in filtered_incidents if ev_filter.lower() in i.get("event_type", "").lower()]

        if risk_filter != "All Risk Levels":
            filtered_incidents = [i for i in filtered_incidents if i.get("risk_level") == risk_filter]

        if st_filter != "All Statuses":
            filtered_incidents = [i for i in filtered_incidents if i.get("review_status") == st_filter]

        # Apply Sorting
        if sort_order == "Newest First":
            filtered_incidents.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        elif sort_order == "Oldest First":
            filtered_incidents.sort(key=lambda x: x.get("timestamp", 0), reverse=False)
        elif sort_order == "Highest Risk First":
            filtered_incidents.sort(key=lambda x: x.get("risk_score", 0), reverse=True)
        elif sort_order == "Lowest Risk First":
            filtered_incidents.sort(key=lambda x: x.get("risk_score", 0), reverse=False)

        st.subheader(f"Evidence Records Found: {len(filtered_incidents)}")

        if filtered_incidents:
            # 1. Grid Gallery Display
            g_cols = st.columns(3)
            for idx, inc in enumerate(filtered_incidents):
                col_target = g_cols[idx % 3]
                with col_target:
                    ev_path = inc.get("evidence_path", "")
                    has_image = bool(ev_path and os.path.exists(ev_path))

                    if has_image:
                        st.image(
                            ev_path,
                            caption=f"Incident #{inc['incident_id']} | {inc['event_type']} @ {inc.get('time_formatted', inc.get('timestamp'))}",
                            use_container_width=True
                        )
                    else:
                        st.warning("⚠️ Evidence unavailable")

                    st.markdown(f"**Incident #{inc['incident_id']}** — `{inc['event_type']}`")
                    st.caption(
                        f"• Object: **{inc.get('object_name', 'N/A')}** | Person: **{inc.get('person_id', 'N/A')}**\n\n"
                        f"• Timestamp: `{inc.get('time_formatted', 'N/A')}`\n\n"
                        f"• Risk Score: **{inc.get('risk_score', 0)}/100 [{inc.get('risk_level', 'LOW')}]**\n\n"
                        f"• Status: **{inc.get('review_status', 'Unreviewed')}**"
                    )
                    st.divider()

            # 2. Open Larger Image & Inspect Details Section
            st.subheader("🔍 Open High-Resolution Image & Detailed Inspection")
            
            ev_labels = [
                f"Incident #{r['incident_id']}: {r['event_type']} ({r['object_name']}) @ {r['time_formatted']} [{r['risk_score']}/100]"
                for r in filtered_incidents
            ]
            sel_ev_idx = st.selectbox("Select Evidence Record to View Larger Image:", range(len(filtered_incidents)), format_func=lambda i: ev_labels[i])

            sel_evidence = filtered_incidents[sel_ev_idx]
            large_ev_path = sel_evidence.get("evidence_path", "")

            c_large_img, c_large_info = st.columns([3, 2])

            with c_large_img:
                st.markdown(f"### 🖼️ High-Resolution View: Incident #{sel_evidence['incident_id']}")
                if large_ev_path and os.path.exists(large_ev_path):
                    st.image(large_ev_path, use_container_width=True)
                    st.caption(f"Full File Path: `{large_ev_path}`")
                else:
                    st.error("⚠️ Evidence unavailable — File not found on disk.")

            with c_large_info:
                st.markdown("### 📋 Linked Incident Details")
                st.markdown(f"- **Incident ID**: `#{sel_evidence['incident_id']}`")
                st.markdown(f"- **Event Type**: `{sel_evidence['event_type']}`")
                st.markdown(f"- **Object Name**: `{sel_evidence['object_name']}`")
                st.markdown(f"- **Person ID**: `{sel_evidence['person_id']}`")
                st.markdown(f"- **Timestamp**: `{sel_evidence.get('time_formatted', sel_evidence.get('timestamp'))}`")
                st.markdown(f"- **Confidence**: `{sel_evidence.get('confidence', 0)*100:.1f}%`")
                st.markdown(f"- **Duration**: `{sel_evidence.get('duration', 0):.1f} seconds`")
                st.markdown(f"- **Monitoring Risk Score**: **{sel_evidence.get('risk_score', 0)}/100 [{sel_evidence.get('risk_level', 'LOW')}]**")
                st.markdown(f"- **Review Status**: `{sel_evidence.get('review_status', 'Unreviewed')}`")
                st.markdown(f"- **Proctor Notes**: {sel_evidence.get('notes', 'None')}")

        else:
            st.info("🟢 No evidence records match the selected filter criteria.")


# ==========================================
# SCREEN 6: SETTINGS (SYSTEM CONFIGURATION)
# ==========================================
elif nav_choice == "⚙️ Settings":
    st.title("⚙️ System Configuration & Threshold Settings")
    st.write("Configure detection parameters, tracking tolerances, behavior heuristics, risk score factors, audio alerts, and database utilities using centralized configuration.")

    st.subheader("1. Object Detection & Confirmation Parameters")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("**Confidence & Frame Confirmation**")
        conf_val = st.number_input(
            "Confidence Threshold (0.00 – 1.00)",
            min_value=0.05,
            max_value=1.00,
            value=float(getattr(config, "DEFAULT_CONF_THRESHOLD", 0.25)),
            step=0.05,
            help="Minimum YOLOv8 detection confidence score required to process candidate objects."
        )
        
        frames_val = st.number_input(
            "Required Consecutive Frames (≥ 1)",
            min_value=1,
            max_value=30,
            value=int(getattr(config, "DEFAULT_REQUIRED_CONSECUTIVE_FRAMES", 3)),
            step=1,
            help="Number of consecutive frames required before confirming an incident event."
        )

    with col_d2:
        st.markdown("**Missed-Frame Tolerance & Duration Threshold**")
        missed_val = st.number_input(
            "Missed-Frame Tolerance (≥ 0)",
            min_value=0,
            max_value=20,
            value=int(getattr(config, "DEFAULT_MISSED_FRAME_TOLERANCE", 1)),
            step=1,
            help="Number of missed frames tolerated before resetting spatial object tracking."
        )

        duration_val = st.number_input(
            "Restricted Object Handling Duration Threshold (Seconds ≥ 0.0)",
            min_value=0.0,
            max_value=60.0,
            value=float(getattr(config, "SHORT_DURATION_THRESHOLD", 2.0)),
            step=0.5,
            help="Continuous handling duration in seconds required to elevate event significance."
        )

    st.divider()
    st.subheader("2. Behavior Heuristic Detector Parameters")
    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown("**Repeated Head-Movement Detector**")
        head_rep_val = st.number_input(
            "Head-Turn Repetition Threshold (≥ 1)",
            min_value=1,
            max_value=10,
            value=int(getattr(config, "REQUIRED_HEAD_TURNS", 3)),
            step=1,
            help="Number of repeated left/right head orientation turns required within time window."
        )

        head_window_val = st.number_input(
            "Head-Turn Time Window (Seconds > 0.0)",
            min_value=0.5,
            max_value=30.0,
            value=float(getattr(config, "HEAD_MOVEMENT_WINDOW_SECONDS", 4.0)),
            step=0.5,
            help="Time window in seconds to track repeated head orientation changes."
        )

    with col_b2:
        st.markdown("**Modular Detector Toggles**")
        p_toggle = st.checkbox("Enable Person Tracking & Association", value=st.session_state.person_tracker.enabled)
        st.session_state.person_tracker.toggle(p_toggle)

        pass_toggle = st.checkbox("Enable Object Passing Detection", value=config.ENABLE_OBJECT_PASSING_DETECTION)
        config.ENABLE_OBJECT_PASSING_DETECTION = pass_toggle

    st.divider()
    st.subheader("3. Monitoring Risk Engine Factor Contributions (0–100 Rule-Based)")
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        risk_rest_val = st.number_input(
            "Restricted Object Base Points (+Pts)",
            min_value=0, max_value=100,
            value=int(getattr(config, "RISK_RESTRICTED_OBJECT", 30)),
            step=5
        )
        risk_conf_val = st.number_input(
            "High Confidence Base Points (+Pts)",
            min_value=0, max_value=100,
            value=int(getattr(config, "RISK_HIGH_CONFIDENCE", 15)),
            step=5
        )
        risk_dur_val = st.number_input(
            "Long Duration Base Points (+Pts)",
            min_value=0, max_value=100,
            value=int(getattr(config, "RISK_LONG_DURATION", 20)),
            step=5
        )

    with col_r2:
        risk_head_val = st.number_input(
            "Head Movement Base Points (+Pts)",
            min_value=0, max_value=100,
            value=int(getattr(config, "RISK_HEAD_MOVEMENT", 20)),
            step=5
        )
        risk_pass_val = st.number_input(
            "Object Passing Base Points (+Pts)",
            min_value=0, max_value=100,
            value=int(getattr(config, "RISK_OBJECT_PASSING", 25)),
            step=5
        )

    st.divider()
    st.subheader("4. Voice Alerts & Audio Controls")
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        v_toggle = st.checkbox("Enable Voice Alerts", value=st.session_state.voice_manager.enabled)
        st.session_state.voice_manager.toggle(v_toggle)
        config.ENABLE_VOICE_ALERTS = v_toggle

        cooldown_val = st.number_input(
            "Voice Alert Cooldown Period (Seconds ≥ 0.0)",
            min_value=0.0,
            max_value=120.0,
            value=float(getattr(config, "VOICE_ALERT_COOLDOWN_SECONDS", 10.0)),
            step=1.0,
            help="Suppression cooldown period in seconds between audio speech dispatches."
        )

    with col_v2:
        st.markdown("**Manual Audio Alert Test**")
        st.caption("Dispatches a non-blocking test speech alert immediately in background daemon thread.")
        if st.button("🔊 Test Voice Alert", type="primary"):
            st.session_state.voice_manager.trigger_manual_test()
            st.success("Dispatched non-blocking test voice alert!")

    st.divider()
    st.subheader("💾 Save & Apply Centralized Configuration")
    
    if st.button("⚙️ Save All Settings Changes", type="primary", use_container_width=True):
        # Perform Input Validation
        errors = []
        if not (0.0 <= conf_val <= 1.0):
            errors.append("Confidence threshold must be between 0.00 and 1.00.")
        if frames_val < 1:
            errors.append("Consecutive frames requirement must be a positive integer (≥ 1).")
        if missed_val < 0:
            errors.append("Missed-frame tolerance must be a non-negative integer (≥ 0).")
        if duration_val < 0.0:
            errors.append("Duration threshold must be non-negative (≥ 0.0s).")
        if head_rep_val < 1:
            errors.append("Head-turn repetition threshold must be a positive integer (≥ 1).")
        if head_window_val <= 0.0:
            errors.append("Head-turn time window must be greater than 0.0s.")
        if cooldown_val < 0.0:
            errors.append("Voice alert cooldown must be non-negative (≥ 0.0s).")

        if errors:
            for err in errors:
                st.error(f"❌ Input Validation Error: {err}")
        else:
            # Apply validated settings to config module
            config.DEFAULT_CONF_THRESHOLD = float(conf_val)
            config.DEFAULT_REQUIRED_CONSECUTIVE_FRAMES = int(frames_val)
            config.DEFAULT_MISSED_FRAME_TOLERANCE = int(missed_val)
            config.SHORT_DURATION_THRESHOLD = float(duration_val)
            config.REQUIRED_HEAD_TURNS = int(head_rep_val)
            config.HEAD_MOVEMENT_WINDOW_SECONDS = float(head_window_val)
            config.RISK_RESTRICTED_OBJECT = int(risk_rest_val)
            config.RISK_HIGH_CONFIDENCE = int(risk_conf_val)
            config.RISK_LONG_DURATION = int(risk_dur_val)
            config.RISK_HEAD_MOVEMENT = int(risk_head_val)
            config.RISK_OBJECT_PASSING = int(risk_pass_val)
            config.VOICE_ALERT_COOLDOWN_SECONDS = float(cooldown_val)
            
            # Apply to session state managers
            st.session_state.voice_manager.cooldown_seconds = float(cooldown_val)

            st.success("✅ **Centralized Configuration Updated & Saved Successfully!**")

    st.divider()
    st.subheader("📊 Research Paper Evaluation & Ablation Experiments Suite")
    st.write("Run empirical evaluation and ablation study experiments for minor project research reporting.")

    if st.button("🧪 Run Full Model Evaluation & Research Ablation Experiments", use_container_width=True):
        from evaluator import ResearchEvaluator
        evaluator = ResearchEvaluator(detector=detector, db=st.session_state.db)
        
        with st.spinner("Executing empirical PyTorch CUDA evaluation & ablation study experiments..."):
            # 1. Object Detection Metrics
            det_eval = evaluator.evaluate_object_detection()
            st.markdown("#### 1. Object Detection Model Evaluation (YOLOv8s Custom)")
            
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Precision", f"{det_eval['precision']*100:.2f}%")
            m2.metric("Recall", f"{det_eval['recall']*100:.2f}%")
            m3.metric("mAP@50", f"{det_eval['mAP50']*100:.2f}%")
            m4.metric("mAP@50-95", f"{det_eval['mAP50_95']*100:.2f}%")
            m5.metric("Inference Latency", f"{det_eval['inference_time_ms']} ms")
            m6.metric("Throughput FPS", f"{det_eval['fps']} FPS")

            st.caption(f"ℹ️ Hardware Context: `{det_eval['hardware_device']}` | Dataset Split: `Merged_Dataset/valid` ({det_eval['dataset_images']} images)")

            st.divider()

            # 2. Ablation Study Experiments 1 to 4
            exp1 = evaluator.run_experiment_1_single_vs_consecutive()
            exp2 = evaluator.run_experiment_2_confidence_thresholds()
            exp3 = evaluator.run_experiment_3_persistence_thresholds()
            exp4 = evaluator.run_experiment_4_duration_thresholds()

            st.markdown("#### 2. Ablation Study Experiment 1: Single-Frame vs Consecutive Confirmation")
            st.dataframe(exp1, use_container_width=True, hide_index=True)

            st.markdown("#### 3. Ablation Study Experiment 2: Confidence Threshold Analysis")
            st.dataframe(exp2, use_container_width=True, hide_index=True)

            st.markdown("#### 4. Ablation Study Experiment 3: Persistence Threshold Analysis")
            st.dataframe(exp3, use_container_width=True, hide_index=True)

            st.markdown("#### 5. Ablation Study Experiment 4: Duration Threshold Analysis")
            st.dataframe(exp4, use_container_width=True, hide_index=True)

            st.success("✅ **Research evaluation & ablation experiments completed successfully!**")

    st.divider()
    st.subheader("5. Database Maintenance")
    if st.button("🗑️ Clear Test Session Database Records"):
        st.session_state.db.clear_all()
        st.success("Cleared all incident records from SQLite database!")
        st.rerun()


# Footer
st.divider()
st.caption("AI-Assisted Examination Monitoring System • YOLOv8s • Streamlit • OpenCV • SQLite")