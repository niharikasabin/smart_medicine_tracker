"""
=============================================================
app/streamlit_app.py — Smart Medicine Adherence Tracker GUI
=============================================================
Main Streamlit application. Run with:
    streamlit run app/streamlit_app.py

Features:
  - Camera / image upload for medicine detection (YOLOv8)
  - Mark doses as taken or missed
  - Adherence dashboard with charts
  - LSTM-based miss probability prediction
  - High-risk alerts
  - Multi-medicine tracking
  - Streak and statistics cards
=============================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# ── Internal modules ──────────────────────────────────────
from app.database import Database
from app.reminder import AlertManager, EmailReminder
from models.inference import MedicineDetector
from models.lstm_model import AdherencePredictor
import utils.visualize as viz


# ═══════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title      = "💊 Medicine Tracker",
    page_icon       = "💊",
    layout          = "wide",
    initial_sidebar_state = "expanded",
    menu_items      = {
        "About": "Smart Medicine Adherence Tracker — AI Project",
    },
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0E1117; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1E2130 0%, #252840 100%);
        border: 1px solid #2D3148;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 6px 0;
        transition: transform 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #FAFAFA; }
    .metric-label { font-size: 0.85rem; color: #888; margin-top: 4px; }
    .metric-delta { font-size: 0.8rem; margin-top: 6px; }
    
    /* Status badges */
    .badge-green  { background:#1A3A2A; color:#4CAF50; border:1px solid #4CAF50; border-radius:20px; padding:4px 12px; font-size:0.8rem; font-weight:600; }
    .badge-yellow { background:#3A2E1A; color:#FF9800; border:1px solid #FF9800; border-radius:20px; padding:4px 12px; font-size:0.8rem; font-weight:600; }
    .badge-red    { background:#3A1A1A; color:#F44336; border:1px solid #F44336; border-radius:20px; padding:4px 12px; font-size:0.8rem; font-weight:600; }
    
    /* Alert box */
    .alert-high {
        background: linear-gradient(135deg, #3A1A1A, #2D1515);
        border: 1px solid #F44336;
        border-left: 4px solid #F44336;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { border-left-color: #F44336; }
        50%       { border-left-color: #FF6B6B; }
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FAFAFA;
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #2D3148;
    }
    
    /* Detection result */
    .detection-box {
        background: #1E2130;
        border: 1px solid #2D3148;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    
    /* Remove streamlit default padding */
    .block-container { padding-top: 1.5rem; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1A1E2E; }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════

def init_session():
    """Initialize Streamlit session state."""
    defaults = {
        "user_id":          None,
        "user_name":        "",
        "detection_result": None,
        "last_prediction":  None,
        "alert_history":    [],
        "risk_history":     [],
        "selected_medicine_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════
# CACHED RESOURCES (loaded once per session)
# ═══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading AI detector...")
def get_detector():
    return MedicineDetector()

@st.cache_resource(show_spinner="Loading prediction model...")
def get_predictor():
    return AdherencePredictor()

@st.cache_resource
def get_db():
    db = Database()
    return db

@st.cache_resource
def get_alert_manager():
    return AlertManager()


# ═══════════════════════════════════════════════════════════
# HELPER COMPONENTS
# ═══════════════════════════════════════════════════════════

def metric_card(value, label, color="#4CAF50", prefix="", suffix=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color}">{prefix}{value}{suffix}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)


def risk_badge(level: str):
    badge_class = {
        "Low":    "badge-green",
        "Medium": "badge-yellow",
        "High":   "badge-red",
    }.get(level, "badge-green")
    st.markdown(f'<span class="{badge_class}">{level} Risk</span>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════

def render_sidebar():
    """Render the sidebar with user profile and medicine list."""
    db = get_db()
    
    with st.sidebar:
        st.markdown("## 💊 Medicine Tracker")
        st.markdown("---")
        
        # ── User setup ────────────────────────────────────
        st.markdown("### 👤 User Profile")
        
        users = db.list_users()
        if not users:
            # First-time setup
            st.info("Set up your profile to get started")
            with st.form("user_setup"):
                name  = st.text_input("Your name")
                email = st.text_input("Email (for reminders)", placeholder="optional")
                if st.form_submit_button("🚀 Get Started", use_container_width=True):
                    uid = db.add_user(name, email or None)
                    st.session_state.user_id   = uid
                    st.session_state.user_name = name
                    st.success(f"Welcome, {name}!")
                    st.rerun()
            return
        
        # User selector
        user_options = {u["name"]: u["user_id"] for u in users}
        user_options["➕ Add new user"] = -1
        
        selected_name = st.selectbox("Select user", list(user_options.keys()))
        
        if user_options[selected_name] == -1:
            new_name = st.text_input("New user name")
            if st.button("Add User") and new_name:
                uid = db.add_user(new_name)
                st.session_state.user_id   = uid
                st.session_state.user_name = new_name
                st.rerun()
        else:
            st.session_state.user_id   = user_options[selected_name]
            st.session_state.user_name = selected_name
        
        uid = st.session_state.user_id
        if not uid:
            return
        
        st.markdown("---")
        
        # ── Medicine management ───────────────────────────
        st.markdown("### 💊 My Medicines")
        medicines = db.get_medicines(uid)
        
        if medicines:
            for med in medicines:
                selected = st.session_state.selected_medicine_id == med["medicine_id"]
                label = f"{'▶ ' if selected else ''}{med['name']}"
                if st.button(label, key=f"med_{med['medicine_id']}", use_container_width=True):
                    st.session_state.selected_medicine_id = med["medicine_id"]
                    st.rerun()
        
        # Add medicine form
        with st.expander("➕ Add Medicine"):
            with st.form("add_medicine"):
                med_name = st.text_input("Medicine name", placeholder="e.g. Metformin 500mg")
                med_type = st.selectbox("Type", ["tablet", "strip", "bottle", "capsule"])
                dose_times_raw = st.text_input("Dose times (HH:MM, comma separated)", value="08:00,20:00")
                if st.form_submit_button("Add", use_container_width=True):
                    if med_name:
                        times = [t.strip() for t in dose_times_raw.split(",")]
                        mid = db.add_medicine(uid, med_name, med_type, times)
                        st.session_state.selected_medicine_id = mid
                        st.success(f"Added: {med_name}")
                        st.rerun()
        
        # Set default medicine
        if medicines and not st.session_state.selected_medicine_id:
            st.session_state.selected_medicine_id = medicines[0]["medicine_id"]
        
        st.markdown("---")
        
        # ── Quick actions ─────────────────────────────────
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌱 Seed Demo", use_container_width=True, help="Load demo data"):
                uid = db.seed_demo_data()
                st.session_state.user_id   = uid
                st.session_state.user_name = "Demo User"
                st.success("Demo data loaded!")
                st.rerun()
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        st.markdown("---")
        st.caption("Smart Medicine Tracker v1.0\nBuilt with YOLOv8 + LSTM")


# ═══════════════════════════════════════════════════════════
# TAB: DETECTION
# ═══════════════════════════════════════════════════════════

def render_detection_tab():
    """Medicine detection via image upload or camera."""
    uid = st.session_state.user_id
    mid = st.session_state.selected_medicine_id
    db  = get_db()
    
    st.markdown('<div class="section-header">📷 Medicine Detection</div>', unsafe_allow_html=True)
    
    if not uid:
        st.warning("Please select or create a user in the sidebar first.")
        return
    
    # ── Input Method ──────────────────────────────────────
    input_method = st.radio(
        "Input method",
        ["📁 Upload Image", "📷 Camera"],
        horizontal=True,
        label_visibility="collapsed",
    )
    
    image_input = None
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        if input_method == "📁 Upload Image":
            uploaded = st.file_uploader(
                "Upload a photo of your medicine",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                label_visibility="collapsed",
            )
            if uploaded:
                image_input = Image.open(uploaded).convert("RGB")
                st.image(image_input, caption="Uploaded Image", use_container_width=True)
        else:
            camera_photo = st.camera_input("Take a photo of your medicine")
            if camera_photo:
                image_input = Image.open(camera_photo).convert("RGB")
    
    # ── Run Detection ─────────────────────────────────────
    if image_input is not None:
        detect_btn = st.button("🔍 Detect Medicine", use_container_width=True, type="primary")
        
        if detect_btn:
            detector = get_detector()
            with st.spinner("Running AI detection..."):
                result = detector.detect(image_input)
            st.session_state.detection_result = result
    
    # ── Show Results ──────────────────────────────────────
    result = st.session_state.detection_result
    
    with col_result:
        if result is not None:
            annotated_pil = get_detector().to_pil(result)
            st.image(annotated_pil, caption="Detection Result", use_container_width=True)
            
            # Detection summary
            if result.medicine_found:
                st.success(f"✅ Detected {result.count} medicine object(s)")
                for cls, count in result.class_counts.items():
                    st.markdown(
                        f'<div class="detection-box">📦 <b>{cls}</b> × {count}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.warning("⚠️ No medicine detected in image")
    
    # ── Log Dose ──────────────────────────────────────────
    if result is not None and mid:
        st.markdown("---")
        st.markdown('<div class="section-header">📝 Log This Dose</div>', unsafe_allow_html=True)
        
        medicines  = db.get_medicines(uid)
        med_names  = {m["medicine_id"]: m["name"] for m in medicines}
        med_name   = med_names.get(mid, "Medicine")
        
        col_taken, col_missed, col_skip = st.columns([2, 2, 1])
        
        with col_taken:
            if st.button("✅ Taken — I took my medicine", use_container_width=True, type="primary"):
                conf = result.detections[0].confidence if result.detections else 1.0
                db.log_dose(uid, mid, taken=1, medicine_name=med_name, confidence=conf,
                            notes="Logged via detection")
                db.log_detection_event(uid, 
                    [{"class": d.class_name, "conf": d.confidence} for d in result.detections],
                    result.medicine_found)
                
                # Run prediction & check alerts
                _run_prediction_and_alert(uid, mid, med_name)
                
                st.balloons()
                st.success(f"✅ Logged: {med_name} — TAKEN at {datetime.now().strftime('%H:%M')}")
                st.session_state.detection_result = None
                st.rerun()
        
        with col_missed:
            if st.button("❌ Missed — I did not take it", use_container_width=True):
                db.log_dose(uid, mid, taken=0, medicine_name=med_name,
                            notes="Logged via detection — missed")
                _run_prediction_and_alert(uid, mid, med_name)
                st.error(f"❌ Logged: {med_name} — MISSED at {datetime.now().strftime('%H:%M')}")
                st.session_state.detection_result = None
                st.rerun()
        
        with col_skip:
            if st.button("Skip", use_container_width=True):
                st.session_state.detection_result = None
                st.rerun()
    
    # ── Manual log (no detection) ─────────────────────────
    elif uid and mid:
        st.markdown("---")
        st.markdown("**Or log manually without detection:**")
        col_a, col_b = st.columns(2)
        medicines = db.get_medicines(uid)
        med_names = {m["medicine_id"]: m["name"] for m in medicines}
        med_name  = med_names.get(mid, "Medicine")
        
        with col_a:
            if st.button(f"✅ Mark {med_name} as TAKEN", use_container_width=True):
                db.log_dose(uid, mid, 1, med_name)
                _run_prediction_and_alert(uid, mid, med_name)
                st.success("Logged ✅")
                st.rerun()
        with col_b:
            if st.button(f"❌ Mark {med_name} as MISSED", use_container_width=True):
                db.log_dose(uid, mid, 0, med_name)
                st.warning("Logged ❌")
                st.rerun()


# ═══════════════════════════════════════════════════════════
# TAB: DASHBOARD
# ═══════════════════════════════════════════════════════════

def render_dashboard_tab():
    """Adherence analytics dashboard."""
    uid = st.session_state.user_id
    mid = st.session_state.selected_medicine_id
    db  = get_db()
    
    if not uid:
        st.warning("Please select a user in the sidebar.")
        return
    
    medicines  = db.get_medicines(uid)
    if not medicines:
        st.info("No medicines registered yet. Add one in the sidebar.")
        return
    
    # ── Today's Summary Cards ─────────────────────────────
    st.markdown('<div class="section-header">📊 Today\'s Summary</div>', unsafe_allow_html=True)
    summary = db.get_today_summary(uid)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        color = "#4CAF50" if summary["adherence_pct"] >= 80 else "#FF9800" if summary["adherence_pct"] >= 60 else "#F44336"
        metric_card(f"{summary['adherence_pct']:.0f}%", "Today's Adherence", color=color)
    with c2:
        metric_card(summary["total_taken"], "Doses Taken", color="#4CAF50")
    with c3:
        metric_card(summary["total_missed"], "Doses Missed", color="#F44336")
    with c4:
        # Overall streak
        if mid:
            seq = db.get_daily_sequence(uid, mid, days=60)
            streak = 0
            for v in reversed(seq.tolist()):
                if v == 1:
                    streak += 1
                else:
                    break
            metric_card(streak, "🔥 Day Streak", color="#FF9800", suffix=" days")
        else:
            metric_card("—", "🔥 Day Streak")
    
    st.markdown("")
    
    # ── Charts Row 1 ──────────────────────────────────────
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown('<div class="section-header">📅 Daily Adherence</div>', unsafe_allow_html=True)
        days_filter = st.slider("Days to show", 7, 90, 30, key="days_slider")
        
        if mid:
            hist_df = db.get_adherence_history(uid, mid, days=days_filter)
        else:
            hist_df = db.get_adherence_history(uid, days=days_filter)
        
        fig = viz.daily_adherence_bar(hist_df, days=days_filter)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown('<div class="section-header">📈 This Week</div>', unsafe_allow_html=True)
        week_hist = db.get_adherence_history(uid, mid, days=7)
        taken_count  = int(week_hist["taken"].sum()) if not week_hist.empty else 0
        missed_count = int((week_hist["taken"] == 0).sum()) if not week_hist.empty else 0
        fig2 = viz.weekly_donut(taken_count, missed_count)
        st.plotly_chart(fig2, use_container_width=True)
    
    # ── Multi-medicine comparison ─────────────────────────
    if len(medicines) > 1:
        st.markdown('<div class="section-header">💊 All Medicines</div>', unsafe_allow_html=True)
        med_stats = []
        for m in medicines:
            hist = db.get_adherence_history(uid, m["medicine_id"], days=30)
            if not hist.empty:
                pct    = hist["taken"].mean() * 100
                seq_   = db.get_daily_sequence(uid, m["medicine_id"], days=30)
                streak_= 0
                for v in reversed(seq_.tolist()):
                    if v == 1: streak_ += 1
                    else: break
                med_stats.append({
                    "name":          m["name"],
                    "adherence_pct": pct,
                    "streak":        streak_,
                })
        if med_stats:
            fig3 = viz.multi_medicine_adherence(med_stats)
            st.plotly_chart(fig3, use_container_width=True)
    
    # ── Recent log ────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Recent Logs</div>', unsafe_allow_html=True)
    recent = db.get_adherence_history(uid, days=7).tail(10)
    if not recent.empty:
        display = recent[["timestamp", "medicine_name", "taken"]].copy()
        display["taken"] = display["taken"].map({1: "✅ Taken", 0: "❌ Missed"})
        display["timestamp"] = pd.to_datetime(display["timestamp"]).dt.strftime("%b %d %H:%M")
        display.columns = ["Time", "Medicine", "Status"]
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("No logs yet. Start by uploading a medicine image.")


# ═══════════════════════════════════════════════════════════
# TAB: AI PREDICTION
# ═══════════════════════════════════════════════════════════

def render_prediction_tab():
    """LSTM miss-probability prediction and risk display."""
    uid = st.session_state.user_id
    mid = st.session_state.selected_medicine_id
    db  = get_db()
    
    if not uid or not mid:
        st.warning("Select a user and medicine to see predictions.")
        return
    
    st.markdown('<div class="section-header">🔮 AI Risk Prediction</div>', unsafe_allow_html=True)
    st.caption("Our LSTM model analyzes your recent adherence patterns to predict the probability of missing your next dose.")
    
    medicines = db.get_medicines(uid)
    med_names = {m["medicine_id"]: m["name"] for m in medicines}
    med_name  = med_names.get(mid, "Medicine")
    
    col_run, col_info = st.columns([1, 2])
    with col_run:
        run_btn = st.button("🔮 Run Prediction", use_container_width=True, type="primary")
    with col_info:
        st.info(f"Analyzing last 14 days for: **{med_name}**")
    
    if run_btn or st.session_state.last_prediction:
        if run_btn:
            predictor = get_predictor()
            with st.spinner("Running LSTM prediction..."):
                prediction = predictor.predict_from_db(uid, mid)
            st.session_state.last_prediction = prediction
            
            # Track risk history
            st.session_state.risk_history.append({
                "date":             datetime.now().strftime("%Y-%m-%d %H:%M"),
                "miss_probability": prediction["miss_probability"],
            })
        
        prediction = st.session_state.last_prediction
        if not prediction:
            return
        
        miss_prob = prediction["miss_probability"]
        
        # ── Risk display ──────────────────────────────────
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prob_color = (
                "#4CAF50" if miss_prob < 0.3 else
                "#FF9800" if miss_prob < 0.6 else
                "#F44336"
            )
            metric_card(f"{miss_prob:.0%}", "Miss Probability", color=prob_color)
        
        with col2:
            metric_card(f"{1 - miss_prob:.0%}", "Take Probability", color="#4CAF50")
        
        with col3:
            metric_card(prediction["recent_streak"], "Day Streak", color="#FF9800", suffix=" days")
        
        st.markdown("")
        
        # ── Probability bar ───────────────────────────────
        fig = viz.miss_probability_bar(miss_prob)
        st.plotly_chart(fig, use_container_width=True)
        
        # ── Risk level badge + recommendation ─────────────
        risk_col, rec_col = st.columns([1, 3])
        with risk_col:
            risk_badge(prediction["risk_level"])
        with rec_col:
            st.markdown(f"💬 {prediction['recommendation']}")
        
        # ── HIGH RISK ALERT ───────────────────────────────
        if miss_prob >= 0.60:
            st.markdown(f"""
            <div class="alert-high">
                <strong>⚠️ HIGH MISS RISK DETECTED</strong><br>
                Our AI predicts a <strong>{miss_prob:.0%} probability</strong> that you may miss 
                your next dose of <strong>{med_name}</strong>.<br>
                <em>Please take your medicine now or set an immediate reminder.</em>
            </div>""", unsafe_allow_html=True)
            
            # Trigger notification
            alert_mgr = get_alert_manager()
            alert_mgr.check_and_alert(
                user_name        = st.session_state.user_name,
                user_email       = "",
                medicine_name    = med_name,
                miss_probability = miss_prob,
            )
        
        # ── Sequence visualization ────────────────────────
        st.markdown('<div class="section-header">📅 Recent 14-Day Pattern</div>', unsafe_allow_html=True)
        
        seq = prediction.get("sequence", [])
        if seq:
            # Visual sequence display
            cols = st.columns(len(seq))
            for i, (col, val) in enumerate(zip(cols, seq)):
                day = (datetime.now() - timedelta(days=len(seq) - i - 1)).strftime("%d")
                with col:
                    emoji = "✅" if val == 1 else "❌"
                    color = "#4CAF50" if val == 1 else "#F44336"
                    st.markdown(
                        f'<div style="text-align:center; font-size:1.2rem">{emoji}</div>'
                        f'<div style="text-align:center; font-size:0.65rem; color:#888">{day}</div>',
                        unsafe_allow_html=True,
                    )
        
        # ── Risk trend ────────────────────────────────────
        if len(st.session_state.risk_history) > 1:
            st.markdown('<div class="section-header">📉 Risk Trend</div>', unsafe_allow_html=True)
            fig2 = viz.risk_trend_chart(st.session_state.risk_history)
            st.plotly_chart(fig2, use_container_width=True)
        
        # ── Streak gauge ──────────────────────────────────
        st.markdown('<div class="section-header">🔥 Adherence Streak</div>', unsafe_allow_html=True)
        fig3 = viz.streak_gauge(prediction["recent_streak"])
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB: SETTINGS
# ═══════════════════════════════════════════════════════════

def render_settings_tab():
    """Settings and configuration."""
    st.markdown('<div class="section-header">⚙️ Settings & Configuration</div>', unsafe_allow_html=True)
    
    with st.expander("📧 Email Reminder Setup", expanded=False):
        st.info("""
        To enable real email reminders:
        1. Create a Gmail account (or use existing)
        2. Enable 2-Factor Authentication
        3. Generate an **App Password** (Google Account → Security → App Passwords)
        4. Enter credentials below
        """)
        smtp_email = st.text_input("Gmail address", placeholder="your@gmail.com")
        smtp_pass  = st.text_input("App password", type="password")
        if st.button("Save Email Config"):
            os.environ["SMTP_EMAIL"]    = smtp_email
            os.environ["SMTP_PASSWORD"] = smtp_pass
            st.success("Email credentials saved for this session!")
    
    with st.expander("🤖 Model Configuration", expanded=False):
        st.markdown("**YOLOv8 Model:**")
        model_path = st.text_input("Path to YOLO model", value="models/medicine_yolo.pt")
        conf_thresh = st.slider("Detection confidence threshold", 0.1, 0.9, 0.45, 0.05)
        
        st.markdown("**LSTM Model:**")
        lstm_path = st.text_input("Path to LSTM model", value="models/adherence_lstm.pt")
        seq_len   = st.slider("Prediction window (days)", 7, 30, 14)
        
        if st.button("Apply Config"):
            st.success("Configuration applied (restart app to take full effect)")
    
    with st.expander("🗃️ Database", expanded=False):
        db = get_db()
        uid = st.session_state.user_id
        
        if uid:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Database location", "data/adherence.db")
            with col2:
                hist = db.get_adherence_history(uid, days=365)
                st.metric("Total log entries", len(hist))
            
            if st.button("📤 Export Logs as CSV"):
                hist = db.get_adherence_history(uid, days=365)
                csv = hist.to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv,
                    file_name=f"adherence_logs_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
    
    with st.expander("ℹ️ About This Project", expanded=False):
        st.markdown("""
        ### Smart Medicine Adherence Tracker
        
        **Technologies used:**
        - 🎯 **YOLOv8** (Ultralytics) — Computer vision for medicine detection
        - 🧠 **LSTM** (PyTorch) — Time-series prediction for adherence
        - 🗃️ **SQLite** — Local persistent storage
        - 📊 **Streamlit + Plotly** — Interactive dashboard
        - 📧 **smtplib** — Email notifications
        
        **Architecture:**
        ```
        User Input (Image/Camera)
              ↓
        YOLOv8 Detection
              ↓
        SQLite Database (time-series logs)
              ↓
        LSTM Model (predict miss probability)
              ↓
        Streamlit Dashboard + Alerts
        ```
        
        **For viva:**
        > The LSTM receives a 14-day binary sequence of adherence data (1=taken, 0=missed) 
        > and outputs the probability of missing the next dose. YOLOv8, trained on medicine 
        > datasets, detects pill strips and bottles in real-time from camera or uploaded images.
        """)


# ═══════════════════════════════════════════════════════════
# HELPER: Run prediction and check alerts
# ═══════════════════════════════════════════════════════════

def _run_prediction_and_alert(uid: int, mid: int, med_name: str):
    """Run prediction after logging a dose and trigger alerts if high risk."""
    try:
        predictor = get_predictor()
        prediction = predictor.predict_from_db(uid, mid)
        st.session_state.last_prediction = prediction
        st.session_state.risk_history.append({
            "date":             datetime.now().strftime("%Y-%m-%d %H:%M"),
            "miss_probability": prediction["miss_probability"],
        })
    except Exception as e:
        pass   # Non-critical — prediction can fail silently


# ═══════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════

def main():
    init_session()
    render_sidebar()
    
    # ── Header ────────────────────────────────────────────
    col_title, col_status = st.columns([3, 1])
    with col_title:
        name = st.session_state.user_name
        st.markdown(
            f"# 💊 {'Smart Medicine Tracker' if not name else f'Hi, {name}!'}"
        )
        st.caption(f"Today: {datetime.now().strftime('%A, %B %d %Y — %H:%M')}")
    with col_status:
        if st.session_state.user_id:
            db = get_db()
            summary = db.get_today_summary(st.session_state.user_id)
            pct = summary["adherence_pct"]
            color = "normal" if pct >= 80 else "off"
            st.metric("Today's Adherence", f"{pct:.0f}%", delta=f"{summary['total_taken']} taken")
    
    st.markdown("---")
    
    # ── Main Tabs ─────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📷 Detection",
        "📊 Dashboard",
        "🔮 AI Prediction",
        "⚙️ Settings",
    ])
    
    with tab1:
        render_detection_tab()
    with tab2:
        render_dashboard_tab()
    with tab3:
        render_prediction_tab()
    with tab4:
        render_settings_tab()


if __name__ == "__main__":
    main()
