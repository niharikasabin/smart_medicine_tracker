"""
=============================================================
app/reminder.py — Reminder & Notification System
=============================================================
Provides:
  1. Desktop notifications (via plyer — cross-platform)
  2. Email reminders (via smtplib / SMTP)
  3. Reminder scheduling (check upcoming doses)
  4. Risk-based alert triggers
=============================================================
"""

import os
import smtplib
import json
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from pathlib import Path


# ── Desktop Notification ──────────────────────────────────

def send_desktop_notification(title: str, message: str, timeout: int = 10):
    """
    Send a desktop push notification.
    Uses plyer (cross-platform: Windows, macOS, Linux).
    Falls back to print if plyer not available.
    """
    try:
        from plyer import notification
        notification.notify(
            title     = title,
            message   = message,
            app_name  = "💊 Medicine Tracker",
            timeout   = timeout,
        )
        print(f"🔔 Desktop notification sent: {title}")
        return True
    except Exception as e:
        # Graceful fallback
        print(f"\n{'='*50}")
        print(f"🔔 NOTIFICATION: {title}")
        print(f"   {message}")
        print(f"{'='*50}\n")
        return False


# ── Email Reminder ────────────────────────────────────────

class EmailReminder:
    """
    Send email reminders via Gmail SMTP (or any SMTP server).
    
    Setup for Gmail:
        1. Enable 2FA on your Google account
        2. Go to Google Account → Security → App Passwords
        3. Generate app password for "Mail"
        4. Use that as SMTP_PASSWORD
    
    Usage:
        reminder = EmailReminder(
            smtp_email="your@gmail.com",
            smtp_password="your_app_password"
        )
        reminder.send_dose_reminder("patient@email.com", "Alice", "Metformin 500mg")
    """

    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT   = 587

    def __init__(
        self,
        smtp_email:    str = None,
        smtp_password: str = None,
    ):
        # Load from environment variables if not provided
        self.smtp_email    = smtp_email    or os.getenv("SMTP_EMAIL", "")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD", "")
        self.enabled       = bool(self.smtp_email and self.smtp_password)
        
        if not self.enabled:
            print("ℹ️  Email reminders disabled (no SMTP credentials)")
            print("   Set SMTP_EMAIL and SMTP_PASSWORD environment variables to enable")

    def send(self, to: str, subject: str, html_body: str) -> bool:
        """Send an HTML email."""
        if not self.enabled:
            self._simulate_email(to, subject, html_body)
            return False
        
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = f"💊 Medicine Tracker <{self.smtp_email}>"
            msg["To"]      = to

            part = MIMEText(html_body, "html")
            msg.attach(part)

            with smtplib.SMTP(self.SMTP_SERVER, self.SMTP_PORT) as server:
                server.ehlo()
                server.starttls()
                server.login(self.smtp_email, self.smtp_password)
                server.sendmail(self.smtp_email, to, msg.as_string())
            
            print(f"✅ Email sent to {to}: {subject}")
            return True

        except Exception as e:
            print(f"❌ Email failed: {e}")
            return False

    def _simulate_email(self, to: str, subject: str, body: str):
        """Print email content when SMTP is not configured."""
        print(f"\n{'─'*55}")
        print(f"📧 EMAIL SIMULATION (configure SMTP to send real emails)")
        print(f"   To:      {to}")
        print(f"   Subject: {subject}")
        print(f"   Body:    [HTML email — see template below]")
        print(f"{'─'*55}\n")

    # ── Email Templates ───────────────────────────────────

    def send_dose_reminder(
        self, to: str, name: str, medicine: str, scheduled_time: str = None
    ) -> bool:
        """Send a dose reminder email."""
        time_str = scheduled_time or datetime.now().strftime("%I:%M %p")
        subject  = f"💊 Time to take your {medicine}"
        body     = self._template_reminder(name, medicine, time_str)
        return self.send(to, subject, body)

    def send_high_risk_alert(
        self, to: str, name: str, medicine: str, miss_probability: float
    ) -> bool:
        """Send high-risk missed dose alert."""
        subject = f"⚠️ High miss risk detected — {medicine}"
        body    = self._template_risk_alert(name, medicine, miss_probability)
        return self.send(to, subject, body)

    def send_weekly_report(
        self, to: str, name: str, adherence_pct: float, streak: int
    ) -> bool:
        """Send weekly adherence summary email."""
        subject = f"📊 Your Weekly Medicine Adherence Report"
        body    = self._template_weekly_report(name, adherence_pct, streak)
        return self.send(to, subject, body)

    # ── HTML Templates ────────────────────────────────────

    def _template_reminder(self, name: str, medicine: str, time: str) -> str:
        return f"""
        <html><body style="font-family:Arial,sans-serif; background:#f5f5f5; padding:20px">
        <div style="max-width:500px; margin:auto; background:#fff; border-radius:12px;
                    padding:30px; box-shadow:0 2px 12px rgba(0,0,0,0.1)">
            <h2 style="color:#2196F3">💊 Medicine Reminder</h2>
            <p style="font-size:16px">Hi <strong>{name}</strong>,</p>
            <p>It's time to take your <strong style="color:#4CAF50">{medicine}</strong>.</p>
            <p style="color:#888">Scheduled time: {time}</p>
            <div style="background:#E3F2FD; border-radius:8px; padding:15px; margin:20px 0">
                <strong>💡 Tip:</strong> Take your medicine with a full glass of water.
            </div>
            <p style="color:#888; font-size:12px">Smart Medicine Adherence Tracker</p>
        </div></body></html>"""

    def _template_risk_alert(self, name: str, medicine: str, prob: float) -> str:
        pct = f"{prob:.0%}"
        return f"""
        <html><body style="font-family:Arial,sans-serif; background:#f5f5f5; padding:20px">
        <div style="max-width:500px; margin:auto; background:#fff; border-radius:12px;
                    padding:30px; box-shadow:0 2px 12px rgba(255,0,0,0.1)">
            <h2 style="color:#F44336">⚠️ High Miss Risk Detected</h2>
            <p>Hi <strong>{name}</strong>,</p>
            <p>Our AI predicts a <strong style="color:#F44336">{pct} probability</strong>
               that you may miss your next dose of <strong>{medicine}</strong>.</p>
            <p>Please make sure to take your medicine as scheduled!</p>
            <p style="color:#888; font-size:12px">Smart Medicine Adherence Tracker</p>
        </div></body></html>"""

    def _template_weekly_report(self, name: str, pct: float, streak: int) -> str:
        color = "#4CAF50" if pct >= 80 else "#FF9800" if pct >= 60 else "#F44336"
        grade = "Excellent" if pct >= 90 else "Good" if pct >= 75 else "Needs Improvement"
        return f"""
        <html><body style="font-family:Arial,sans-serif; background:#f5f5f5; padding:20px">
        <div style="max-width:500px; margin:auto; background:#fff; border-radius:12px;
                    padding:30px; box-shadow:0 2px 12px rgba(0,0,0,0.1)">
            <h2 style="color:#2196F3">📊 Weekly Report</h2>
            <p>Hi <strong>{name}</strong>, here's your weekly summary:</p>
            <div style="text-align:center; padding:20px">
                <div style="font-size:48px; font-weight:bold; color:{color}">{pct:.0f}%</div>
                <div style="color:{color}; font-size:18px">{grade}</div>
            </div>
            <p>🔥 Current streak: <strong>{streak} days</strong></p>
            <p style="color:#888; font-size:12px">Smart Medicine Adherence Tracker</p>
        </div></body></html>"""


# ── Alert Manager ─────────────────────────────────────────

class AlertManager:
    """
    Central manager for triggering reminders and alerts.
    Combines desktop + email notifications.
    """

    RISK_THRESHOLD = 0.60     # Trigger alert if miss probability > 60%

    def __init__(self, email_sender: EmailReminder = None):
        self.email = email_sender or EmailReminder()

    def check_and_alert(
        self,
        user_name:       str,
        user_email:      str,
        medicine_name:   str,
        miss_probability: float,
    ) -> Dict:
        """
        Check risk and trigger appropriate notifications.
        
        Returns:
            {'alerted': bool, 'method': str, 'risk_level': str}
        """
        result = {
            "alerted":    False,
            "method":     [],
            "risk_level": "Low",
        }

        if miss_probability >= self.RISK_THRESHOLD:
            result["risk_level"] = "High" if miss_probability >= 0.75 else "Medium"

            # Desktop notification
            send_desktop_notification(
                title   = f"⚠️ Medicine Alert — {medicine_name}",
                message = f"Risk of missing dose: {miss_probability:.0%}. Take it now!",
            )
            result["method"].append("desktop")

            # Email alert
            if user_email:
                self.email.send_high_risk_alert(
                    user_email, user_name, medicine_name, miss_probability
                )
                result["method"].append("email")

            result["alerted"] = True
        
        return result

    def send_scheduled_reminder(
        self, user_name: str, user_email: str, medicine_name: str
    ):
        """Send scheduled dose reminder (called by scheduler)."""
        send_desktop_notification(
            title   = f"💊 Time for {medicine_name}",
            message = f"Hi {user_name}! Don't forget your {medicine_name}.",
        )
        if user_email:
            self.email.send_dose_reminder(user_email, user_name, medicine_name)


# ── Usage example ─────────────────────────────────────────
if __name__ == "__main__":
    alert = AlertManager()

    # Simulate high-risk alert
    result = alert.check_and_alert(
        user_name        = "Alice",
        user_email       = "alice@example.com",
        medicine_name    = "Metformin 500mg",
        miss_probability = 0.78,
    )
    print(f"Alert result: {result}")
