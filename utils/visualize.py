"""
=============================================================
utils/visualize.py — Chart & Visualization Utilities
=============================================================
Generates Plotly charts for the Streamlit dashboard:
  - Adherence calendar heatmap
  - Daily adherence bar chart
  - Risk trend line chart
  - Streak indicator
  - Dose history pie chart
=============================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── Color Palette ─────────────────────────────────────────
COLORS = {
    "taken":     "#4CAF50",    # Green
    "missed":    "#F44336",    # Red
    "primary":   "#2196F3",    # Blue
    "warning":   "#FF9800",    # Orange
    "bg":        "#0E1117",    # Dark background (Streamlit default)
    "card":      "#1E2130",    # Card background
    "text":      "#FAFAFA",    # Light text
    "grid":      "#2D3148",    # Grid lines
}

CHART_THEME = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(color=COLORS["text"], family="Inter, sans-serif"),
    margin        = dict(l=20, r=20, t=40, b=20),
)


# ── Adherence Calendar Heatmap ────────────────────────────

def adherence_calendar(df: pd.DataFrame, title: str = "Adherence Calendar") -> go.Figure:
    """
    GitHub-style calendar heatmap showing daily adherence.
    
    Args:
        df: DataFrame with 'timestamp' (datetime) and 'taken' (int) columns
    """
    if df.empty:
        return _empty_chart(title)
    
    # Daily aggregation
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily = df.groupby("date")["taken"].mean().reset_index()
    daily.columns = ["date", "adherence"]
    daily["date"] = pd.to_datetime(daily["date"])
    
    # Fill missing dates
    date_range = pd.date_range(daily["date"].min(), daily["date"].max())
    daily = daily.set_index("date").reindex(date_range, fill_value=np.nan).reset_index()
    daily.columns = ["date", "adherence"]
    
    daily["week"] = daily["date"].dt.isocalendar().week
    daily["weekday"] = daily["date"].dt.weekday
    daily["date_str"] = daily["date"].dt.strftime("%b %d, %Y")

    fig = go.Figure(go.Heatmap(
        z            = daily["adherence"],
        x            = daily["week"],
        y            = daily["weekday"],
        text         = daily["date_str"],
        hovertemplate= "%{text}<br>Adherence: %{z:.0%}<extra></extra>",
        colorscale   = [[0, COLORS["missed"]], [0.5, COLORS["warning"]], [1, COLORS["taken"]]],
        showscale    = True,
        colorbar     = dict(title="Adherence", tickformat=".0%"),
        zmin=0, zmax=1,
    ))

    fig.update_layout(
        title     = title,
        yaxis     = dict(
            tickvals = [0, 1, 2, 3, 4, 5, 6],
            ticktext = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        ),
        xaxis     = dict(title="Week"),
        **CHART_THEME,
    )
    return fig


# ── Daily Adherence Bar Chart ─────────────────────────────

def daily_adherence_bar(df: pd.DataFrame, days: int = 30) -> go.Figure:
    """
    Bar chart of daily taken vs missed doses over the last N days.
    """
    if df.empty:
        return _empty_chart("Daily Adherence")
    
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    
    # Aggregate
    daily = df.groupby(["date", "taken"]).size().unstack(fill_value=0)
    if 1 not in daily.columns:
        daily[1] = 0
    if 0 not in daily.columns:
        daily[0] = 0
    
    # Last N days
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days - 1)
    full_range = pd.date_range(start_date, end_date).date
    daily = daily.reindex(full_range, fill_value=0)
    
    dates = [str(d) for d in daily.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name   = "✅ Taken",
        x      = dates,
        y      = daily[1],
        marker = dict(color=COLORS["taken"], opacity=0.85),
    ))
    fig.add_trace(go.Bar(
        name   = "❌ Missed",
        x      = dates,
        y      = daily[0],
        marker = dict(color=COLORS["missed"], opacity=0.85),
    ))
    
    fig.update_layout(
        barmode   = "stack",
        title     = f"Daily Adherence (Last {days} Days)",
        xaxis     = dict(title="Date", tickangle=-45),
        yaxis     = dict(title="Doses"),
        legend    = dict(orientation="h", y=1.1),
        **CHART_THEME,
    )
    return fig


# ── Risk Trend Line Chart ─────────────────────────────────

def risk_trend_chart(risk_history: List[Dict]) -> go.Figure:
    """
    Line chart showing how miss probability has changed over time.
    
    Args:
        risk_history: List of {'date': str, 'miss_probability': float}
    """
    if not risk_history:
        return _empty_chart("Risk Trend")
    
    df = pd.DataFrame(risk_history)
    df["date"] = pd.to_datetime(df["date"])
    
    fig = go.Figure()
    
    # Risk zones
    fig.add_hrect(y0=0.0,  y1=0.3,  fillcolor=COLORS["taken"],  opacity=0.07, layer="below", line_width=0)
    fig.add_hrect(y0=0.3,  y1=0.6,  fillcolor=COLORS["warning"], opacity=0.07, layer="below", line_width=0)
    fig.add_hrect(y0=0.6,  y1=1.0,  fillcolor=COLORS["missed"],  opacity=0.07, layer="below", line_width=0)
    
    # Risk line
    fig.add_trace(go.Scatter(
        x    = df["date"],
        y    = df["miss_probability"],
        mode = "lines+markers",
        name = "Miss Probability",
        line = dict(color=COLORS["primary"], width=2.5),
        marker= dict(size=6),
        hovertemplate="Date: %{x|%b %d}<br>Miss Risk: %{y:.1%}<extra></extra>",
    ))
    
    # Threshold lines
    for y, label, color in [
        (0.30, "Low/Medium", COLORS["warning"]),
        (0.60, "Medium/High", COLORS["missed"]),
    ]:
        fig.add_hline(
            y=y, line_dash="dot", line_color=color, opacity=0.5,
            annotation_text=label, annotation_position="right",
        )
    
    fig.update_layout(
        title  = "Miss Probability Trend",
        xaxis  = dict(title="Date"),
        yaxis  = dict(title="Miss Probability", tickformat=".0%", range=[0, 1]),
        **CHART_THEME,
    )
    return fig


# ── Streak Gauge ──────────────────────────────────────────

def streak_gauge(streak: int, max_streak: int = 30) -> go.Figure:
    """Gauge chart showing current adherence streak."""
    pct = min(streak / max_streak, 1.0)
    color = COLORS["taken"] if pct > 0.5 else COLORS["warning"] if pct > 0.2 else COLORS["missed"]
    
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = streak,
        title = dict(text="🔥 Current Streak (days)", font=dict(size=14)),
        gauge = dict(
            axis       = dict(range=[0, max_streak], tickcolor="white"),
            bar        = dict(color=color),
            bgcolor    = COLORS["card"],
            bordercolor= COLORS["grid"],
            steps      = [
                dict(range=[0, max_streak * 0.3],  color=COLORS["card"]),
                dict(range=[max_streak * 0.3, max_streak * 0.7], color="#1E3040"),
                dict(range=[max_streak * 0.7, max_streak], color="#1E4030"),
            ],
            threshold  = dict(
                line  = dict(color=COLORS["taken"], width=3),
                thickness=0.75, value=max_streak * 0.8
            ),
        ),
        number= dict(suffix=" days", font=dict(color=color)),
    ))
    
    fig.update_layout(height=200, **CHART_THEME)
    return fig


# ── Weekly Summary Donut ──────────────────────────────────

def weekly_donut(taken: int, missed: int) -> go.Figure:
    """Donut chart showing weekly taken vs missed split."""
    total = taken + missed
    if total == 0:
        taken, missed = 1, 0   # Default display

    fig = go.Figure(go.Pie(
        values   = [taken, missed],
        labels   = ["Taken", "Missed"],
        hole     = 0.60,
        marker   = dict(colors=[COLORS["taken"], COLORS["missed"]]),
        textinfo = "percent",
        hovertemplate = "%{label}: %{value} doses (%{percent})<extra></extra>",
    ))
    
    pct = int(taken / total * 100) if total > 0 else 0
    fig.add_annotation(
        text     = f"<b>{pct}%</b>",
        x=0.5, y=0.5, showarrow=False,
        font     = dict(size=24, color=COLORS["text"]),
    )
    
    fig.update_layout(
        title  = "This Week",
        legend = dict(orientation="h", y=-0.1),
        height = 250,
        **CHART_THEME,
    )
    return fig


# ── Risk Probability Bar ──────────────────────────────────

def miss_probability_bar(miss_prob: float) -> go.Figure:
    """Horizontal bar showing miss probability with risk zones."""
    take_prob = 1 - miss_prob
    
    if miss_prob < 0.3:
        color, label = COLORS["taken"], "Low Risk"
    elif miss_prob < 0.6:
        color, label = COLORS["warning"], "Medium Risk"
    else:
        color, label = COLORS["missed"], "High Risk"
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x            = [miss_prob],
        y            = ["Miss Risk"],
        orientation  = "h",
        marker       = dict(color=color),
        text         = [f"{miss_prob:.1%}  ({label})"],
        textposition = "inside",
        hovertemplate= f"Miss Probability: {miss_prob:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x           = [take_prob],
        y           = ["Miss Risk"],
        orientation = "h",
        marker      = dict(color=COLORS["grid"], opacity=0.4),
        showlegend  = False,
        hoverinfo   = "skip",
    ))
    
    fig.update_layout(
        barmode = "stack",
        xaxis   = dict(range=[0, 1], tickformat=".0%", showgrid=False),
        yaxis   = dict(showticklabels=False),
        height  = 90,
        margin  = dict(l=5, r=5, t=5, b=5),
        **CHART_THEME,
    )
    return fig


# ── Multi-Medicine Comparison ─────────────────────────────

def multi_medicine_adherence(medicine_stats: List[Dict]) -> go.Figure:
    """
    Horizontal bar chart comparing adherence % across medicines.
    
    Args:
        medicine_stats: [{'name': 'Metformin', 'adherence_pct': 85.0, 'streak': 7}, ...]
    """
    if not medicine_stats:
        return _empty_chart("Medicine Adherence Comparison")
    
    df = pd.DataFrame(medicine_stats).sort_values("adherence_pct")
    colors = [
        COLORS["taken"]   if p >= 80 else
        COLORS["warning"] if p >= 60 else
        COLORS["missed"]
        for p in df["adherence_pct"]
    ]
    
    fig = go.Figure(go.Bar(
        x           = df["adherence_pct"],
        y           = df["name"],
        orientation = "h",
        marker      = dict(color=colors),
        text        = [f"{p:.0f}%" for p in df["adherence_pct"]],
        textposition= "outside",
    ))
    
    fig.add_vline(x=80, line_dash="dot", line_color=COLORS["warning"],
                  annotation_text="80% target")
    
    fig.update_layout(
        title  = "Medicine Adherence Comparison",
        xaxis  = dict(title="Adherence %", range=[0, 110]),
        yaxis  = dict(title=""),
        height = max(200, len(df) * 50 + 100),
        **CHART_THEME,
    )
    return fig


# ── Helper ────────────────────────────────────────────────

def _empty_chart(title: str) -> go.Figure:
    """Return a styled empty chart placeholder."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available yet",
        x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14, color="#888"),
    )
    fig.update_layout(title=title, height=200, **CHART_THEME)
    return fig
