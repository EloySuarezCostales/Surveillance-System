import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
from datetime import datetime, date
import os
import plotly.express as px

from configROI import (
    CSV_PATH,
    OUTPUT_DIR,
    CLIPS_DIR
)


# CONFIGURATION
CSV_PATH   = "event_log.csv"
OUTPUT_DIR = "detected_events"

st.set_page_config(page_title="Surveillance System", layout="wide")


# STYLES
st.markdown("""
<style>
    /* General background */
    .stApp { background-color: #0e1117; color: white; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #1c2333;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px 20px;
    }

    /* Metric label and value */
    div[data-testid="metric-container"] label,
    div[data-testid="metric-container"] p,
    div[data-testid="metric-container"] span,
    div[data-testid="metric-container"] div {
        color: white !important;
    }

    /* Data table */
    div[data-testid="stDataFrame"] {
        border: 1px solid #2d3748;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Section title */
    .section-title {
        color: #63b3ed;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #2d3748;
    }

    /* Event type badges */
    .badge-detection {
        background: #2b4a6f;
        color: #90cdf4;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-saved {
        background: #276749;
        color: #9ae6b4;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    /* Force metric text to white */
    div[data-testid="metric-container"],
    div[data-testid="metric-container"] *,
    [data-testid="stMetric"],
    [data-testid="stMetric"] *,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    /* Refresh button - default state */
    div[data-testid="stButton"] button {
        background-color: #1c2333 !important;
        color: #ffffff !important;
        border: 1px solid #2d3748 !important;
        border-radius: 10px !important;
    }

    /* Download clip button - same style as refresh */
    div[data-testid="stDownloadButton"] button {
        background-color: #1c2333 !important;
        color: #ffffff !important;
        border: 1px solid #2d3748 !important;
        border-radius: 10px !important;
    }

    /* Text and icon inside download button */
    div[data-testid="stDownloadButton"] button * {
        color: #ffffff !important;
    }

    /* Download button hover */
    div[data-testid="stDownloadButton"] button:hover {
        background-color: #263248 !important;
        color: #ffffff !important;
        border-color: #63b3ed !important;
    }

    /* Text and icon inside button */
    div[data-testid="stButton"] button * {
        color: #ffffff !important;
    }

    /* Button hover */
    div[data-testid="stButton"] button:hover {
        background-color: #263248 !important;
        color: #ffffff !important;
        border-color: #63b3ed !important;
    }
</style>
""", unsafe_allow_html=True)


# HEADER
# Left column is 3x wider than right column
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# Surveillance System")
    st.markdown("Dashboard for detected events and saved captures")
with col_h2:
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.divider()


# DATA LOADING
@st.cache_data(ttl=15)
def load_data():
    if not Path(CSV_PATH).exists():
        return pd.DataFrame(columns=["timestamp", "type", "confidence", "persons_count", "image_path"])
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["persons_count"] = pd.to_numeric(df["persons_count"], errors="coerce")
    return df.sort_values("timestamp", ascending=False).reset_index(drop=True)

df = load_data()

if df.empty:
    st.info("⚠️ No events recorded yet. Run the surveillance system to get started.")
    st.stop()

# Filter by event type, dropping any malformed rows
df_detections = df[df["type"] == "detection"]
df_saved      = df[df["type"] == "saved_image"]


# SUMMARY METRICS
st.markdown('<p class="section-title">📊 General summary</p>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Total detections", len(df_detections))

with c2:
    st.metric("Saved images", len(df_saved))

with c3:
    if not df_saved.empty and df_saved["confidence"].notna().any():
        avg_conf = df_saved["confidence"].mean()
        st.metric("Average confidence", f"{avg_conf:.0%}")
    else:
        st.metric("Average confidence", "—")

with c4:
    if not df.empty and df["timestamp"].notna().any():
        last_event = df["timestamp"].max()
        elapsed    = datetime.now() - last_event
        mins       = int(elapsed.total_seconds() // 60)
        label      = f"{mins} min ago" if mins < 60 else f"{mins // 60} h ago"
        st.metric("Last detection", label)
    else:
        st.metric("Last detection", "—")

with c5:
    if len(df_detections) > 0:
        save_rate = len(df_saved) / len(df_detections)
        st.metric("Save rate", f"{save_rate:.2%}")
    else:
        st.metric("Save rate", "—")


# SIDEBAR – FILTERS
with st.sidebar:
    st.markdown("## 🔍 Filters")

    # Date range
    valid_dates = df["timestamp"].dropna()
    if not valid_dates.empty:
        date_min = valid_dates.min().date()
        date_max = valid_dates.max().date()
        date_range = st.date_input("Date range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max)
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            date_start, date_end = date_range
        else:
            date_start = date_end = date_range
    else:
        date_start = date_end = date.today()

    # Event type
    type_filter = st.multiselect("Event type",
        options=["detection", "saved_image"], default=["detection", "saved_image"],
        format_func=lambda x: "🔍 Detection" if x == "detection" else "📸 Saved image")

    # Minimum persons
    min_persons = st.slider("Minimum persons detected", 0, 10, 0)

    # Minimum confidence (saved images only)
    min_conf = st.slider("Minimum confidence (captures)", 0.0, 1.0, 0.0, step=0.05)

    st.divider()
    st.caption("Data refreshes every 15 s automatically or when you press 'Refresh'.")


# APPLY FILTERS
df_filtered = df.copy()

# Keep only rows within the selected date range
if df["timestamp"].notna().any():
    df_filtered = df_filtered[
        (df_filtered["timestamp"].dt.date >= date_start) &
        (df_filtered["timestamp"].dt.date <= date_end)]

# Keep only selected event types
if type_filter:
    df_filtered = df_filtered[df_filtered["type"].isin(type_filter)]

# Keep only events with at least the minimum number of persons
df_filtered = df_filtered[df_filtered["persons_count"].fillna(0) >= min_persons]

# Apply confidence filter to saved images only
conf_mask = ((df_filtered["type"] != "saved_image") |
    (df_filtered["confidence"].fillna(0) >= min_conf))
df_filtered = df_filtered[conf_mask]


# EVENT LOG TABLE
st.markdown('<p class="section-title">📋 Event history</p>', unsafe_allow_html=True)
st.caption(f"{len(df_filtered)} events match the current filters")

if df_filtered.empty:
    st.warning("No events match the selected filters.")
else:
    table = df_filtered[["timestamp", "type", "confidence", "persons_count"]].copy()
    table.columns = ["Date & time", "Type", "Confidence", "Persons"]
    table["Type"] = table["Type"].map({
        "detection":  "🔍 Detection",
        "saved_image": "📸 Saved image",
    })
    table["Confidence"] = table["Confidence"].apply(
        lambda x: f"{x:.0%}" if pd.notna(x) else "—")
    table["Persons"] = table["Persons"].apply(
        lambda x: int(x) if pd.notna(x) else "—")
    table["Date & time"] = table["Date & time"].dt.strftime("%Y-%m-%d  %H:%M:%S")

    st.dataframe(table, use_container_width=True, hide_index=True)


# SAVED CAPTURES GALLERY
st.markdown('<p class="section-title">🖼️ Saved captures</p>', unsafe_allow_html=True)

images_dir = Path(OUTPUT_DIR)
jpg_files = sorted(images_dir.glob("*.jpg"), reverse=True) if images_dir.exists() else []

# Filter image files by selected date range
def date_from_filename(p: Path):
    try:
        return datetime.strptime(p.stem.split("_", 1)[1], "%Y-%m-%d_%H-%M-%S").date()
    except Exception:
        return None

filtered_files = []
for f in jpg_files:
    fd = date_from_filename(f)
    if fd is None or (date_start <= fd <= date_end):
        filtered_files.append(f)

if not filtered_files:
    st.info("No captures found for the selected date range.")
else:
    st.caption(f"{len(filtered_files)} images found")

    filenames = [f.name for f in filtered_files]
    selected = st.selectbox("Select a capture to enlarge", filenames)

    # Large preview on the left, thumbnail grid on the right
    col_big, col_grid = st.columns([2, 3])

    with col_big:
        selected_path = images_dir / selected
        if selected_path.exists():
            img = Image.open(selected_path)
            st.image(img, caption=selected, use_container_width=True)

            # Show metadata for the selected image
            row = df_saved[df_saved["image_path"].str.endswith(selected, na=False)]
            if not row.empty:
                r = row.iloc[0]
                m1, m2 = st.columns(2)
                # notna guard for missing values
                m1.metric("Confidence", f"{float(r['confidence']):.0%}" if pd.notna(r["confidence"]) else "—")
                m2.metric("Persons", int(r["persons_count"]) if pd.notna(r["persons_count"]) else "—")

    with col_grid:
        st.markdown("**All captures**")
        cols = st.columns(3)
        for i, file in enumerate(filtered_files[:12]):   # Max 12 in grid
            # Cycle through 3 columns to build rows
            with cols[i % 3]:
                thumb = Image.open(file)
                st.image(thumb, caption=file.name, use_container_width=True)

    if len(filtered_files) > 12:
        st.caption(f"Showing the 12 most recent of {len(filtered_files)} total captures.")


# EVENT CLIPS
st.markdown('<p class="section-title">🎬 Event clips</p>', unsafe_allow_html=True)

clips_dir = Path(CLIPS_DIR)

video_files = sorted(
    list(clips_dir.glob("*.mp4")) + list(clips_dir.glob("*.avi")),
    reverse=True
) if clips_dir.exists() else []

if not video_files:
    st.info("No clips saved yet.")
else:
    st.caption(f"{len(video_files)} clips found")

    clip_names = [clip.name for clip in video_files]

    selected_clip = st.selectbox("Select a clip to view", clip_names)

    clip_path = clips_dir / selected_clip

    if clip_path.exists():
        st.write(f"Selected clip: `{selected_clip}`")

        st.warning("🎬 Preview not available.\n\n")

        with open(clip_path, "rb") as f:
            st.download_button(
                label="⬇️ Download clip",
                data=f,
                file_name=selected_clip,
                mime="video/x-msvideo" if selected_clip.endswith(".avi") else "video/mp4"
            )


# ACTIVITY STATISTICS
st.markdown('<p class="section-title">📈 Activity statistics</p>', unsafe_allow_html=True)

chart_period = st.selectbox("Analysis period", ["Day", "Week", "Month"])

df_chart = df_detections.copy()

if not df_chart.empty:

    today = datetime.now().date()

    if chart_period == "Day":
        period_start = today

    elif chart_period == "Week":
        period_start = today - pd.Timedelta(days=6)

    else:  # Month
        period_start = today - pd.Timedelta(days=29)

    df_chart = df_chart[df_chart["timestamp"].dt.date >= period_start]

    if not df_chart.empty:

        df_chart["hour"] = df_chart["timestamp"].dt.hour

        detections_by_hour = (df_chart
            .groupby("hour")
            .size()
            .reindex(range(24), fill_value=0))

        df_plot = detections_by_hour.reset_index()
        df_plot.columns = ["Hour", "Detections"]

        # Treat each hour as a category for correct ordering
        df_plot["Hour"] = df_plot["Hour"].astype(str)

        fig = px.bar(df_plot,
            x="Hour",
            y="Detections",
            title=f"Detections per hour ({chart_period})")

        fig.update_layout(
            xaxis_title="Hour of day",
            yaxis_title="Number of detections",
            xaxis_tickangle=0)

        fig.update_xaxes(
            categoryorder="array",
            categoryarray=[str(i) for i in range(24)])

        st.plotly_chart(fig, use_container_width=True)

        # Peak hour summary
        peak_hour = detections_by_hour.idxmax()
        peak_total = detections_by_hour.max()

        st.info(f"🕐 Peak activity hour: "
            f"{peak_hour}:00 ({peak_total} detections)")

    else:
        st.info("No detections found for the selected period.")

else:
    st.info("Not enough data to generate statistics.")