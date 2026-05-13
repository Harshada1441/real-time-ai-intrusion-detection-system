import sys
import time
from collections import Counter, deque
from pathlib import Path

import pandas as pd
import streamlit as st
from scapy.all import get_if_list

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sniffer import PacketSniffer

st.set_page_config(page_title="Real-Time AI IDS", layout="wide")


def init_state() -> None:
    if "sniffer" not in st.session_state:
        st.session_state.sniffer = None
    if "alerts" not in st.session_state:
        st.session_state.alerts = deque(maxlen=500)


def start_sniffer(model_dir: str, interface: str, window: int, threshold: float) -> None:
    if st.session_state.sniffer and st.session_state.sniffer.is_running:
        return
    st.session_state.sniffer = PacketSniffer(
        model_dir=model_dir,
        interface=interface,
        window_seconds=window,
        prediction_threshold=threshold,
    )
    st.session_state.sniffer.start()


def stop_sniffer() -> None:
    if st.session_state.sniffer:
        st.session_state.sniffer.stop()
        st.session_state.sniffer = None


def drain_alerts() -> None:
    if not st.session_state.sniffer:
        return
    alerts = st.session_state.sniffer.drain_alerts(limit=200)
    for alert in alerts:
        st.session_state.alerts.append(alert)


def render_dashboard() -> None:
    st.title("Real-Time AI Intrusion Detection System")

    if not st.session_state.alerts:
        st.info("No alerts yet")
        return

    df = pd.DataFrame(list(st.session_state.alerts))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    counts = Counter(df["prediction"].tolist())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Alerts", sum(counts.values()))
    col2.metric("Top Class", counts.most_common(1)[0][0])
    col3.metric("Unique Classes", len(counts))
    col4.metric("Latest Score", float(df.iloc[-1]["score"]))

    st.subheader("Recent Alerts")
    st.dataframe(df.tail(50), use_container_width=True)

    st.subheader("Alert Volume")
    timeline = df.set_index("timestamp").resample("10s").size()
    st.line_chart(timeline, use_container_width=True)


init_state()

interfaces = get_if_list()
if not interfaces:
    interfaces = [""]

with st.sidebar:
    st.header("Controls")
    model_dir = st.text_input("Model directory", value=str(ROOT / "models"))
    interface = st.selectbox("Interface", options=interfaces, index=0)
    window = st.number_input("Flow window (seconds)", min_value=1, max_value=60, value=5)
    threshold = st.slider("Prediction threshold", 0.0, 1.0, 0.6, 0.05)
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_interval = st.number_input("Refresh interval (seconds)", min_value=1, max_value=10, value=1)

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("Start", type="primary"):
            try:
                start_sniffer(model_dir, interface, window, threshold)
            except Exception as exc:
                st.error(f"Failed to start sniffer: {exc}")
    with col_stop:
        if st.button("Stop"):
            stop_sniffer()

if st.session_state.sniffer and st.session_state.sniffer.is_running:
    st.success("Sniffer running")
else:
    st.warning("Sniffer stopped")

try:
    drain_alerts()
except Exception as exc:
    st.error(f"Error draining alerts: {exc}")

render_dashboard()

if auto_refresh:
    time.sleep(int(refresh_interval))
    st.rerun()
