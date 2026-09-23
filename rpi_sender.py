"""
rpi_sender.py — Raspberry Pi Camera Streaming Client
=====================================================
Captures frames from the RPi camera and streams them to the Jetson Orin
over TCP for real-time YOLO inference benchmarking.

Run on RPi:  streamlit run rpi_sender.py

Dependencies (RPi):
    pip install streamlit opencv-python numpy
"""

import socket
import struct
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Import shared protocol (must be in same directory or on PYTHONPATH)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from frame_protocol import (
    encode_frame, send_packet, handle_ping,
    recv_exact, DEFAULT_PORT, DEFAULT_QUALITY,
    PING_SIZE, PING_FMT, recv_detections_nonblocking
)
import struct as _struct

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RPi Camera Sender",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #0d1117; }
[data-testid="stSidebar"] { background-color: #161b22; border-right:1px solid #30363d; }

.stat-box {
    background:#161b22; border-radius:8px;
    padding:14px; margin:5px 0;
    border-left:4px solid #58a6ff;
    display:flex; justify-content:space-between; align-items:center;
}
.stat-box .k { color:#8b949e; font-size:0.85em; }
.stat-box .v { color:#c9d1d9; font-size:1.3em; font-weight:700; }

.status-ok  { color:#3fb950; font-weight:700; }
.status-err { color:#f78166; font-weight:700; }
.status-wait{ color:#e3b341; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [
    ("running",     False),
    ("sock",        None),
    ("stats",       {}),
    ("frame_count", 0),
    ("errors",      0),
    ("bytes_sent",  0),
    ("start_time",  None),
    ("last_fps",    0.0),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar settings ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📷 RPi Sender")
    st.divider()

    st.markdown("### 🌐 Connection")
    jetson_ip  = st.text_input("Jetson IP Address", value="192.168.1.100",
                               help="IP of the Jetson Orin on the local network")
    port       = st.number_input("Port", value=DEFAULT_PORT, min_value=1024, max_value=65535)

    st.divider()

    st.markdown("### 📹 Camera")
    cam_idx    = st.selectbox("Camera Device", [0, 1, 2], index=0)
    resolution = st.selectbox("Resolution", ["1280×720", "640×480", "1920×1080"], index=0)
    fps_limit  = st.slider("Max FPS", 5, 60, 30, 5,
                           help="Frame rate cap on the sender side")

    st.divider()

    st.markdown("### 🗜️ Encoding")
    jpeg_quality = st.slider("JPEG Quality", 30, 100, DEFAULT_QUALITY, 5,
                             help="Higher = better quality, larger packet size")

    st.divider()

    col_a, col_b = st.columns(2)
    start_btn = col_a.button("▶ Connect & Stream", type="primary", use_container_width=True)
    stop_btn  = col_b.button("⏹ Stop",              use_container_width=True)


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("# 📡 Camera → Jetson Live Stream")

col_feed, col_stats = st.columns([2, 1])

with col_feed:
    st.markdown("#### 🎥 Preview")
    feed_ph = st.empty()

with col_stats:
    st.markdown("#### 📊 Streaming Stats")
    stats_ph = st.empty()

    st.markdown("#### 🌐 Network")
    net_ph   = st.empty()

    st.markdown("#### ℹ️ Session")
    sess_ph  = st.empty()

status_ph = st.empty()


# ── Helpers ────────────────────────────────────────────────────────────────────
def parse_resolution(s: str) -> tuple[int, int]:
    w, h = s.replace("×", "x").split("x")
    return int(w), int(h)


def render_stats(stats: dict):
    if not stats:
        return '<p style="color:#8b949e;">Waiting…</p>'

    def row(k, v):
        return (
            f'<div class="stat-box">'
            f'<span class="k">{k}</span>'
            f'<span class="v">{v}</span>'
            f'</div>'
        )

    html  = row("FPS",           f"{stats.get('fps', 0.0):.1f}")
    html += row("Frames sent",   f"{stats.get('frames', 0):,}")
    html += row("Bandwidth",     f"{stats.get('bw_mbps', 0.0):.2f} Mbps")
    html += row("Avg pkt size",  f"{stats.get('avg_pkt_kb', 0.0):.1f} KB")
    html += row("Encode time",   f"{stats.get('avg_enc_ms', 0.0):.1f} ms")
    html += row("Errors",        f"{stats.get('errors', 0)}")
    return html


def render_net(jetson_ip, port, quality):
    return f"""
    <div style="background:#161b22;border-radius:8px;padding:12px;">
      <div style="display:flex;justify-content:space-between;padding:4px 0;
                  border-bottom:1px solid #21262d;font-size:0.85em;">
        <span style="color:#8b949e;">Destination</span>
        <span style="color:#c9d1d9;font-weight:600;">{jetson_ip}:{port}</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:4px 0;
                  border-bottom:1px solid #21262d;font-size:0.85em;">
        <span style="color:#8b949e;">Protocol</span>
        <span style="color:#c9d1d9;font-weight:600;">TCP (reliable)</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:4px 0;
                  font-size:0.85em;">
        <span style="color:#8b949e;">JPEG Quality</span>
        <span style="color:#c9d1d9;font-weight:600;">{quality}%</span>
      </div>
    </div>"""


# ── Control logic ──────────────────────────────────────────────────────────────
if stop_btn:
    st.session_state.running = False
    if st.session_state.sock:
        try:
            st.session_state.sock.close()
        except Exception:
            pass
        st.session_state.sock = None


if start_btn:
    st.session_state.frame_count = 0
    st.session_state.errors      = 0
    st.session_state.bytes_sent  = 0
    st.session_state.start_time  = None

    # Connect to Jetson
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((jetson_ip, int(port)))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        st.session_state.sock = sock

        # Handshake: respond to Jetson's RTT pings and wait for START token
        sock.settimeout(5.0)
        while True:
            tag = sock.recv(4)
            if tag == b"PING":
                rest = recv_exact(sock, PING_SIZE - 4)
                handle_ping(b"PING" + rest, sock)
            elif tag == b"STAR":
                sock.recv(1)  # Read remaining 'T' of 'START'
                break
            elif not tag:
                break
        sock.settimeout(None)
        st.session_state.running = True
        status_ph.success(f"✅ Handshake synchronized! Streaming to {jetson_ip}:{port}")
    except Exception as e:
        status_ph.error(f"❌ Connection failed: {e}")
        st.stop()


# ── Streaming loop ─────────────────────────────────────────────────────────────
if st.session_state.running and st.session_state.sock:

    res_w, res_h = parse_resolution(resolution)
    cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_V4L2)
    # Crucial: enable hardware MJPEG on the webcam to unlock full 30 FPS at 720p/1080p
    # (Without this, USB 2.0 uncompressed YUYV format is hardware-throttled to 10 FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  res_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)
    cap.set(cv2.CAP_PROP_FPS, int(fps_limit))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        status_ph.error(f"❌ Could not open camera {cam_idx}")
        st.stop()

    sock            = st.session_state.sock
    frame_interval  = 1.0 / fps_limit
    t_prev          = time.perf_counter()
    enc_times:list  = []
    pkt_sizes:list  = []
    fps_smooth      = 0.0
    t_session       = time.perf_counter()
    latest_detections: list = []

    status_ph.markdown(
        '<span class="status-ok">🟢 Streaming…</span>', unsafe_allow_html=True
    )

    while st.session_state.running:

        # Frame rate cap
        elapsed = time.perf_counter() - t_prev
        if elapsed < frame_interval:
            time.sleep(max(0, frame_interval - elapsed - 0.001))

        ret, frame = cap.read()
        if not ret:
            st.session_state.errors += 1
            continue

        # ── Handle any incoming messages (PING or DETS) ──
        sock.setblocking(False)
        try:
            while True:
                tag = sock.recv(4, socket.MSG_PEEK)
                if len(tag) < 4:
                    break
                if tag == b"PING":
                    sock.setblocking(True)
                    data = recv_exact(sock, PING_SIZE)
                    handle_ping(data, sock)
                    sock.setblocking(False)
                elif tag == b"DETS":
                    sock.setblocking(True)
                    res = recv_detections_nonblocking(sock)
                    if res:
                        _, latest_detections = res
                    sock.setblocking(False)
                else:
                    # Not enough data or unknown tag
                    break
        except BlockingIOError:
            pass
        except Exception:
            st.session_state.errors += 1
        sock.setblocking(True)

        # ── Encode + send ─────────────────────────
        t_enc_0 = time.perf_counter()
        try:
            packet, _ = encode_frame(frame, st.session_state.frame_count, jpeg_quality)
            send_packet(sock, packet)
        except Exception as e:
            st.session_state.errors += 1
            status_ph.error(f"⚠️ Send error: {e}")
            break
        t_enc_1 = time.perf_counter()

        # ── Update stats ─────────────────────────
        enc_ms  = (t_enc_1 - t_enc_0) * 1000
        enc_times.append(enc_ms)
        pkt_sizes.append(len(packet))

        st.session_state.frame_count += 1
        st.session_state.bytes_sent  += len(packet)

        t_now      = time.perf_counter()
        fps_smooth = 0.85 * fps_smooth + 0.15 / max(t_now - t_prev, 1e-6)
        t_prev     = t_now

        # ── UI update (every ~15 frames to avoid Streamlit CPU bottleneck on RPi) ──
        if st.session_state.frame_count % 15 == 0:
            session_s  = t_now - t_session
            bw_mbps    = (st.session_state.bytes_sent * 8 / 1e6) / max(session_s, 1e-6)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Draw detections
            h_f, w_f = frame_rgb.shape[:2]
            for (x1_n, y1_n, x2_n, y2_n, conf, cls_id) in latest_detections:
                x1, y1 = int(x1_n * w_f), int(y1_n * h_f)
                x2, y2 = int(x2_n * w_f), int(y2_n * h_f)
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 60, 60), 2)
                label = f"cls:{cls_id} {conf:.2f}"
                cv2.putText(frame_rgb, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Overlay FPS on preview
            cv2.putText(frame_rgb, f"FPS: {fps_smooth:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 80), 2)
            feed_ph.image(frame_rgb, use_container_width=True)

            stats_dict = {
                "fps":         fps_smooth,
                "frames":      st.session_state.frame_count,
                "bw_mbps":     bw_mbps,
                "avg_pkt_kb":  np.mean(pkt_sizes[-50:]) / 1024 if pkt_sizes else 0,
                "avg_enc_ms":  np.mean(enc_times[-50:]) if enc_times else 0,
                "errors":      st.session_state.errors,
            }
            stats_ph.markdown(render_stats(stats_dict), unsafe_allow_html=True)

            elapsed_str = f"{int(session_s // 60)}m {int(session_s % 60)}s"
            sess_ph.markdown(f"""
            <div style="background:#161b22;border-radius:8px;padding:12px;font-size:0.85em;">
              <div style="display:flex;justify-content:space-between;
                          border-bottom:1px solid #21262d;padding:4px 0;">
                <span style="color:#8b949e;">Session time</span>
                <span style="color:#c9d1d9;font-weight:600;">{elapsed_str}</span>
              </div>
              <div style="display:flex;justify-content:space-between;
                          border-bottom:1px solid #21262d;padding:4px 0;">
                <span style="color:#8b949e;">Total sent</span>
                <span style="color:#c9d1d9;font-weight:600;">
                  {st.session_state.bytes_sent / 1e6:.1f} MB</span>
              </div>
              <div style="display:flex;justify-content:space-between;padding:4px 0;">
                <span style="color:#8b949e;">Resolution</span>
                <span style="color:#c9d1d9;font-weight:600;">{res_w}×{res_h}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        net_ph.markdown(render_net(jetson_ip, port, jpeg_quality), unsafe_allow_html=True)

    cap.release()
    status_ph.info("⏹ Streaming stopped.")

else:
    # Idle state
    feed_ph.markdown("""
    <div style="background:#161b22;border:2px dashed #30363d;border-radius:12px;
                padding:80px;text-align:center;color:#8b949e;">
      <div style="font-size:3em;">📡</div>
      <h3 style="color:#8b949e;">Not connected</h3>
      <p>Set the Jetson IP and click <b>▶ Connect &amp; Stream</b></p>
    </div>""", unsafe_allow_html=True)

    net_ph.markdown(render_net(jetson_ip, int(port), jpeg_quality), unsafe_allow_html=True)
