"""
frame_protocol.py — Shared camera-to-Jetson transport protocol
===============================================================
Used by:
  rpi_sender.py  — Raspberry Pi (transmitter)
  benchmark.py   — Jetson Orin  (receiver + inference)

Wire format (per frame)
────────────────────────
  Bytes  0–3  : JPEG payload length  (uint32 big-endian)
  Bytes  4–11 : sender timestamp     (uint64 µs since epoch, sender clock)
  Bytes 12–15 : frame sequence number (uint32)
  Bytes 16–19 : original frame width  (uint32)
  Bytes 20–23 : original frame height (uint32)
  Bytes 24+   : JPEG-encoded frame data

RTT ping-pong (for one-way latency estimation)
───────────────────────────────────────────────
  PING: b'PING' + uint64 sender_ts_us  (12 bytes)
  PONG: b'PONG' + uint64 sender_ts_us + uint64 recv_ts_us  (20 bytes)
"""

import struct
import socket
import time

# ── Header ────────────────────────────────────────────────────────────────────
FRAME_HDR_FMT  = "!IQIIII"          # see layout above
FRAME_HDR_SIZE = struct.calcsize(FRAME_HDR_FMT)   # 28 bytes

PING_FMT  = "!4sQ"
PONG_FMT  = "!4sQQ"
PING_SIZE = struct.calcsize(PING_FMT)   # 12 bytes
PONG_SIZE = struct.calcsize(PONG_FMT)   # 20 bytes

# ── Detection result packet (Jetson → RPi) ────────────────────────────────────
# DETS header: "DETS" tag + frame_num (uint32) + n_dets (uint32) = 12 bytes
# Per detection: x1_norm, y1_norm, x2_norm, y2_norm, conf, class_id = 6 × float32 = 24 bytes
# Coordinates are NORMALISED to [0, 1] relative to the ORIGINAL frame (before YOLO resize)
DETS_HDR_FMT  = "!4sII"                       # tag, frame_num, n_dets
DETS_HDR_SIZE = struct.calcsize(DETS_HDR_FMT) # 12 bytes
DET_FMT       = "!6f"                         # x1, y1, x2, y2, conf, class_id
DET_SIZE      = struct.calcsize(DET_FMT)       # 24 bytes

DEFAULT_PORT    = 9876
DEFAULT_QUALITY = 75       # JPEG quality 0–100


# ── Encoding / decoding ───────────────────────────────────────────────────────
def encode_frame(frame_bgr, frame_num: int,
                 jpeg_quality: int = DEFAULT_QUALITY) -> tuple[bytes, int]:
    """
    Encode a BGR frame into a wire packet.
    Returns (packet_bytes, send_timestamp_us).
    """
    import cv2
    h, w = frame_bgr.shape[:2]
    ts_us = int(time.time() * 1_000_000)
    ret, jpeg = cv2.imencode(
        ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    )
    if not ret:
        raise RuntimeError("JPEG encoding failed")
    jpeg_bytes = jpeg.tobytes()
    header = struct.pack(
        FRAME_HDR_FMT,
        len(jpeg_bytes),   # jpeg payload length
        ts_us,             # sender timestamp µs
        frame_num,         # sequence number
        w,                 # frame width
        h,                 # frame height
        jpeg_quality,      # quality used
    )
    return header + jpeg_bytes, ts_us


def decode_frame(packet_bytes: bytes) -> tuple:
    """
    Decode a wire packet.
    Returns (frame_bgr, frame_num, send_timestamp_us, width, height, quality).
    """
    import cv2
    import numpy as np

    jpeg_len, ts_us, frame_num, w, h, quality = struct.unpack_from(
        FRAME_HDR_FMT, packet_bytes, 0
    )
    jpeg_bytes = packet_bytes[FRAME_HDR_SIZE : FRAME_HDR_SIZE + jpeg_len]
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    return frame, frame_num, ts_us, w, h, quality


# ── Socket helpers ─────────────────────────────────────────────────────────────
def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from socket. Returns b'' on disconnect."""
    buf = bytearray(n)
    view = memoryview(buf)
    pos  = 0
    while pos < n:
        count = sock.recv_into(view[pos:], n - pos)
        if count == 0:
            return b""
        pos += count
    return bytes(buf)


def recv_packet(sock: socket.socket) -> tuple[bytes | None, int]:
    """
    Receive one complete frame packet.
    Returns (packet_bytes, recv_timestamp_us) or (None, 0) on disconnect.
    """
    hdr = recv_exact(sock, FRAME_HDR_SIZE)
    if not hdr:
        return None, 0
    recv_ts_us = int(time.time() * 1_000_000)
    jpeg_len   = struct.unpack_from("!I", hdr, 0)[0]
    jpeg       = recv_exact(sock, jpeg_len)
    if not jpeg:
        return None, 0
    return hdr + jpeg, recv_ts_us


def send_packet(sock: socket.socket, packet_bytes: bytes) -> int:
    """Send a complete packet. Returns bytes sent."""
    sock.sendall(packet_bytes)
    return len(packet_bytes)


# ── RTT ping-pong ─────────────────────────────────────────────────────────────
def send_ping(sock: socket.socket) -> int:
    """Send a PING. Returns send timestamp µs."""
    ts = int(time.time() * 1_000_000)
    sock.sendall(struct.pack(PING_FMT, b"PING", ts))
    return ts


def handle_ping(data: bytes, sock: socket.socket):
    """Receiver: parse PING and send PONG back."""
    _, send_ts = struct.unpack(PING_FMT, data)
    recv_ts    = int(time.time() * 1_000_000)
    sock.sendall(struct.pack(PONG_FMT, b"PONG", send_ts, recv_ts))


def recv_pong(sock: socket.socket, send_ts: int) -> dict:
    """
    Sender: receive PONG and compute RTT / estimated one-way latency.
    Returns dict with rtt_us, one_way_us, recv_ts_us (at Jetson).
    """
    data       = recv_exact(sock, PONG_SIZE)
    t3         = int(time.time() * 1_000_000)
    _, orig_ts, jetson_recv_ts = struct.unpack(PONG_FMT, data)
    rtt_us     = t3 - orig_ts
    one_way_us = rtt_us / 2          # assumes symmetric path
    return {
        "rtt_us":          rtt_us,
        "one_way_us":      one_way_us,
        "rtt_ms":          rtt_us   / 1000,
        "one_way_ms":      one_way_us / 1000,
        "jetson_recv_ts":  jetson_recv_ts,
    }


def measure_rtt(sock: socket.socket, n_pings: int = 20) -> dict:
    """
    Perform n_pings RTT measurements.
    Returns statistics dict: mean, std, min, max, all values (ms).
    """
    import numpy as np
    rtts = []
    for _ in range(n_pings):
        ts   = send_ping(sock)
        res  = recv_pong(sock, ts)
        rtts.append(res["rtt_ms"])
        time.sleep(0.01)
    a = np.array(rtts)
    return {
        "rtts_ms":        rtts,
        "rtt_mean_ms":    float(a.mean()),
        "rtt_std_ms":     float(a.std(ddof=1)),
        "rtt_min_ms":     float(a.min()),
        "rtt_max_ms":     float(a.max()),
        "one_way_est_ms": float(a.mean() / 2),
        "jitter_ms":      float(a.std(ddof=1)),   # RFC 3550 jitter ≈ std
        "n":              n_pings,
    }


# ── Detection result send / receive ───────────────────────────────────────────
def send_detections(sock: socket.socket, dets: list, frame_num: int) -> bool:
    """
    Jetson → RPi: send detection results for a frame.

    dets: list of (x1_norm, y1_norm, x2_norm, y2_norm, conf, class_id)
          All coordinates NORMALISED to [0, 1] relative to the camera frame.
    Returns True on success, False on socket error.
    """
    try:
        hdr = struct.pack(DETS_HDR_FMT, b"DETS", frame_num, len(dets))
        payload = bytearray(hdr)
        for (x1, y1, x2, y2, conf, cls) in dets:
            payload += struct.pack(DET_FMT, x1, y1, x2, y2, conf, cls)
        sock.sendall(bytes(payload))
        return True
    except Exception:
        return False


def recv_detections_nonblocking(sock: socket.socket) -> tuple | None:
    """
    RPi: receive one DETS packet (call when you have already confirmed 4-byte peek == b'DETS').
    Reads the full header + all detection structs in blocking mode.

    Returns (frame_num, dets_list) where dets_list is:
        [(x1_norm, y1_norm, x2_norm, y2_norm, conf, class_id), ...]
    Returns None on error / incomplete data.
    """
    try:
        hdr = recv_exact(sock, DETS_HDR_SIZE)
        if not hdr or hdr[:4] != b"DETS":
            return None
        _, frame_num, n_dets = struct.unpack(DETS_HDR_FMT, hdr)
        dets = []
        for _ in range(n_dets):
            raw = recv_exact(sock, DET_SIZE)
            if not raw:
                return None
            x1, y1, x2, y2, conf, cls = struct.unpack(DET_FMT, raw)
            dets.append((x1, y1, x2, y2, conf, int(cls)))
        return frame_num, dets
    except Exception:
        return None
