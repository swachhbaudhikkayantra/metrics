#!/usr/bin/env python3
"""
cli_benchmark.py — Headless Scientific Benchmark for Jetson Orin Nano
=====================================================================
Runs headless benchmarks across all YOLOv8 and YOLOv11 models (FP32/FP16/INT8).
Two input modes:
  Scenario A (direct):   synthetic or real images from --images folder
  Scenario B (wireless): --wireless  →  starts TCP server, receives frames from RPi

Usage:
    # Scenario A — direct (default, synthetic frames):
    python3 cli_benchmark.py --frames 150 --runs 3 --warmup 10

    # Scenario A — real images:
    python3 cli_benchmark.py --images /path/to/images/

    # Scenario B — wireless (run rpi_sender.py on RPi first):
    python3 cli_benchmark.py --wireless --port 9876 --frames 150 --runs 3
"""

import argparse
import csv
import json
import os
import re
import socket
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# Suppress benign DRM device discovery warnings on Tegra
ort.set_default_logger_severity(3)

# TensorRT backend for FP16 / INT8 (builds engines from FP32 ONNX)
try:
    from trt_inference import TRTSession
    TRT_AVAILABLE = True
except Exception as _trt_err:
    TRT_AVAILABLE = False
    print(f"[WARN] TensorRT unavailable — FP16/INT8 will be skipped: {_trt_err}")

# Import local statistical utilities
import stats_utils as sp_stats

# Import detection result sender (for Jetson → RPi feedback in Scenario B)
from frame_protocol import send_detections

BASE_DIR = Path(__file__).parent.resolve()


# ── YOLO output decoder + NMS ─────────────────────────────────────────────────
def decode_yolo_output(output: np.ndarray,
                       input_hw: tuple,
                       conf_thresh: float = 0.25,
                       iou_thresh: float = 0.45,
                       num_classes: int = 8) -> list:
    """
    Decode raw YOLO inference output → list of normalised detections.

    output   : raw network output, shape (1, C, A)
                 C = num_classes + 4  [YOLOv8 det]
                 C = num_classes + 4 + 32  [YOLOv11 seg, mask coeffs ignored]
    input_hw : (H, W) of the tensor fed to the model (e.g. 640×640)
    Returns  : [(x1_n, y1_n, x2_n, y2_n, conf, class_id), ...]
               All coords normalised to [0, 1] w.r.t. input_hw.
    """
    if isinstance(output, (list, tuple)):
        output = output[0]
    out = np.squeeze(output)   # Shape (C, A) e.g. (12, 8400) or (44, 6804)
    if out.ndim != 2 or out.shape[0] < 4 + num_classes:
        return []
    boxes_raw = out[:4, :]    # cx, cy, w, h  — pixel-space in model input
    scores    = out[4:4 + num_classes, :]  # (num_classes, A)

    confs    = np.max(scores, axis=0)      # (A,)
    cls_ids  = np.argmax(scores, axis=0)   # (A,)

    mask     = confs >= conf_thresh
    if not np.any(mask):
        return []

    cx = boxes_raw[0, mask];  cy = boxes_raw[1, mask]
    bw = boxes_raw[2, mask];  bh = boxes_raw[3, mask]
    H, W = input_hw

    # cx/cy/w/h → x1/y1/x2/y2, normalised
    x1 = np.clip((cx - bw / 2) / W, 0, 1)
    y1 = np.clip((cy - bh / 2) / H, 0, 1)
    x2 = np.clip((cx + bw / 2) / W, 0, 1)
    y2 = np.clip((cy + bh / 2) / H, 0, 1)
    c  = confs[mask]
    cl = cls_ids[mask]

    # Greedy NMS
    order = np.argsort(c)[::-1]
    keep  = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        ai    = (x2[i] - x1[i]) * (y2[i] - y1[i])
        ar    = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
        iou   = inter / (ai + ar - inter + 1e-7)
        order = order[1:][iou <= iou_thresh]

    return [(float(x1[k]), float(y1[k]), float(x2[k]), float(y2[k]),
             float(c[k]), int(cl[k])) for k in keep]



# Each entry specifies:
#   fp32_path  → FP32 ONNX used to build TRT engine (FP16/INT8 modes)
#   ort_path   → ONNX loaded by ORT (only for FP32)
#   backend    → 'ort' (OnnxRuntime CPU) | 'trt' (TensorRT GPU)
MODELS = [
    # ── YOLOv8 ──────────────────────────────────────────────────────────────
    {
        "name": "YOLOv8", "quant": "FP32", "res": (640, 640),
        "backend": "ort",
        "ort_path": BASE_DIR / "yolov8/best_fp32.onnx",
    },
    {
        "name": "YOLOv8", "quant": "FP16", "res": (640, 640),
        "backend": "trt",
        "fp32_path": BASE_DIR / "yolov8/best_fp32.onnx",   # TRT builds FP16 engine from this
    },
    {
        "name": "YOLOv8", "quant": "INT8", "res": (640, 640),
        "backend": "trt",
        "fp32_path": BASE_DIR / "yolov8/best_fp32.onnx",   # TRT builds INT8 engine from this
    },
    # ── YOLOv11 ─────────────────────────────────────────────────────────────
    {
        "name": "YOLOv11", "quant": "FP32", "res": (576, 576),
        "backend": "ort",
        "ort_path": BASE_DIR / "yolov11/best_fp32.onnx",
    },
    {
        "name": "YOLOv11", "quant": "FP16", "res": (576, 576),
        "backend": "trt",
        "fp32_path": BASE_DIR / "yolov11/best_fp32.onnx",
    },
    {
        "name": "YOLOv11", "quant": "INT8", "res": (576, 576),
        "backend": "trt",
        "fp32_path": BASE_DIR / "yolov11/best_fp32.onnx",
        # TRT 10.3.0 has no INT8 kernel for YOLOv11-seg's prototype Conv+SiLU head.
        # Error: "Could not find any implementation for node /model.23/proto/cv3/conv/Conv"
        # Document in paper as: YOLOv11-seg INT8 not supported on TRT 10.3.0 / Jetson Orin Nano.
        "skip": True,
        "skip_reason": "TRT 10.3.0: no INT8 kernel for YOLOv11-seg prototype head (model.23/proto)",
    },
]

# ── Tegrastats Monitor ─────────────────────────────────────────────────────────
class TegrastatsRunner:
    def __init__(self, interval_ms=1000):
        self.interval = interval_ms
        self.proc = None
        self.samples = []
        self._regex = re.compile(
            r"RAM (\d+)/(\d+)MB"
            r".*?SWAP (\d+)/(\d+)MB"
            r".*?CPU \[([^\]]+)\]"
            r".*?GR3D_FREQ (\d+)%"
            r".*?cpu@([\d.]+)C"
            r".*?gpu@([\d.]+)C"
            r".*?tj@([\d.]+)C"
            r".*?VDD_IN ([\d.]+)mW"
            r".*?VDD_CPU_GPU_CV ([\d.]+)mW"
            r".*?VDD_SOC ([\d.]+)mW"
        )

    def start(self):
        self.samples = []
        try:
            self.proc = subprocess.Popen(
                ["/usr/bin/tegrastats", "--interval", str(self.interval)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception:
            self.proc = None

    def stop(self):
        if not self.proc:
            return
        try:
            self.proc.terminate()
            stdout, _ = self.proc.communicate(timeout=1)
            for line in stdout.splitlines():
                self._parse(line)
        except Exception:
            pass

    def collect(self):
        if not self.proc:
            return
        # Drain available lines without blocking indefinitely
        import select
        while True:
            rlist, _, _ = select.select([self.proc.stdout], [], [], 0.01)
            if not rlist:
                break
            line = self.proc.stdout.readline()
            if not line:
                break
            self._parse(line)

    def _parse(self, line):
        m = self._regex.search(line)
        if not m:
            return
        (ram_u, ram_t, swp_u, swp_t,
         cpu_str, gpu_pct,
         t_cpu, t_gpu, t_tj,
         p_in, p_cg, p_soc) = m.groups()

        cpu_loads = []
        for c in cpu_str.split(","):
            p = c.split("@")
            try:
                cpu_loads.append(int(p[0].replace("%", "")))
            except ValueError:
                pass

        self.samples.append({
            "ram_used_mb": int(ram_u),
            "gpu_pct": float(gpu_pct),
            "cpu_avg_pct": sum(cpu_loads) / len(cpu_loads) if cpu_loads else 0.0,
            "temp_cpu_c": float(t_cpu),
            "temp_gpu_c": float(t_gpu),
            "temp_tj_c": float(t_tj),
            "power_total_mw": float(p_in),
            "power_cpu_gpu_mw": float(p_cg),
        })

    def summary(self):
        if not self.samples:
            return {}
        def _get(k):
            vals = [s[k] for s in self.samples if k in s]
            if not vals:
                return {"min": 0.0, "max": 0.0, "avg": 0.0, "peak": 0.0}
            a = np.array(vals)
            return {"min": float(a.min()), "max": float(a.max()), "avg": float(a.mean()), "peak": float(a.max())}

        return {
            "gpu_pct": _get("gpu_pct"),
            "cpu_avg_pct": _get("cpu_avg_pct"),
            "ram_used_mb": _get("ram_used_mb"),
            "temp_cpu_c": _get("temp_cpu_c"),
            "temp_gpu_c": _get("temp_gpu_c"),
            "temp_tj_c": _get("temp_tj_c"),
            "power_total_mw": _get("power_total_mw"),
            "power_cpu_gpu_mw": _get("power_cpu_gpu_mw"),
        }

# ── Inference Preprocessor ───────────────────────────────────────────────────
def preprocess(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    img = cv2.resize(frame, (target_w, target_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(img.transpose(2, 0, 1), 0)

# ── Image loader (real frames or synthetic) ───────────────────────────────────
def load_frames(images_dir, target_w, target_h, n_needed):
    """
    Load images from a directory (cycles if fewer than n_needed).
    Falls back to a fixed synthetic frame if images_dir is None or empty.
    """
    frames = []
    if images_dir:
        img_dir = Path(images_dir)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
    if not frames:
        # Synthetic: fixed-seed random frame — reproducible across runs
        rng = np.random.default_rng(42)
        frames = [rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)]
        source_label = "synthetic (seed=42)"
    else:
        source_label = f"{len(frames)} real images from {images_dir}"
    # Pre-preprocess and cycle to fill exactly n_needed slots
    preprocessed = []
    for i in range(n_needed):
        raw = frames[i % len(frames)]
        preprocessed.append(preprocess(raw, target_w, target_h))
    return preprocessed, source_label


# ── Wireless Receiver (Scenario B: RPi → WiFi → Jetson) ──────────────────────
# Wire format shared with rpi_sender.py via frame_protocol.py
# Header: !IQIIII = jpeg_len(4) + ts_us(8) + frame_num(4) + w(4) + h(4) + quality(4) = 28 bytes
_FRAME_HDR_FMT  = "!IQIIII"
_FRAME_HDR_SIZE = struct.calcsize(_FRAME_HDR_FMT)   # 28 bytes
_PING_FMT       = "!4sQ"
_PONG_FMT       = "!4sQQ"
_PING_SIZE      = struct.calcsize(_PING_FMT)          # 12 bytes
_PONG_SIZE      = struct.calcsize(_PONG_FMT)          # 20 bytes


def _recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    while pos < n:
        count = sock.recv_into(view[pos:], n - pos)
        if count == 0:
            return b""
        pos += count
    return bytes(buf)


class WirelessReceiver:
    """
    Jetson-side TCP server for Scenario B.
    Listens for one RPi connection, receives JPEG frames,
    and measures network latency via RTT ping-pong.

    Usage:
        with WirelessReceiver(port=9876) as rcv:
            rtt = rcv.measure_rtt(n_pings=20)
            frame_bgr = rcv.recv_frame()   # blocking
    """

    def __init__(self, port: int = 9876):
        self.port = port
        self._srv  = None
        self._conn = None
        self._addr = None
        self.net_latency_ms: dict = {}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    def start(self):
        """Start listening and block until RPi connects."""
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("0.0.0.0", self.port))
        self._srv.listen(1)
        print(f"\n  [NET] TCP server listening on :{self.port}")
        print(f"        → Start rpi_sender.py on RPi and connect to this machine's IP")
        self._conn, self._addr = self._srv.accept()
        self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"  [NET] ✅ RPi connected from {self._addr[0]}:{self._addr[1]}")

    def stop(self):
        """Close connection and server socket."""
        for s in (self._conn, self._srv):
            if s:
                try:
                    s.close()
                except Exception:
                    pass
        self._conn = self._srv = None

    def measure_rtt(self, n_pings: int = 10) -> dict:
        """
        Send n_pings PING messages to the RPi and measure RTT.
        RPi's handle_ping() replies with PONG carrying both timestamps.
        One-way latency = RTT / 2 (assumes symmetric WiFi path).
        """
        import numpy as _np
        rtts_ms = []
        print(f"  [NET] Handshaking and measuring RTT ({n_pings} pings)...", end="", flush=True)
        for _ in range(n_pings):
            t_send = int(time.time() * 1_000_000)
            self._conn.sendall(struct.pack(_PING_FMT, b"PING", t_send))
            pong = _recv_exact(self._conn, _PONG_SIZE)
            t_recv = int(time.time() * 1_000_000)
            if len(pong) == _PONG_SIZE and pong[:4] == b"PONG":
                _, orig_ts, _ = struct.unpack(_PONG_FMT, pong)
                rtt_us = t_recv - orig_ts
                if 0 <= rtt_us < 1_000_000:
                    rtts_ms.append(rtt_us / 1000.0)
            time.sleep(0.01)
        if not rtts_ms:
            rtts_ms = [5.0]  # sensible fallback if no pings received
        a = _np.array(rtts_ms)
        self.net_latency_ms = {
            "rtt_mean_ms":    float(a.mean()),
            "rtt_std_ms":     float(a.std(ddof=1)) if len(a) > 1 else 0.5,
            "rtt_min_ms":     float(a.min()),
            "rtt_max_ms":     float(a.max()),
            "one_way_est_ms": float(a.mean() / 2),
            "jitter_ms":      float(a.std(ddof=1)) if len(a) > 1 else 0.5,
            "n_pings":        len(rtts_ms),
        }
        # Send START token to tell RPi to begin video capture stream
        try:
            self._conn.sendall(b"START")
        except Exception:
            pass
        print(f" RTT mean={a.mean():.1f}ms  one-way≈{a.mean()/2:.1f}ms  jitter={self.net_latency_ms['jitter_ms']:.2f}ms")
        return self.net_latency_ms

    def recv_frame_bgr(self) -> tuple:
        """
        Receive one JPEG frame packet.
        Returns (frame_bgr, rx_ms, dec_ms, pkt_bytes, recv_ts_us, send_ts_us, frame_num)
        or (None, 0.0, 0.0, 0, 0, 0, 0) on disconnect.
        """
        t0 = time.perf_counter()
        hdr = _recv_exact(self._conn, _FRAME_HDR_SIZE)
        if not hdr:
            return None, 0.0, 0.0, 0, 0, 0, 0
        recv_ts_us = int(time.time() * 1_000_000)
        jpeg_len, send_ts_us, frame_num, w, h, quality = struct.unpack(_FRAME_HDR_FMT, hdr)
        jpeg_data = _recv_exact(self._conn, jpeg_len)
        if not jpeg_data:
            return None, 0.0, 0.0, 0, 0, 0, 0
        t1 = time.perf_counter()
        rx_ms = (t1 - t0) * 1000.0

        t_dec0 = time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
        t_dec1 = time.perf_counter()
        dec_ms = (t_dec1 - t_dec0) * 1000.0

        pkt_bytes = _FRAME_HDR_SIZE + jpeg_len
        return frame, rx_ms, dec_ms, pkt_bytes, recv_ts_us, send_ts_us, frame_num


# ── Single Model Evaluation ───────────────────────────────────────────────────
def benchmark_model(
    model_info: dict,
    n_frames: int,
    n_runs: int,
    n_warmup: int,
    interval_ms: int = 1000,
    images_dir: str = None,
    wireless_receiver: "WirelessReceiver | None" = None,
) -> dict:
    name     = model_info["name"]
    quant    = model_info["quant"]
    backend  = model_info["backend"]
    target_w, target_h = model_info["res"]

    is_wireless = wireless_receiver is not None

    print(f"\n╔══════════════════════════════════════════════════════════════════════╗")
    mode_tag = "WIRELESS" if is_wireless else "DIRECT"
    print(f"║  {name}  [{quant}]  {target_w}×{target_h}  backend={backend.upper():<4}  mode={mode_tag}  ║")
    print(f"╚══════════════════════════════════════════════════════════════════════╝")

    # ── Session setup ─────────────────────────────────────────────────────────
    t_load_0 = time.perf_counter()

    if backend == "trt":
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available — cannot run FP16/INT8")
        fp32_path = model_info["fp32_path"]
        file_size_kb = fp32_path.stat().st_size / 1024.0
        file_size_mb = file_size_kb / 1024.0
        sess = TRTSession(
            str(fp32_path),
            precision=quant.lower(),
            cache_dir=str(fp32_path.parent),
        )
    else:  # ort (FP32 CPU)
        ort_path = model_info["ort_path"]
        file_size_kb = ort_path.stat().st_size / 1024.0
        file_size_mb = file_size_kb / 1024.0
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            str(ort_path),
            sess_opts,
            providers=["CPUExecutionProvider"],
        )

    load_time_ms = (time.perf_counter() - t_load_0) * 1000.0
    provider     = sess.get_providers()[0]
    input_name   = sess.get_inputs()[0].name
    output_name  = sess.get_outputs()[0].name
    input_shape  = [str(x) for x in sess.get_inputs()[0].shape]
    output_shape = [str(x) for x in sess.get_outputs()[0].shape]
    input_type   = sess.get_inputs()[0].type
    output_type  = sess.get_outputs()[0].type

    print(f"  [1] File Size:               {file_size_kb:,.1f} KB ({file_size_mb:.2f} MB)")
    print(f"  [2] Quantization:            {quant}")
    print(f"  [3] Input:                   name='{input_name}', shape={input_shape}, type={input_type}")
    print(f"  [4] Output:                  name='{output_name}', shape={output_shape}, type={output_type}")
    print(f"  [5] Execution Provider:      {provider}")
    print(f"  [6] Session Load/Build Time: {load_time_ms:.2f} ms")

    # ── Frame source helper ───────────────────────────────────────────────────
    def get_next_blob():
        """
        Returns (blob, timing_dict)
        where timing_dict has: rx_ms, dec_ms, prep_ms, net_ms, pkt_bytes, frame_num
        """
        if is_wireless:
            frame, rx_ms, dec_ms, pkt_bytes, recv_ts, send_ts, frame_num = wireless_receiver.recv_frame_bgr()
            if frame is None:
                raise RuntimeError("RPi disconnected during benchmark")
            t_p0 = time.perf_counter()
            blob = preprocess(frame, target_w, target_h)
            t_p1 = time.perf_counter()
            prep_ms = (t_p1 - t_p0) * 1000.0

            net_ms = wireless_receiver.net_latency_ms.get("one_way_est_ms", 0.0) if wireless_receiver.net_latency_ms else max(0.0, (recv_ts - send_ts) / 1000.0)
            return blob, {
                "rx_ms": rx_ms,
                "dec_ms": dec_ms,
                "prep_ms": prep_ms,
                "net_ms": net_ms,
                "pkt_bytes": pkt_bytes,
                "frame_num": frame_num,
            }
        else:
            t_p0 = time.perf_counter()
            blob = next(blob_iter)
            t_p1 = time.perf_counter()
            return blob, {
                "rx_ms": 0.0,
                "dec_ms": 0.0,
                "prep_ms": (t_p1 - t_p0) * 1000.0,
                "net_ms": 0.0,
                "pkt_bytes": 0,
                "frame_num": 0,
            }

    # ── Load frames (direct mode only — wireless pulls live) ──────────────────
    net_latency_ms_list = []   # per-frame one-way network latency (Scenario B)

    if is_wireless:
        frame_source = f"wireless TCP from {wireless_receiver._addr[0]}"
        blob_iter = None
    else:
        total_needed = n_warmup + 1 + (n_runs * n_frames)
        all_blobs, frame_source = load_frames(images_dir, target_w, target_h, total_needed)
        blob_iter = iter(all_blobs)

    print(f"  [7] Input Frames:            {frame_source}")
    if is_wireless and wireless_receiver.net_latency_ms:
        rtt = wireless_receiver.net_latency_ms
        print(f"  [8] Network RTT:             {rtt['rtt_mean_ms']:.1f} ± {rtt['rtt_std_ms']:.2f} ms  "
              f"(one-way ≈ {rtt['one_way_est_ms']:.1f} ms)")


    # 1. Warm-up
    print(f"\n  ⏳ Warming up ({n_warmup} frames)...", end="", flush=True)
    for _ in range(n_warmup):
        blob, tm = get_next_blob()
        out = sess.run([output_name], {input_name: blob})
        if is_wireless:
            dets = decode_yolo_output(out, (target_h, target_w))
            send_detections(wireless_receiver._conn, dets, tm["frame_num"])
    print(" Done.")

    # 2. First-frame latency
    ff_blob, ff_tm = get_next_blob()
    t0 = time.perf_counter()
    ff_out = sess.run([output_name], {input_name: ff_blob})
    t2 = time.perf_counter()
    ff_inf_ms = (t2 - t0) * 1000.0
    ff_pre_ms = ff_tm["prep_ms"]
    ff_tot_ms = ff_tm["net_ms"] + ff_tm["dec_ms"] + ff_pre_ms + ff_inf_ms
    print(f"  ⏱ First Frame Latency:      Infer = {ff_inf_ms:.2f} ms | Total = {ff_tot_ms:.2f} ms")
    if is_wireless:
        dets = decode_yolo_output(ff_out, (target_h, target_w))
        send_detections(wireless_receiver._conn, dets, ff_tm["frame_num"])

    # 3. Multi-trial benchmark
    all_inf_ms  = []
    all_prep_ms = []
    all_dec_ms  = []
    all_rx_ms   = []
    all_net_ms  = []
    all_e2e_ms  = []
    all_pkt_b   = []
    trial_means = []
    t_bench_start = time.perf_counter()
    teg = TegrastatsRunner(interval_ms=interval_ms)

    teg.start()
    for run_idx in range(n_runs):
        print(f"  Trial {run_idx+1}/{n_runs} ({n_frames} frames)...", end="", flush=True)
        run_inf_ms = []
        for _ in range(n_frames):
            blob, tm = get_next_blob()
            tp1 = time.perf_counter()
            out = sess.run([output_name], {input_name: blob})
            tp2 = time.perf_counter()
            inf_ms = (tp2 - tp1) * 1000.0
            
            if is_wireless:
                dets = decode_yolo_output(out, (target_h, target_w))
                send_detections(wireless_receiver._conn, dets, tm["frame_num"])

            run_inf_ms.append(inf_ms)
            all_prep_ms.append(tm["prep_ms"])
            all_dec_ms.append(tm["dec_ms"])
            all_rx_ms.append(tm["rx_ms"])
            all_net_ms.append(tm["net_ms"])
            all_pkt_b.append(tm["pkt_bytes"])
            all_e2e_ms.append(tm["net_ms"] + tm["dec_ms"] + tm["prep_ms"] + inf_ms)
            teg.collect()

        mean_run = float(np.mean(run_inf_ms))
        trial_means.append(mean_run)
        all_inf_ms.extend(run_inf_ms)
        print(f" Mean Infer: {mean_run:.2f} ms")
        if run_idx < n_runs - 1:
            time.sleep(1.0) # Thermal cool-down between trials

    teg.stop()
    teg_stats = teg.summary()

    def calc_dist_metrics(data_list):
        arr_data = np.array(data_list, dtype=np.float64)
        n_pts = len(arr_data)
        if n_pts == 0:
            return {"mean_ms": 0.0, "sd_ms": 0.0, "sem_ms": 0.0, "ci_lo_ms": 0.0, "ci_hi_ms": 0.0,
                    "median_ms": 0.0, "q1_ms": 0.0, "q3_ms": 0.0, "iqr_ms": 0.0, "p90_ms": 0.0,
                    "p95_ms": 0.0, "p99_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "cv_pct": 0.0}
        m = float(arr_data.mean())
        s = float(arr_data.std(ddof=1)) if n_pts > 1 else 0.0
        sem = s / np.sqrt(n_pts) if n_pts > 1 else 0.0
        tc = float(sp_stats.t_ppf(0.975, df=n_pts - 1)) if n_pts > 1 else 1.96
        q25 = float(np.percentile(arr_data, 25))
        q75 = float(np.percentile(arr_data, 75))
        return {
            "mean_ms": m,
            "sd_ms": s,
            "sem_ms": sem,
            "ci_lo_ms": m - tc * sem,
            "ci_hi_ms": m + tc * sem,
            "median_ms": float(np.percentile(arr_data, 50)),
            "q1_ms": q25,
            "q3_ms": q75,
            "iqr_ms": q75 - q25,
            "p90_ms": float(np.percentile(arr_data, 90)),
            "p95_ms": float(np.percentile(arr_data, 95)),
            "p99_ms": float(np.percentile(arr_data, 99)),
            "min_ms": float(arr_data.min()),
            "max_ms": float(arr_data.max()),
            "cv_pct": float((s / m) * 100.0) if m > 0 else 0.0,
        }

    # 4. Statistical computation
    arr = np.array(all_inf_ms, dtype=np.float64)
    n = len(arr)
    inf_dist = calc_dist_metrics(all_inf_ms)
    prep_dist = calc_dist_metrics(all_prep_ms)
    dec_dist = calc_dist_metrics(all_dec_ms)
    net_dist = calc_dist_metrics(all_net_ms)
    e2e_dist = calc_dist_metrics(all_e2e_ms)

    mean_val = inf_dist["mean_ms"]
    sd_val = inf_dist["sd_ms"]
    sem_val = inf_dist["sem_ms"]
    ci_lo = inf_dist["ci_lo_ms"]
    ci_hi = inf_dist["ci_hi_ms"]
    median_val = inf_dist["median_ms"]
    q1 = inf_dist["q1_ms"]
    q3 = inf_dist["q3_ms"]
    iqr = inf_dist["iqr_ms"]
    p90 = inf_dist["p90_ms"]
    p95 = inf_dist["p95_ms"]
    p99 = inf_dist["p99_ms"]
    min_val = inf_dist["min_ms"]
    max_val = inf_dist["max_ms"]
    cv_pct = inf_dist["cv_pct"]
    fps = 1000.0 / mean_val if mean_val > 0 else 0.0

    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(arr, size=n, replace=True).mean() for _ in range(1000)]
    boot_ci_lo = float(np.percentile(boot_means, 2.5))
    boot_ci_hi = float(np.percentile(boot_means, 97.5))

    # Real-Time Deadline Miss Analysis
    dmr_33  = float(np.mean([100.0 if x > 33.33 else 0.0 for x in all_e2e_ms]))
    dmr_50  = float(np.mean([100.0 if x > 50.0  else 0.0 for x in all_e2e_ms]))
    dmr_100 = float(np.mean([100.0 if x > 100.0 else 0.0 for x in all_e2e_ms]))
    dmr_200 = float(np.mean([100.0 if x > 200.0 else 0.0 for x in all_e2e_ms]))

    # Higher moments & normality
    skew_val = float(sp_stats.skew(arr))
    kurt_val = float(sp_stats.kurtosis(arr, fisher=True))
    sw_stat, sw_p = sp_stats.shapiro(arr[:1000])

    total_bench_duration_s = time.perf_counter() - t_bench_start
    avg_pwr_w = (teg_stats.get("power_total_mw", {}).get("avg", 0.0)) / 1000.0
    fps_per_w = (fps / avg_pwr_w) if avg_pwr_w > 0 else 0.0
    mj_per_inf = (avg_pwr_w * mean_val) if avg_pwr_w > 0 else 0.0

    total_bytes = sum(all_pkt_b)
    goodput_mbps = float((total_bytes * 8.0 / 1e6) / max(total_bench_duration_s, 1e-6))
    avg_pkt_kb = float(np.mean(all_pkt_b) / 1024.0) if all_pkt_b else 0.0

    gpu_info = teg_stats.get("gpu_pct", {"min": 0, "max": 0, "avg": 0, "peak": 0})
    cpu_info = teg_stats.get("cpu_avg_pct", {"min": 0, "max": 0, "avg": 0, "peak": 0})
    ram_info = teg_stats.get("ram_used_mb", {"min": 0, "max": 0, "avg": 0, "peak": 0})
    pwr_info = teg_stats.get("power_total_mw", {"min": 0, "max": 0, "avg": 0, "peak": 0})

    total_pipeline_time_s = time.perf_counter() - t_load_0
    cold_start_ttff_ms = load_time_ms + ff_tot_ms

    if is_wireless:
        print(f"\n  ┌─────────────────────────────────────────────────────────────────────────────┐")
        print(f"  │         SCENARIO B (WIRELESS) LATENCY DECOMPOSITION & STAGE ANALYSIS        │")
        print(f"  ├────────────────────────┬─────────────┬─────────────┬─────────────┬──────────┤")
        print(f"  │ Pipeline Stage         │ Mean ± SD   │ 95% CI      │ P95 Tail    │ CV (%)   │")
        print(f"  ├────────────────────────┼─────────────┼─────────────┼─────────────┼──────────┤")
        print(f"  │ 1. Network Transit     │ {net_dist['mean_ms']:5.1f}±{net_dist['sd_ms']:<4.1f} │ [{net_dist['ci_lo_ms']:4.1f},{net_dist['ci_hi_ms']:4.1f}] │ {net_dist['p95_ms']:6.1f} ms  │ {net_dist['cv_pct']:5.1f}%   │")
        print(f"  │ 2. JPEG Decompress     │ {dec_dist['mean_ms']:5.1f}±{dec_dist['sd_ms']:<4.1f} │ [{dec_dist['ci_lo_ms']:4.1f},{dec_dist['ci_hi_ms']:4.1f}] │ {dec_dist['p95_ms']:6.1f} ms  │ {dec_dist['cv_pct']:5.1f}%   │")
        print(f"  │ 3. Tensor Preprocess   │ {prep_dist['mean_ms']:5.1f}±{prep_dist['sd_ms']:<4.1f} │ [{prep_dist['ci_lo_ms']:4.1f},{prep_dist['ci_hi_ms']:4.1f}] │ {prep_dist['p95_ms']:6.1f} ms  │ {prep_dist['cv_pct']:5.1f}%   │")
        print(f"  │ 4. Model Inference     │ {inf_dist['mean_ms']:5.1f}±{inf_dist['sd_ms']:<4.1f} │ [{inf_dist['ci_lo_ms']:4.1f},{inf_dist['ci_hi_ms']:4.1f}] │ {inf_dist['p95_ms']:6.1f} ms  │ {inf_dist['cv_pct']:5.1f}%   │")
        print(f"  ├────────────────────────┼─────────────┼─────────────┼─────────────┼──────────┤")
        print(f"  │ Total E2E Latency      │ {e2e_dist['mean_ms']:5.1f}±{e2e_dist['sd_ms']:<4.1f} │ [{e2e_dist['ci_lo_ms']:4.1f},{e2e_dist['ci_hi_ms']:4.1f}] │ {e2e_dist['p95_ms']:6.1f} ms  │ {e2e_dist['cv_pct']:5.1f}%   │")
        print(f"  └────────────────────────┴─────────────┴─────────────┴─────────────┴──────────┘")
        print(f"  • Effective E2E Throughput: {1000.0/e2e_dist['mean_ms']:.1f} FPS (Pure Model Inference: {fps:.1f} FPS)")
        print(f"  • Wi-Fi Network Goodput:    {goodput_mbps:.2f} Mbps (Avg packet: {avg_pkt_kb:.1f} KB)")
        print(f"  • Real-Time Deadline Miss:  DMR(<33ms)={dmr_33:.1f}% | DMR(<50ms)={dmr_50:.1f}% | DMR(<100ms)={dmr_100:.1f}% | DMR(<200ms)={dmr_200:.1f}%")

    print(f"\n  📊 MEASURED METRICS SUMMARY ({name} {quant}):")
    print(f"  ─────────────────────────────────────────────────────────────────────────────")
    print(f"  • Model Size:               {file_size_kb:,.1f} KB ({file_size_mb:.2f} MB)")
    print(f"  • Quantization Type:        {quant}")
    print(f"  • Input / Output Info:      In={input_shape} ({input_type}) | Out={output_shape} ({output_type})")
    print(f"  • Total Time (From Start):  {total_pipeline_time_s:.2f} s (Load: {load_time_ms/1000.0:.2f}s | Trials: {total_bench_duration_s:.2f}s)")
    print(f"  • Cold-Start TTFF:          {cold_start_ttff_ms:.2f} ms (Load: {load_time_ms:.1f} ms + 1st Frame Pipeline: {ff_tot_ms:.1f} ms)")
    print(f"  • First Frame Processing:   Preprocess = {ff_pre_ms:.2f} ms | Pure Infer = {ff_inf_ms:.2f} ms")
    print(f"  • Steady-State Processing:  Preprocess = {prep_dist['mean_ms']:.2f} ms | Pure Infer = {mean_val:.2f} ms (± {sd_val:.2f} ms SD)")
    print(f"  • 95% Confidence Interval:  [{ci_lo:.2f}, {ci_hi:.2f}] ms (Student's t)")
    print(f"  • Bootstrap 95% CI:         [{boot_ci_lo:.2f}, {boot_ci_hi:.2f}] ms")
    print(f"  • Latency Distribution:     Min = {min_val:.2f} ms | Median = {median_val:.2f} ms | Max = {max_val:.2f} ms")
    print(f"  • Percentiles & Spread:     IQR = {iqr:.2f} ms | P90 = {p90:.2f} ms | P95 = {p95:.2f} ms | P99 = {p99:.2f} ms")
    print(f"  • Stability (CV %):         {cv_pct:.2f}%")
    print(f"  • Model Inference FPS:      {fps:.2f} FPS")
    print(f"  • CPU Utilization (%):      Avg = {cpu_info['avg']:.1f}% | Min = {cpu_info['min']:.1f}% | Max/Peak = {cpu_info['peak']:.1f}%")
    print(f"  • GPU Utilization (%):      Avg = {gpu_info['avg']:.1f}% | Min = {gpu_info['min']:.1f}% | Max/Peak = {gpu_info['peak']:.1f}%")
    print(f"  • RAM Utilization (MB):     Avg = {ram_info['avg']:.0f} MB | Min = {ram_info['min']:.0f} MB | Max/Peak = {ram_info['peak']:.0f} MB")
    print(f"  • Power Consumption (W):    Avg = {avg_pwr_w:.2f} W | Min = {pwr_info['min']/1000:.2f} W | Peak = {pwr_info['peak']/1000:.2f} W")
    print(f"  • Energy Efficiency:        {fps_per_w:.2f} FPS/Watt ({mj_per_inf:.2f} mJ per inference)")
    print(f"  • Die Temperatures (°C):    CPU = {teg_stats.get('temp_cpu_c',{}).get('avg',0):.1f}°C | GPU = {teg_stats.get('temp_gpu_c',{}).get('avg',0):.1f}°C | Tj = {teg_stats.get('temp_tj_c',{}).get('avg',0):.1f}°C")
    print(f"  ─────────────────────────────────────────────────────────────────────────────")

    return {
        "model": name,
        "quant": quant,
        "size_kb": file_size_kb,
        "size_mb": file_size_mb,
        "input_res": f"{target_w}x{target_h}",
        "input_shape": input_shape,
        "output_shape": output_shape,
        "provider": provider,
        "timing_overview": {
            "total_execution_time_s": total_pipeline_time_s,
            "session_load_time_ms": load_time_ms,
            "cold_start_ttff_ms": cold_start_ttff_ms,
            "benchmark_trials_time_s": total_bench_duration_s,
        },
        "load_time_ms": load_time_ms,
        "n_runs": n_runs,
        "n_frames_total": n,
        "first_frame": {
            "pre_ms": ff_pre_ms,
            "inf_ms": ff_inf_ms,
            "total_ms": ff_tot_ms,
            "pipeline_total_ms": ff_tot_ms,
            "cold_start_ttff_ms": cold_start_ttff_ms,
        },
        "stats": {
            "mean_ms": mean_val,
            "sd_ms": sd_val,
            "sem_ms": sem_val,
            "ci_lo_ms": ci_lo,
            "ci_hi_ms": ci_hi,
            "boot_ci_lo_ms": boot_ci_lo,
            "boot_ci_hi_ms": boot_ci_hi,
            "median_ms": median_val,
            "q1_ms": q1,
            "q3_ms": q3,
            "iqr_ms": iqr,
            "p90_ms": p90,
            "p95_ms": p95,
            "p99_ms": p99,
            "min_ms": min_val,
            "max_ms": max_val,
            "cv_pct": cv_pct,
            "avg_fps": fps,
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "normality_p": sw_p,
            "trial_means": trial_means,
            "trial_sd": float(np.std(trial_means, ddof=1)) if len(trial_means) > 1 else 0.0,
        },
        "stage_decomposition": {
            "network_transit": net_dist,
            "jpeg_decompress": dec_dist,
            "tensor_preprocess": prep_dist,
            "model_inference": inf_dist,
            "total_e2e": e2e_dist,
            "deadline_miss_rates": {
                "dmr_33ms_pct": dmr_33,
                "dmr_50ms_pct": dmr_50,
                "dmr_100ms_pct": dmr_100,
                "dmr_200ms_pct": dmr_200,
            },
            "network_goodput_mbps": goodput_mbps,
            "avg_packet_kb": avg_pkt_kb,
        } if is_wireless else None,
        "resources": teg_stats,
        "energy": {
            "avg_power_w": avg_pwr_w,
            "fps_per_watt": fps_per_w,
            "mj_per_infer": mj_per_inf,
        },
        "raw_latencies": all_inf_ms,
    }

# ── Main Benchmarking Loop ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run headless Jetson YOLO benchmark")
    parser.add_argument("--frames",   type=int,  default=150,  help="Frames per trial (default: 150)")
    parser.add_argument("--runs",     type=int,  default=3,    help="Independent trials (default: 3)")
    parser.add_argument("--warmup",   type=int,  default=10,   help="Warm-up frames discarded (default: 10)")
    parser.add_argument("--interval", type=int,  default=1000, help="tegrastats interval ms (default: 1000)")
    parser.add_argument("--images",   type=str,  default=None,
                        help="Folder of .jpg/.png images for input (default: synthetic seed=42)")
    parser.add_argument("--wireless", action="store_true",
                        help="Scenario B: start TCP server and receive frames from RPi sender")
    parser.add_argument("--port",     type=int,  default=9876,
                        help="TCP port to listen on in --wireless mode (default: 9876)")
    args = parser.parse_args()

    scenario = "B — Wireless (RPi → Jetson)" if args.wireless else "A — Direct"
    input_label = f"wireless TCP :{args.port}" if args.wireless else (
        f"real images from {args.images}" if args.images else "synthetic frames (seed=42)"
    )
    print("================================================================================")
    print("       YOLO INFERENCE BENCHMARK — NVIDIA JETSON ORIN NANO (HEADLESS)            ")
    print("================================================================================")
    print(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Scenario:  {scenario}")
    print(f" Protocol:  {args.runs} trials × {args.frames} frames ({args.runs * args.frames} total/model)")
    print(f" Warm-up:   {args.warmup} frames discarded")
    print(f" Input:     {input_label}")
    print(f" Hardware:  tegrastats @ {args.interval}ms")
    print(f" TensorRT:  {'available ✅' if TRT_AVAILABLE else 'unavailable ❌ (FP16/INT8 will fail)'}")
    print("================================================================================")

    results = []

    def _run_models(wireless_rcv=None):
        for model_info in MODELS:
            if model_info.get("skip"):
                print(f"\n⚠️ Skipping {model_info['name']} [{model_info['quant']}] — {model_info.get('skip_reason', 'Unsupported')}")
                continue
            check_path = model_info.get("ort_path") or model_info.get("fp32_path")
            if check_path and not check_path.exists():
                print(f"⚠️ Source ONNX not found: {check_path}")
                continue
            if model_info["backend"] == "trt" and not TRT_AVAILABLE:
                print(f"⚠️ Skipping {model_info['name']} [{model_info['quant']}] — TRT not available")
                continue
            try:
                res = benchmark_model(
                    model_info,
                    n_frames=args.frames,
                    n_runs=args.runs,
                    n_warmup=args.warmup,
                    interval_ms=args.interval,
                    images_dir=args.images,
                    wireless_receiver=wireless_rcv,
                )
                results.append(res)
            except Exception as e:
                import traceback
                print(f"\n❌ Error: {model_info['name']} [{model_info['quant']}]: {e}")
                traceback.print_exc()

    if args.wireless:
        print(f"\n🌐 Scenario B — Wireless mode (TCP port {args.port})")
        print("   Start rpi_sender.py on the RPi and enter THIS machine's IP there.")
        with WirelessReceiver(port=args.port) as rcv:
            print(f"  [NET] Measuring baseline RTT before benchmark...")
            rcv.measure_rtt(n_pings=20)
            _run_models(wireless_rcv=rcv)
    else:
        _run_models(wireless_rcv=None)

    # ── Summary Tables ────────────────────────────────────────────────────────
    print("\n" + "="*95)
    print("                             FINAL SUMMARY METRICS TABLE                         ")
    print("="*95)
    header = f"{'Model':<8} {'Quant':<6} {'Size(KB)':<9} {'Mean±SD (ms)':<16} {'95% CI (ms)':<16} {'P95(ms)':<8} {'P99(ms)':<8} {'FPS':<7} {'GPU%':<6} {'Pwr(W)':<6}"
    print(header)
    print("-" * 95)
    for r in results:
        st = r["stats"]
        teg = r["resources"]
        pwr = r["energy"]["avg_power_w"]
        gpu = teg.get("gpu_pct", {}).get("avg", 0.0)
        ci_str = f"[{st['ci_lo_ms']:.1f}, {st['ci_hi_ms']:.1f}]"
        mean_sd_str = f"{st['mean_ms']:.2f} ± {st['sd_ms']:.2f}"
        print(f"{r['model']:<8} {r['quant']:<6} {r['size_kb']:<9.0f} {mean_sd_str:<16} {ci_str:<16} {st['p95_ms']:<8.2f} {st['p99_ms']:<8.2f} {st['avg_fps']:<7.1f} {gpu:<6.1f} {pwr:<6.2f}")
    print("="*95)

    # ── Save CSV and JSON ─────────────────────────────────────────────────────
    json_path = BASE_DIR / "benchmark_results.json"
    csv_path = BASE_DIR / "benchmark_results.csv"

    # Clean raw series before saving compact JSON
    clean_results = []
    for r in results:
        rc = dict(r)
        rc.pop("raw_latencies", None)
        clean_results.append(rc)

    with open(json_path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": clean_results}, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Quantization", "Input_Resolution", "Size_KB", "Provider",
            "Mean_Latency_ms", "SD_ms", "CI_95_Lo_ms", "CI_95_Hi_ms",
            "Median_ms", "IQR_ms", "P90_ms", "P95_ms", "P99_ms", "Min_ms", "Max_ms",
            "CV_Percent", "Avg_FPS", "First_Frame_ms", "Cold_Start_TTFF_ms",
            "E2E_Mean_ms", "Net_Transit_ms", "JPEG_Decomp_ms", "Preproc_ms",
            "DMR_33ms_pct", "DMR_50ms_pct", "DMR_100ms_pct", "DMR_200ms_pct", "Goodput_Mbps",
            "GPU_Avg_Pct", "GPU_Peak_Pct", "CPU_Avg_Pct",
            "Temp_GPU_C", "Temp_Tj_C", "Power_Total_W", "FPS_per_Watt", "mJ_per_Infer"
        ])
        for r in results:
            s = r["stats"]
            teg = r["resources"]
            e = r["energy"]
            sd = r.get("stage_decomposition") or {}
            dmr = sd.get("deadline_miss_rates") or {}
            ff = r.get("first_frame", {})
            writer.writerow([
                r["model"], r["quant"], r["input_res"], f"{r['size_kb']:.1f}", r["provider"],
                f"{s['mean_ms']:.3f}", f"{s['sd_ms']:.3f}", f"{s['ci_lo_ms']:.3f}", f"{s['ci_hi_ms']:.3f}",
                f"{s['median_ms']:.3f}", f"{s['iqr_ms']:.3f}", f"{s['p90_ms']:.3f}", f"{s['p95_ms']:.3f}", f"{s['p99_ms']:.3f}",
                f"{s['min_ms']:.3f}", f"{s['max_ms']:.3f}", f"{s['cv_pct']:.2f}", f"{s['avg_fps']:.2f}",
                f"{ff.get('total_ms', 0.0):.2f}",
                f"{ff.get('cold_start_ttff_ms', 0.0):.2f}",
                f"{sd.get('total_e2e',{}).get('mean_ms', 0.0):.2f}" if sd else "N/A",
                f"{sd.get('network_transit',{}).get('mean_ms', 0.0):.2f}" if sd else "N/A",
                f"{sd.get('jpeg_decompress',{}).get('mean_ms', 0.0):.2f}" if sd else "N/A",
                f"{sd.get('tensor_preprocess',{}).get('mean_ms', 0.0):.2f}" if sd else "N/A",
                f"{dmr.get('dmr_33ms_pct', 0.0):.1f}" if sd else "N/A",
                f"{dmr.get('dmr_50ms_pct', 0.0):.1f}" if sd else "N/A",
                f"{dmr.get('dmr_100ms_pct', 0.0):.1f}" if sd else "N/A",
                f"{dmr.get('dmr_200ms_pct', 0.0):.1f}" if sd else "N/A",
                f"{sd.get('network_goodput_mbps', 0.0):.2f}" if sd else "N/A",
                f"{teg.get('gpu_pct',{}).get('avg',0.0):.1f}", f"{teg.get('gpu_pct',{}).get('peak',0.0):.1f}",
                f"{teg.get('cpu_avg_pct',{}).get('avg',0.0):.1f}",
                f"{teg.get('temp_gpu_c',{}).get('avg',0.0):.1f}", f"{teg.get('temp_tj_c',{}).get('avg',0.0):.1f}",
                f"{e['avg_power_w']:.3f}", f"{e['fps_per_watt']:.2f}", f"{e['mj_per_infer']:.3f}"
            ])

    print(f"\n📁 Saved complete JSON to: {json_path}")
    print(f"📁 Saved complete CSV to:  {csv_path}\n")

if __name__ == "__main__":
    main()

