"""
benchmark.py — YOLO Inference Benchmark Suite (Scientific Edition)
====================================================================
Rigorous performance evaluation for Scientific Reports (Q1 journal).

Two experimental scenarios
───────────────────────────
  Scenario A  Direct     Camera → USB/CSI → Jetson → infer
  Scenario B  Wireless   Camera → RPi → WiFi → Jetson → infer

Statistical methodology
────────────────────────
  • N_WARMUP discarded frames (JIT/cache warm-up)
  • N_FRAMES measurement frames per trial
  • K independent trials per model (multi-run protocol)
  • Statistics: mean, SD (ddof=1), SEM, 95 % CI (t-distribution),
    bootstrap CI (B=2000), median, Q1/Q3, IQR, P10/P90/P95/P99,
    CV, skewness, kurtosis, Shapiro–Wilk normality test,
    Grubbs outlier count, between-trial variance
  • Model comparison: Welch's t-test, Mann–Whitney U, Cohen's d

Hardware monitoring (Jetson Orin)
───────────────────────────────────
  tegrastats (100 ms polling):
    CPU load per core, CPU frequency, GPU load, RAM, SWAP,
    temperatures (cpu, gpu, soc0-2, tj), power (VDD_IN, VDD_CPU_GPU_CV, VDD_SOC)

Run:  streamlit run benchmark.py
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import csv, io, json, os, re, socket, struct, subprocess, sys
import threading, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import stats_utils as sp_stats   # pure numpy — no scipy binary needed

sys.path.insert(0, str(Path(__file__).parent))
from frame_protocol import (
    recv_packet, decode_frame, handle_ping,
    measure_rtt, recv_exact,
    DEFAULT_PORT, PING_SIZE, PING_FMT,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLO Benchmark — Scientific",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp{background:#0d1117;}
[data-testid="stSidebar"]{background:#161b22;border-right:1px solid #30363d;}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;
      padding:16px;margin:6px 0;}
.info-row{display:flex;justify-content:space-between;padding:5px 0;
           border-bottom:1px solid #21262d;font-size:0.84em;}
.info-row:last-child{border:none;}
.info-row .k{color:#8b949e;} .info-row .v{color:#c9d1d9;font-weight:600;}
.section{font-size:1em;font-weight:700;color:#58a6ff;margin:14px 0 6px;
          border-bottom:1px solid #21262d;padding-bottom:4px;}
.tag{display:inline-block;padding:2px 10px;border-radius:12px;
     font-size:0.78em;font-weight:700;}
.tag-fp32{background:#1f3a5f;color:#58a6ff;}
.tag-fp16{background:#1f3a2f;color:#3fb950;}
.tag-int8{background:#3d2600;color:#f78166;}
.tag-direct{background:#0f2d1f;color:#39d353;}
.tag-wireless{background:#1f1a3f;color:#a371f7;}
.warn{background:#2d1f00;border-left:4px solid #e3b341;padding:10px 14px;
      border-radius:6px;color:#e3b341;font-size:0.85em;margin:8px 0;}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

MODELS = {
    "YOLOv8": {
        "fp32": BASE_DIR / "yolov8/best_fp32.onnx",
        "fp16": BASE_DIR / "yolov8/best_fp16.onnx",
        "int8": BASE_DIR / "yolov8/best_int8.onnx",
        "input_size": (640, 640),
    },
    "YOLOv11": {
        "fp32": BASE_DIR / "yolov11/best_fp32.onnx",
        "fp16": BASE_DIR / "yolov11/best_fp16.onnx",
        "int8": BASE_DIR / "yolov11/best_int8 (1).onnx",
        "input_size": (576, 576),
    },
}

GPU_SYSFS   = "/sys/devices/platform/bus@0/17000000.gpu/load"
BOOTSTRAP_B = 2000
CONFIDENCE  = 0.95


# ══════════════════════════════════════════════════════════════════════════════
#  System snapshot
# ══════════════════════════════════════════════════════════════════════════════
def get_system_info() -> dict:
    def _read(path, default="N/A"):
        try:
            return Path(path).read_text().strip()
        except Exception:
            return default

    def _cmd(cmd, default="N/A"):
        try:
            return subprocess.check_output(cmd, shell=True, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return default

    # JetPack / L4T
    l4t = _read("/etc/nv_tegra_release", "N/A")
    revision = re.search(r"REVISION:\s*([\d.]+)", l4t)
    l4t_rev  = revision.group(1) if revision else "N/A"

    return {
        "device":        _read("/proc/device-tree/model", "Unknown").rstrip("\x00"),
        "l4t_revision":  l4t_rev,
        "os":            _cmd("uname -srm"),
        "python":        sys.version.split()[0],
        "ort_version":   _cmd("python3 -c 'import onnxruntime; print(onnxruntime.__version__)'"),
        "cv2_version":   _cmd("python3 -c 'import cv2; print(cv2.__version__)'"),
        "scipy_version": _cmd("python3 -c 'import scipy; print(scipy.__version__)'"),
        "numpy_version": np.__version__,
        "cpu_cores":     _cmd("nproc"),
        "ram_total_mb":  f"{int(_cmd('grep MemTotal /proc/meminfo | awk {print $2}', '0')) // 1024} MB",
        "timestamp":     datetime.now().isoformat(timespec="seconds"),
        "hostname":      _cmd("hostname"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Tegrastats Monitor
# ══════════════════════════════════════════════════════════════════════════════
_TGRE = re.compile(
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

def _parse_teg(line: str) -> dict | None:
    m = _TGRE.search(line)
    if not m:
        return None
    (ram_u, ram_t, swp_u, swp_t,
     cpu_str, gpu_pct,
     t_cpu, t_gpu, t_tj,
     p_in, p_cg, p_soc) = m.groups()

    cpu_loads, cpu_freqs = [], []
    for c in cpu_str.split(","):
        p = c.split("@")
        try:
            cpu_loads.append(int(p[0].replace("%", "")))
            if len(p) > 1:
                cpu_freqs.append(int(p[1]))
        except ValueError:
            pass

    return {
        "ram_used_mb":      int(ram_u),
        "ram_total_mb":     int(ram_t),
        "ram_pct":          int(ram_u) / int(ram_t) * 100,
        "swap_used_mb":     int(swp_u),
        "gpu_pct":          float(gpu_pct),
        "cpu_loads":        cpu_loads,
        "cpu_avg_pct":      sum(cpu_loads) / len(cpu_loads) if cpu_loads else 0.0,
        "cpu_freqs_mhz":    cpu_freqs,
        "cpu_avg_freq_mhz": sum(cpu_freqs) / len(cpu_freqs) if cpu_freqs else 0.0,
        "temp_cpu_c":       float(t_cpu),
        "temp_gpu_c":       float(t_gpu),
        "temp_tj_c":        float(t_tj),
        "power_total_mw":   float(p_in),
        "power_cpu_gpu_mw": float(p_cg),
        "power_soc_mw":     float(p_soc),
    }


class TegrastatsMonitor:
    def __init__(self, interval_ms: int = 100):
        self._interval  = interval_ms
        self.samples:   list[dict] = []
        self._stop      = threading.Event()
        self._proc      = None
        self._thread    = None

    def start(self):
        self._stop.clear()
        self._proc = subprocess.Popen(
            ["/usr/bin/tegrastats", "--interval", str(self._interval)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            s = _parse_teg(line)
            if s:
                self.samples.append(s)

    def stop(self):
        self._stop.set()
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                pass

    @staticmethod
    def _arr_stats(vals: list) -> dict:
        if not vals:
            return {"min": None, "max": None, "avg": None, "std": None, "peak": None}
        a = np.array(vals, dtype=float)
        return {
            "min":  float(a.min()),
            "max":  float(a.max()),
            "avg":  float(a.mean()),
            "std":  float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "peak": float(a.max()),
        }

    def results(self) -> dict:
        s = self.samples
        if not s:
            return {}
        return {
            "n_samples":        len(s),
            "gpu_pct":          self._arr_stats([x["gpu_pct"]          for x in s]),
            "cpu_avg_pct":      self._arr_stats([x["cpu_avg_pct"]      for x in s]),
            "cpu_avg_freq_mhz": self._arr_stats([x["cpu_avg_freq_mhz"] for x in s]),
            "ram_used_mb":      self._arr_stats([x["ram_used_mb"]      for x in s]),
            "ram_pct":          self._arr_stats([x["ram_pct"]          for x in s]),
            "swap_used_mb":     self._arr_stats([x["swap_used_mb"]     for x in s]),
            "temp_cpu_c":       self._arr_stats([x["temp_cpu_c"]       for x in s]),
            "temp_gpu_c":       self._arr_stats([x["temp_gpu_c"]       for x in s]),
            "temp_tj_c":        self._arr_stats([x["temp_tj_c"]        for x in s]),
            "power_total_mw":   self._arr_stats([x["power_total_mw"]   for x in s]),
            "power_cpu_gpu_mw": self._arr_stats([x["power_cpu_gpu_mw"] for x in s]),
            "power_soc_mw":     self._arr_stats([x["power_soc_mw"]     for x in s]),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Statistical Analysis
# ══════════════════════════════════════════════════════════════════════════════
def compute_statistics(arr_ms: list, confidence: float = CONFIDENCE) -> dict:
    """
    Compute comprehensive descriptive and inferential statistics.
    Uses scipy where available for exact t/Shapiro-Wilk results.
    """
    a    = np.array(arr_ms, dtype=np.float64)
    n    = len(a)
    mean = float(a.mean())
    sd   = float(a.std(ddof=1))
    sem  = sd / np.sqrt(n)
    cv   = (sd / mean * 100) if mean != 0 else float("nan")

    # Parametric CI (t-distribution)
    t_crit = float(sp_stats.t_ppf((1 + confidence) / 2, df=n - 1))
    ci_lo  = mean - t_crit * sem
    ci_hi  = mean + t_crit * sem

    # Bootstrap CI (bias-corrected percentile)
    rng         = np.random.default_rng(42)
    boot_means  = [rng.choice(a, size=n, replace=True).mean()
                   for _ in range(BOOTSTRAP_B)]
    boot_ci_lo  = float(np.percentile(boot_means, (1 - confidence) / 2 * 100))
    boot_ci_hi  = float(np.percentile(boot_means, (1 + confidence) / 2 * 100))

    # Quartiles & percentiles
    q1, med, q3 = float(np.percentile(a, 25)), float(np.median(a)), float(np.percentile(a, 75))
    iqr          = q3 - q1

    # Outliers (Tukey fences)
    fence_lo     = q1 - 1.5 * iqr
    fence_hi     = q3 + 1.5 * iqr
    n_outliers   = int(np.sum((a < fence_lo) | (a > fence_hi)))

    # Shapiro–Wilk (limited to 5000 samples per scipy docs)
    sw_sample    = a[:5000] if n > 5000 else a
    sw_stat, sw_p = sp_stats.shapiro(sw_sample)
    is_normal    = bool(sw_p > 0.05)

    # Higher moments
    skew = float(sp_stats.skew(a))
    kurt = float(sp_stats.kurtosis(a, fisher=True))

    # FPS (infer-only)
    fps = 1000.0 / mean if mean > 0 else 0.0

    return {
        "n":              n,
        "mean_ms":        mean,
        "sd_ms":          sd,
        "sem_ms":         sem,
        "cv_pct":         cv,
        "ci_lo_ms":       ci_lo,
        "ci_hi_ms":       ci_hi,
        "ci_margin_ms":   t_crit * sem,
        "boot_ci_lo_ms":  boot_ci_lo,
        "boot_ci_hi_ms":  boot_ci_hi,
        "min_ms":         float(a.min()),
        "max_ms":         float(a.max()),
        "median_ms":      med,
        "q1_ms":          q1,
        "q3_ms":          q3,
        "iqr_ms":         iqr,
        "p10_ms":         float(np.percentile(a, 10)),
        "p90_ms":         float(np.percentile(a, 90)),
        "p95_ms":         float(np.percentile(a, 95)),
        "p99_ms":         float(np.percentile(a, 99)),
        "n_outliers":     n_outliers,
        "sw_stat":        float(sw_stat),
        "sw_p":           float(sw_p),
        "is_normal":      is_normal,
        "skewness":       skew,
        "kurtosis":       kurt,
        "avg_fps":        fps,
    }


def compare_two(a_ms: list, b_ms: list, label_a: str, label_b: str) -> dict:
    """Welch t-test + Mann-Whitney U + Cohen's d between two latency distributions."""
    a, b = np.array(a_ms, dtype=float), np.array(b_ms, dtype=float)

    t_stat, t_p = sp_stats.ttest_ind(a, b, equal_var=False)
    u_stat, u_p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")

    pooled_sd = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    cohens_d  = (a.mean() - b.mean()) / pooled_sd if pooled_sd != 0 else 0.0

    magnitude = (
        "negligible" if abs(cohens_d) < 0.2 else
        "small"      if abs(cohens_d) < 0.5 else
        "medium"     if abs(cohens_d) < 0.8 else
        "large"
    )

    return {
        "label_a":    label_a,
        "label_b":    label_b,
        "mean_a":     float(a.mean()),
        "mean_b":     float(b.mean()),
        "delta_ms":   float(a.mean() - b.mean()),
        "delta_pct":  float((a.mean() - b.mean()) / b.mean() * 100) if b.mean() else 0,
        "welch_t":    float(t_stat),
        "welch_p":    float(t_p),
        "significant":bool(t_p < 0.05),
        "mw_u":       float(u_stat),
        "mw_p":       float(u_p),
        "cohens_d":   float(cohens_d),
        "effect_mag": magnitude,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Inference Engine
# ══════════════════════════════════════════════════════════════════════════════
class InferenceEngine:
    def __init__(self, model_name: str, precision: str):
        import onnxruntime as ort
        cfg        = MODELS[model_name]
        self.W, self.H = cfg["input_size"]
        path       = str(cfg[precision])
        sess_opts  = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess  = ort.InferenceSession(
            path, sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.inp   = self.sess.get_inputs()[0].name
        self.out   = self.sess.get_outputs()[0].name
        self.prov  = self.sess.get_providers()[0].replace("ExecutionProvider", "")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.W, self.H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.expand_dims(img.transpose(2, 0, 1), 0)

    def infer_raw(self, blob: np.ndarray) -> tuple[object, float]:
        t0  = time.perf_counter()
        out = self.sess.run([self.out], {self.inp: blob})
        ms  = (time.perf_counter() - t0) * 1000
        return out, ms


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark — Scenario A (Direct)
# ══════════════════════════════════════════════════════════════════════════════
def run_direct_trial(
    engine:     InferenceEngine,
    n_warmup:   int,
    n_frames:   int,
    use_camera: bool,
    cam_idx:    int,
    teg:        TegrastatsMonitor,
    progress_cb = None,
) -> dict:
    """Single trial for Scenario A (direct camera → Jetson)."""
    cap = cv2.VideoCapture(int(cam_idx)) if use_camera else None
    if cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_frame():
        if cap:
            ret, f = cap.read()
            return f if ret else np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # ── Warm-up ────────────────────────────────
    for _ in range(n_warmup):
        engine.infer_raw(engine.preprocess(get_frame()))

    # ── First-frame latency ────────────────────
    t_ff0      = time.perf_counter()
    f0         = get_frame()
    t_ff1      = time.perf_counter()
    blob0      = engine.preprocess(f0)
    t_ff2      = time.perf_counter()
    engine.infer_raw(blob0)
    t_ff3      = time.perf_counter()

    ff_cap_ms  = (t_ff1 - t_ff0) * 1000
    ff_pre_ms  = (t_ff2 - t_ff1) * 1000
    ff_inf_ms  = (t_ff3 - t_ff2) * 1000
    ff_tot_ms  = (t_ff3 - t_ff0) * 1000

    # ── Benchmark loop ─────────────────────────
    inf_ms_list:  list[float] = []
    pre_ms_list:  list[float] = []
    cap_ms_list:  list[float] = []
    pipe_ms_list: list[float] = []

    t_total_0 = time.perf_counter()
    teg.start()

    for i in range(n_frames):
        t_cap0 = time.perf_counter()
        frame  = get_frame()
        t_cap1 = time.perf_counter()
        blob   = engine.preprocess(frame)
        t_pre1 = time.perf_counter()
        engine.infer_raw(blob)
        t_inf1 = time.perf_counter()

        cap_ms_list.append((t_cap1 - t_cap0) * 1000)
        pre_ms_list.append((t_pre1 - t_cap1) * 1000)
        inf_ms_list.append((t_inf1 - t_pre1) * 1000)
        pipe_ms_list.append((t_inf1 - t_cap0) * 1000)

        if progress_cb:
            progress_cb(i + 1, n_frames)

    teg.stop()
    total_ms = (time.perf_counter() - t_total_0) * 1000
    if cap:
        cap.release()

    return {
        "scenario":    "Direct",
        "ff_cap_ms":   ff_cap_ms,
        "ff_pre_ms":   ff_pre_ms,
        "ff_inf_ms":   ff_inf_ms,
        "ff_tot_ms":   ff_tot_ms,
        "total_ms":    total_ms,
        "inf_ms":      inf_ms_list,
        "pre_ms":      pre_ms_list,
        "cap_ms":      cap_ms_list,
        "pipe_ms":     pipe_ms_list,
        "resources":   teg.results(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark — Scenario B (Wireless)
# ══════════════════════════════════════════════════════════════════════════════
def run_wireless_trial(
    engine:    InferenceEngine,
    conn_sock: socket.socket,
    n_warmup:  int,
    n_frames:  int,
    teg:       TegrastatsMonitor,
    progress_cb = None,
) -> dict:
    """Single trial for Scenario B (RPi → WiFi → Jetson)."""

    # ── RTT / network latency measurement ─────
    rtt_stats = measure_rtt(conn_sock, n_pings=20)

    def recv_and_infer():
        packet, recv_ts = recv_packet(conn_sock)
        if packet is None:
            return None
        t_recv     = time.perf_counter()
        frame, frame_num, send_ts_us, w, h, _ = decode_frame(packet)
        t_dec      = time.perf_counter()
        blob       = engine.preprocess(frame)
        t_pre      = time.perf_counter()
        engine.infer_raw(blob)
        t_inf      = time.perf_counter()
        return {
            "recv_ts_us":  recv_ts,
            "send_ts_us":  send_ts_us,
            "dec_ms":      (t_dec - t_recv) * 1000,
            "pre_ms":      (t_pre - t_dec)  * 1000,
            "inf_ms":      (t_inf - t_pre)  * 1000,
            "pipe_ms":     (t_inf - t_recv) * 1000,
            "total_ms":    (t_inf - t_recv) * 1000 + rtt_stats["one_way_est_ms"],
        }

    # ── Warm-up ────────────────────────────────
    for _ in range(n_warmup):
        r = recv_and_infer()
        if r is None:
            break

    # ── First frame ────────────────────────────
    ff = recv_and_infer()
    if ff is None:
        return {"error": "No frames from RPi"}

    # ── Benchmark loop ─────────────────────────
    dec_ms_list:   list[float] = []
    pre_ms_list:   list[float] = []
    inf_ms_list:   list[float] = []
    pipe_ms_list:  list[float] = []
    total_ms_list: list[float] = []
    pkt_sizes:     list[int]   = []

    t_total_0 = time.perf_counter()
    teg.start()

    for i in range(n_frames):
        r = recv_and_infer()
        if r is None:
            break
        dec_ms_list.append(r["dec_ms"])
        pre_ms_list.append(r["pre_ms"])
        inf_ms_list.append(r["inf_ms"])
        pipe_ms_list.append(r["pipe_ms"])
        total_ms_list.append(r["total_ms"])
        if progress_cb:
            progress_cb(i + 1, n_frames)

    teg.stop()
    session_ms = (time.perf_counter() - t_total_0) * 1000

    return {
        "scenario":        "Wireless",
        "rtt":             rtt_stats,
        "net_latency_est": rtt_stats["one_way_est_ms"],
        "ff_tot_ms":       ff["total_ms"],
        "ff_dec_ms":       ff["dec_ms"],
        "ff_pre_ms":       ff["pre_ms"],
        "ff_inf_ms":       ff["inf_ms"],
        "session_ms":      session_ms,
        "inf_ms":          inf_ms_list,
        "pre_ms":          pre_ms_list,
        "dec_ms":          dec_ms_list,
        "pipe_ms":         pipe_ms_list,
        "total_ms":        total_ms_list,
        "resources":       teg.results(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-run orchestrator
# ══════════════════════════════════════════════════════════════════════════════
def run_full_benchmark(
    model_name:  str,
    precision:   str,
    scenario:    str,            # "Direct" | "Wireless"
    n_runs:      int,
    n_warmup:    int,
    n_frames:    int,
    use_camera:  bool = True,
    cam_idx:     int  = 0,
    conn_sock:   socket.socket | None = None,
    progress_cb  = None,
) -> dict:
    """
    Run K independent trials and aggregate statistics.
    Idle 2 s between trials to reduce thermal carry-over.
    """
    import onnxruntime as ort

    cfg       = MODELS[model_name]
    path      = str(cfg[precision])
    t_load_0  = time.perf_counter()
    engine    = InferenceEngine(model_name, precision)
    load_ms   = (time.perf_counter() - t_load_0) * 1000

    trials:    list[dict] = []
    all_inf:   list[float] = []

    for k in range(n_runs):
        teg = TegrastatsMonitor(interval_ms=100)

        if scenario == "Direct":
            trial = run_direct_trial(
                engine, n_warmup, n_frames, use_camera, cam_idx, teg,
                lambda d, t: progress_cb(k, n_runs, d, t) if progress_cb else None,
            )
        else:
            if conn_sock is None:
                return {"error": "No RPi connection for Wireless scenario"}
            trial = run_wireless_trial(
                engine, conn_sock, n_warmup, n_frames, teg,
                lambda d, t: progress_cb(k, n_runs, d, t) if progress_cb else None,
            )

        trial["run_index"] = k
        trials.append(trial)
        all_inf.extend(trial.get("inf_ms", []))

        if k < n_runs - 1:
            time.sleep(2)      # cool-down between runs

    # ── Aggregate statistics ────────────────────────────────────────
    # Pool all inference times across runs (primary metric)
    inf_stats  = compute_statistics(all_inf)

    # Per-component stats (mean across runs)
    def pool(key):
        pooled = []
        for t in trials:
            pooled.extend(t.get(key, []))
        return pooled

    # Between-run variance of per-run means
    run_means  = [float(np.mean(t["inf_ms"])) for t in trials if t.get("inf_ms")]
    run_mean   = float(np.mean(run_means))
    run_std    = float(np.std(run_means, ddof=1)) if len(run_means) > 1 else 0.0

    # Resource aggregation across all runs
    def agg_res(key):
        vals = []
        for t in trials:
            res = t.get("resources", {})
            s   = res.get(key, {})
            if s.get("avg") is not None:
                vals.append(s["avg"])
        if not vals:
            return {"min": None, "max": None, "avg": None, "peak": None}
        a = np.array(vals)
        return {"min": float(a.min()), "max": float(a.max()),
                "avg": float(a.mean()), "peak": float(a.max())}

    res_keys = ["gpu_pct", "cpu_avg_pct", "ram_used_mb", "ram_pct",
                "temp_cpu_c", "temp_gpu_c", "temp_tj_c",
                "power_total_mw", "power_cpu_gpu_mw", "power_soc_mw"]

    resources = {k: agg_res(k) for k in res_keys}

    # Power-related derived metrics
    avg_power_w = (resources["power_total_mw"]["avg"] or 0) / 1000
    avg_fps     = inf_stats["avg_fps"]
    fps_per_w   = avg_fps / avg_power_w if avg_power_w > 0 else 0
    mj_per_inf  = (avg_power_w * inf_stats["mean_ms"]) if avg_power_w > 0 else 0

    # First-frame across runs
    ff_ms_all  = [t.get("ff_tot_ms", 0) for t in trials]

    return {
        # Identity
        "model":          model_name,
        "precision":      precision,
        "quantization":   precision,
        "scenario":       scenario,
        "provider":       engine.prov,
        "timestamp":      datetime.now().isoformat(timespec="seconds"),

        # Protocol
        "n_runs":         n_runs,
        "n_warmup":       n_warmup,
        "n_frames_trial": n_frames,
        "n_frames_total": len(all_inf),

        # Model file
        "size_kb":        Path(path).stat().st_size / 1024,
        "size_mb":        Path(path).stat().st_size / 1024 / 1024,
        "input_res":      f"{engine.W}×{engine.H}",
        "model_load_ms":  load_ms,

        # Core inference statistics (pooled across all runs)
        "inf":            inf_stats,

        # Pipeline component stats
        "pre":            compute_statistics(pool("pre_ms")) if pool("pre_ms") else {},
        "dec":            compute_statistics(pool("dec_ms")) if scenario == "Wireless" and pool("dec_ms") else None,
        "pipe":           compute_statistics(pool("pipe_ms")) if pool("pipe_ms") else {},
        "total_e2e":      compute_statistics(pool("total_ms")) if pool("total_ms") else {},

        # Between-run reproducibility
        "run_means_ms":   run_means,
        "run_mean_ms":    run_mean,
        "run_std_ms":     run_std,

        # First-frame latency
        "ff_ms":          {
            "mean": float(np.mean(ff_ms_all)),
            "std":  float(np.std(ff_ms_all, ddof=1)) if len(ff_ms_all) > 1 else 0.0,
            "min":  float(np.min(ff_ms_all)),
            "max":  float(np.max(ff_ms_all)),
            "all":  ff_ms_all,
        },

        # Wireless-only network metrics
        "rtt":            trials[0].get("rtt") if scenario == "Wireless" else None,
        "net_latency_est_ms": trials[0].get("net_latency_est") if scenario == "Wireless" else None,

        # Resources (averaged across runs)
        "resources":      resources,

        # Derived / efficiency
        "avg_power_w":    avg_power_w,
        "fps_per_watt":   fps_per_w,
        "mj_per_infer":   mj_per_inf,

        # Raw series (first run, for plotting)
        "inf_series":     trials[0].get("inf_ms", []) if trials else [],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  LaTeX Table Generator
# ══════════════════════════════════════════════════════════════════════════════




# ══════════════════════════════════════════════════════════════════════════════
#  UI Helpers
# ══════════════════════════════════════════════════════════════════════════════
def tag(precision: str) -> str:
    c = {"fp32": "tag-fp32", "fp16": "tag-fp16", "int8": "tag-int8"}.get(precision, "")
    return f'<span class="tag {c}">{precision}</span>'

def sc_tag(scenario: str) -> str:
    c = "tag-direct" if scenario == "Direct" else "tag-wireless"
    return f'<span class="tag {c}">{scenario}</span>'

def fmt(v, u="", d=2):
    return f"{v:.{d}f} {u}".strip() if v is not None else "—"

def irow(k, v):
    return f'<div class="info-row"><span class="k">{k}</span><span class="v">{v}</span></div>'

def render_stat_block(s: dict, title: str) -> str:
    if not s:
        return ""
    return f"""
    <div class="section">{title}</div>
    {irow("Mean ± SD (ms)",  f"{s.get('mean_ms',0):.3f} ± {s.get('sd_ms',0):.3f}")}
    {irow("95% CI (ms)",     f"[{s.get('ci_lo_ms',0):.3f}, {s.get('ci_hi_ms',0):.3f}]")}
    {irow("Bootstrap 95% CI", f"[{s.get('boot_ci_lo_ms',0):.3f}, {s.get('boot_ci_hi_ms',0):.3f}]")}
    {irow("SEM (ms)",         fmt(s.get('sem_ms')))}
    {irow("CV (%)",           f"{s.get('cv_pct',0):.2f}")  }
    {irow("Min / Median / Max", f"{s.get('min_ms',0):.2f} / {s.get('median_ms',0):.2f} / {s.get('max_ms',0):.2f}")}
    {irow("Q1 / Q3 / IQR",   f"{s.get('q1_ms',0):.2f} / {s.get('q3_ms',0):.2f} / {s.get('iqr_ms',0):.2f}")}
    {irow("P95 / P99 (ms)",  f"{s.get('p95_ms',0):.2f} / {s.get('p99_ms',0):.2f}")}
    {irow("Skewness",         fmt(s.get('skewness'), d=3))}
    {irow("Kurtosis",         fmt(s.get('kurtosis'), d=3))}
    {irow("Shapiro–Wilk p",  f"{s.get('sw_p',0):.4f}  ({'normal ✓' if s.get('is_normal') else 'non-normal ✗'})")}
    {irow("Outliers (IQR)",   str(s.get('n_outliers', '—')))}
    {irow("Avg FPS",          f"{s.get('avg_fps',0):.2f}")}
    """

def render_result_card(r: dict) -> str:
    inf   = r.get("inf", {})
    res   = r.get("resources", {})
    rtt   = r.get("rtt")
    ff    = r.get("ff_ms", {})

    def res_row(label, key):
        s = res.get(key, {})
        v = f"avg {fmt(s.get('avg'),d=1)} | min {fmt(s.get('min'),d=1)} | max {fmt(s.get('max'),d=1)} | peak {fmt(s.get('peak'),d=1)}"
        return irow(label, v if s.get('avg') is not None else "—")

    rtt_block = ""
    if rtt:
        rtt_block = f"""
        <div class="section">📡 Network (RTT Ping-Pong, n=20)</div>
        {irow("RTT mean ± std (ms)", f"{rtt['rtt_mean_ms']:.3f} ± {rtt['rtt_std_ms']:.3f}")}
        {irow("One-way est. (ms)",   f"{rtt['one_way_est_ms']:.3f}")}
        {irow("Jitter (ms)",         f"{rtt['jitter_ms']:.3f}")}
        {irow("RTT min / max (ms)",  f"{rtt['rtt_min_ms']:.3f} / {rtt['rtt_max_ms']:.3f}")}
        """

    return f"""
    <div class="card">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
        <b style="font-size:1.1em;">{r['model']}</b>
        {tag(r['precision'])} {sc_tag(r.get('scenario',''))}
        <span style="color:#8b949e;font-size:0.78em;margin-left:auto;">{r.get('timestamp','')}</span>
      </div>

      <div class="section">📁 Model</div>
      {irow("File size",        f"{r['size_kb']:.1f} KB ({r['size_mb']:.3f} MB)")}
      {irow("Quantization",     r['precision'])}
      {irow("Input resolution", r.get('input_res','—'))}
      {irow("Provider",         r.get('provider','—'))}
      {irow("Load time (ms)",   fmt(r.get('model_load_ms'), d=1))}
      {irow("Trials × frames",  f"{r.get('n_runs','?')} × {r.get('n_frames_trial','?')} = {r.get('n_frames_total','?')} total")}

      {render_stat_block(inf, "⚡ Inference Latency (ORT only)")}

      <div class="section">🔁 Between-Trial Reproducibility</div>
      {irow("Per-trial means (ms)", ", ".join(f"{v:.2f}" for v in r.get('run_means_ms',[])))}
      {irow("Grand mean ± SD (ms)", f"{r.get('run_mean_ms',0):.3f} ± {r.get('run_std_ms',0):.3f}")}

      <div class="section">⏱️ First-Frame Latency</div>
      {irow("Mean (ms)", fmt(ff.get('mean'), d=3))}
      {irow("SD (ms)",   fmt(ff.get('std'),  d=3))}

      {rtt_block}

      <div class="section">🖥️ Resources (across all trials)</div>
      {res_row("GPU %",          "gpu_pct")}
      {res_row("CPU avg %",      "cpu_avg_pct")}
      {res_row("RAM used (MB)",  "ram_used_mb")}
      {res_row("Temp CPU (°C)",  "temp_cpu_c")}
      {res_row("Temp GPU (°C)",  "temp_gpu_c")}
      {res_row("Temp Tj  (°C)",  "temp_tj_c")}
      {res_row("Power total (mW)","power_total_mw")}
      {res_row("Power CPU+GPU (mW)","power_cpu_gpu_mw")}

      <div class="section">⚡ Energy Efficiency</div>
      {irow("Avg total power (W)",  fmt(r.get('avg_power_w'), d=3))}
      {irow("FPS / Watt",           fmt(r.get('fps_per_watt'), d=2))}
      {irow("mJ / inference",       fmt(r.get('mj_per_infer'), d=3))}
    </div>"""


def build_summary_rows(results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        s = r.get("inf", {})
        ff = r.get("ff_ms", {})
        res = r.get("resources", {})
        rtt = r.get("rtt")
        rows.append({
            "Model":           r["model"],
            "Quant.":          r["precision"],
            "Scenario":        r.get("scenario", "—"),
            "Size (KB)":       f"{r['size_kb']:.0f}",
            "Input":           r.get("input_res", "—"),
            "N total":         str(s.get("n", "—")),
            "Mean±SD (ms)":    f"{s.get('mean_ms',0):.3f}±{s.get('sd_ms',0):.3f}",
            "95% CI (ms)":     f"[{s.get('ci_lo_ms',0):.3f},{s.get('ci_hi_ms',0):.3f}]",
            "CV (%)":          f"{s.get('cv_pct',0):.2f}",
            "P95 (ms)":        f"{s.get('p95_ms',0):.3f}",
            "P99 (ms)":        f"{s.get('p99_ms',0):.3f}",
            "Avg FPS":         f"{s.get('avg_fps',0):.2f}",
            "1st Frame (ms)":  f"{ff.get('mean',0):.2f}±{ff.get('std',0):.2f}",
            "Net est (ms)":    f"{r['net_latency_est_ms']:.2f}" if rtt else "—",
            "RTT (ms)":        f"{rtt['rtt_mean_ms']:.2f}±{rtt['rtt_std_ms']:.2f}" if rtt else "—",
            "Jitter (ms)":     f"{rtt['jitter_ms']:.2f}" if rtt else "—",
            "SW normal?":      "Yes" if s.get("is_normal") else "No",
            "SW p":            f"{s.get('sw_p',0):.4f}",
            "GPU avg %":       fmt(res.get("gpu_pct",{}).get("avg"), d=1),
            "GPU peak %":      fmt(res.get("gpu_pct",{}).get("peak"), d=1),
            "CPU avg %":       fmt(res.get("cpu_avg_pct",{}).get("avg"), d=1),
            "Temp GPU (°C)":   fmt(res.get("temp_gpu_c",{}).get("avg"), d=1),
            "Temp Tj (°C)":    fmt(res.get("temp_tj_c",{}).get("avg"), d=1),
            "Pwr total (W)":   fmt(r.get("avg_power_w"), d=3),
            "FPS/W":           fmt(r.get("fps_per_watt"), d=2),
            "mJ/infer":        fmt(r.get("mj_per_infer"), d=3),
            "Provider":        r.get("provider","—"),
            "Timestamp":       r.get("timestamp","—"),
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  Session State
# ══════════════════════════════════════════════════════════════════════════════
for k, v in [
    ("results",        []),
    ("system_info",    {}),
    ("comparisons",    []),
    ("wireless_sock",  None),
    ("wireless_conn",  None),
    ("wireless_ready", False),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 Benchmark Setup")
    st.divider()

    # ── System snapshot ─────────────────────────
    if st.button("📋 Capture System Snapshot", use_container_width=True):
        with st.spinner("Capturing…"):
            st.session_state.system_info = get_system_info()
        st.success("System info captured")

    st.divider()

    # ── Models ──────────────────────────────────
    st.markdown("### 🤖 Models")
    sel = {}
    for mn in ["YOLOv8", "YOLOv11"]:
        st.markdown(f"**{mn}**")
        c1, c2, c3 = st.columns(3)
        sel[f"{mn}_fp32"] = c1.checkbox("fp32", value=True, key=f"{mn}_fp32")
        sel[f"{mn}_fp16"] = c2.checkbox("fp16", value=True, key=f"{mn}_fp16")
        sel[f"{mn}_int8"] = c3.checkbox("int8", value=True, key=f"{mn}_int8")

    st.divider()

    # ── Protocol ────────────────────────────────
    st.markdown("### 🔬 Protocol")
    n_runs   = st.slider("Independent trials (K)", 1, 10, 3, 1,
                         help="More trials → lower between-run variance, longer runtime")
    n_warmup = st.slider("Warm-up frames",         5, 50,  10, 5)
    n_frames = st.slider("Frames per trial",       50, 500, 200, 50)
    st.caption(f"Total frames/model: {n_runs}×{n_frames} = **{n_runs*n_frames}**")

    st.divider()

    # ── Scenario ────────────────────────────────
    st.markdown("### 📡 Scenario")
    scenario = st.radio("", ["Direct (Camera → Jetson)", "Wireless (RPi → WiFi → Jetson)"],
                        label_visibility="collapsed")
    is_direct   = scenario.startswith("Direct")
    is_wireless = not is_direct

    if is_direct:
        use_camera = st.toggle("Use live camera", value=True)
        cam_idx    = st.selectbox("Camera index", [0, 1, 2], disabled=not use_camera)
    else:
        st.markdown("**Wireless Server**")
        server_port = st.number_input("Listen port", value=DEFAULT_PORT, min_value=1024)
        st.markdown(
            '<div class="warn">Start this server first, then connect RPi sender.</div>',
            unsafe_allow_html=True,
        )
        if st.button("🔌 Start TCP Server", use_container_width=True, type="primary"):
            try:
                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind(("0.0.0.0", int(server_port)))
                srv.listen(1)
                srv.settimeout(60)
                st.session_state.wireless_sock = srv
                st.success(f"Listening on :{server_port} — connect RPi now")
                conn, addr = srv.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                st.session_state.wireless_conn  = conn
                st.session_state.wireless_ready = True
                st.success(f"✅ RPi connected from {addr[0]}:{addr[1]}")
            except Exception as e:
                st.error(f"Server error: {e}")

        if st.session_state.wireless_ready:
            st.success("🟢 RPi connected")
        else:
            st.warning("⚪ Waiting for RPi…")

    st.divider()

    run_btn   = st.button("▶ Run Benchmark", type="primary", use_container_width=True)
    clear_btn = st.button("🗑 Clear Results",  use_container_width=True)
    comp_btn  = st.button("📊 Compute Comparisons", use_container_width=True)

    if clear_btn:
        st.session_state.results     = []
        st.session_state.comparisons = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🔬 YOLO Inference Benchmark — Scientific Reports")

tab_sys, tab_res, tab_summary, tab_compare, tab_export = st.tabs([
    "🖥️ System & Protocol",
    "🏁 Results",
    "📋 Summary Table",
    "📊 Statistical Comparison",
    "📥 Export",
])



# ─── Tab: System & Protocol ────────────────────────────────────────────────────
with tab_sys:
    st.markdown("#### 🖥️ System Snapshot")
    si = st.session_state.system_info
    if si:
        c1, c2 = st.columns(2)
        hw = {
            "Device":        si.get("device"),
            "L4T Revision":  si.get("l4t_revision"),
            "OS":            si.get("os"),
            "CPU cores":     si.get("cpu_cores"),
            "RAM total":     si.get("ram_total_mb"),
            "Hostname":      si.get("hostname"),
        }
        sw = {
            "Python":        si.get("python"),
            "ONNX Runtime":  si.get("ort_version"),
            "OpenCV":        si.get("cv2_version"),
            "SciPy":         si.get("scipy_version"),
            "NumPy":         si.get("numpy_version"),
            "Captured at":   si.get("timestamp"),
        }
        c1.markdown('<div class="card">' +
                    "".join(irow(k, v) for k, v in hw.items()) +
                    "</div>", unsafe_allow_html=True)
        c2.markdown('<div class="card">' +
                    "".join(irow(k, v) for k, v in sw.items()) +
                    "</div>", unsafe_allow_html=True)
    else:
        st.info("Click **📋 Capture System Snapshot** in the sidebar")

    st.markdown("#### 🔬 Experimental Protocol")
    st.markdown("""
    | Parameter | Value |
    |---|---|
    | Inference engine | ONNX Runtime (ORT_ENABLE_ALL optimization) |
    | Warm-up frames | Configurable (default 10, discarded) |
    | Measurement frames | Configurable per trial |
    | Independent trials | Configurable (default 3) |
    | Idle between trials | 2 s (thermal cool-down) |
    | Random seed | 42 (reproducible synthetic frames) |
    | Resource monitor | tegrastats @ 100 ms |
    | CI method | Student's t (parametric) + Bootstrap B=2000 (non-parametric) |
    | Normality test | Shapiro–Wilk (α = 0.05) |
    | Comparison tests | Welch's t-test + Mann–Whitney U (α = 0.05) |
    | Effect size | Cohen's d |
    """)

    st.markdown("#### 📐 Measurement Architecture")
    st.markdown("""
    **Scenario A — Direct**
    ```
    [Camera] ─USB/CSI─► [Jetson Orin]
                             │
                      t_cap  │  t_pre   │  t_inf
                        ◄────┤──────────┤──────────►
                             │          │
                         capture    preprocess   ORT infer
    t_e2e = t_cap + t_pre + t_inf
    ```

    **Scenario B — Wireless**
    ```
    [Camera]─►[RPi]──WiFi TCP──►[Jetson Orin]
               │ t_enc  │   t_net*  │ t_dec │ t_pre │ t_inf
               ├────────┤───────────┼───────┼───────┼──────►
             encode    send       recv   decode  pre   infer

    t_e2e = t_cap† + t_enc + t_net* + t_dec + t_pre + t_inf
    (*) t_net estimated as RTT/2 via 20-ping ping-pong (assumes symmetric path)
    (†) t_cap measured on RPi side and reported separately
    ```
    """)


# ── Run benchmark ──────────────────────────────────────────────────────────────
if run_btn:
    queue = [(mn, pr)
             for mn in ["YOLOv8", "YOLOv11"]
             for pr in ["fp32", "fp16", "int8"]
             if sel.get(f"{mn}_{pr}")]

    if not queue:
        st.warning("Select at least one model in the sidebar.")
    elif is_wireless and not st.session_state.wireless_ready:
        st.error("❌ No RPi connected. Start the TCP server and connect RPi first.")
    else:
        sc_label = "Direct" if is_direct else "Wireless"
        prog_area = st.empty()

        for job_i, (mn, pr) in enumerate(queue):
            with prog_area.container():
                st.markdown(
                    f"⚙️ **{mn} ({pr}) — {sc_label}** "
                    f"[{job_i+1}/{len(queue)}]"
                )
                master_bar = st.progress(0)
                detail_txt = st.empty()

            def make_cb(mbar, dtxt, job, total_jobs, mn=mn, pr=pr):
                def cb(run_i, n_runs, done, total):
                    mbar.progress((run_i * total + done) / (n_runs * total))
                    dtxt.caption(
                        f"{mn} ({pr}) — Trial {run_i+1}/{n_runs}, "
                        f"frame {done}/{total}"
                    )
                return cb

            result = run_full_benchmark(
                model_name  = mn,
                precision   = pr,
                scenario    = sc_label,
                n_runs      = n_runs,
                n_warmup    = n_warmup,
                n_frames    = n_frames,
                use_camera  = is_direct and use_camera,
                cam_idx     = cam_idx if is_direct else 0,
                conn_sock   = st.session_state.wireless_conn if is_wireless else None,
                progress_cb = make_cb(master_bar, detail_txt, job_i, len(queue)),
            )
            st.session_state.results.append(result)

        prog_area.empty()
        st.success(f"✅ Benchmark complete — {len(queue)} model(s) tested ({sc_label})")
        st.rerun()


# ─── Tab: Results ─────────────────────────────────────────────────────────────
with tab_res:
    results = st.session_state.results
    if not results:
        st.markdown("""
        <div class="card" style="text-align:center;padding:60px;color:#8b949e;">
          <div style="font-size:2.5em;">🔬</div>
          <h3 style="color:#8b949e;">No results yet</h3>
          <p>Configure the protocol in the sidebar and click <b>▶ Run Benchmark</b></p>
        </div>""", unsafe_allow_html=True)
    else:
        # KPIs
        best_inf  = min(results, key=lambda r: r["inf"].get("mean_ms", 9e9))
        best_fps  = max(results, key=lambda r: r["inf"].get("avg_fps", 0))
        best_cv   = min(results, key=lambda r: r["inf"].get("cv_pct", 9e9))
        best_eff  = max(results, key=lambda r: r.get("fps_per_watt", 0))

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🥇 Best Avg Inference",
                  f"{best_inf['inf']['mean_ms']:.3f} ms",
                  f"{best_inf['model']} {best_inf['precision']}")
        k2.metric("⚡ Best FPS",
                  f"{best_fps['inf']['avg_fps']:.2f}",
                  f"{best_fps['model']} {best_fps['precision']}")
        k3.metric("📐 Most Stable (CV)",
                  f"{best_cv['inf']['cv_pct']:.2f}%",
                  f"{best_cv['model']} {best_cv['precision']}")
        k4.metric("🌿 Best FPS/Watt",
                  f"{best_eff.get('fps_per_watt',0):.2f}",
                  f"{best_eff['model']} {best_eff['precision']}")

        st.divider()

        # Charts
        labels   = [f"{r['model']}\n{r['precision']}\n{r.get('scenario','')}" for r in results]
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Avg Inference (ms)")
            st.bar_chart({
                "Model": labels,
                "ms":    [r["inf"]["mean_ms"] for r in results]
            }, x="Model", y="ms", color="#58a6ff")

        with col2:
            st.markdown("#### Avg FPS")
            st.bar_chart({
                "Model": labels,
                "FPS":   [r["inf"]["avg_fps"] for r in results]
            }, x="Model", y="FPS", color="#3fb950")

        with col3:
            st.markdown("#### CV % (Stability)")
            st.bar_chart({
                "Model": labels,
                "CV%":   [r["inf"]["cv_pct"] for r in results]
            }, x="Model", y="CV%", color="#e3b341")

        # Inference time series
        if any(r.get("inf_series") for r in results):
            st.markdown("#### 📈 Inference Time per Frame (first trial)")
            series_data = {}
            for r in results:
                k = f"{r['model']} {r['precision']} ({r.get('scenario','')})"
                series_data[k] = r.get("inf_series", [])
            max_len = max(len(v) for v in series_data.values())
            padded  = {k: v + [None] * (max_len - len(v)) for k, v in series_data.items()}
            st.line_chart(padded)

        # Detailed cards
        st.markdown("#### 📋 Detailed Statistical Cards")
        for i in range(0, len(results), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(results):
                    col.markdown(render_result_card(results[i+j]), unsafe_allow_html=True)


# ─── Tab: Summary Table ────────────────────────────────────────────────────────
with tab_summary:
    results = st.session_state.results
    if not results:
        st.info("Run a benchmark to see the summary table.")
    else:
        rows = build_summary_rows(results)
        st.markdown("#### All Metrics — Side by Side")
        st.dataframe(rows, use_container_width=True, hide_index=True)


# ─── Tab: Statistical Comparison ──────────────────────────────────────────────
with tab_compare:
    results = st.session_state.results
    if len(results) < 2:
        st.info("Run at least 2 models to compare.")
    else:
        if comp_btn or not st.session_state.comparisons:
            comps = []
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    a, b = results[i], results[j]
                    la = f"{a['model']} {a['precision']} ({a.get('scenario','')})"
                    lb = f"{b['model']} {b['precision']} ({b.get('scenario','')})"
                    if a.get("inf_series") and b.get("inf_series"):
                        comps.append(compare_two(
                            a["inf_series"], b["inf_series"], la, lb
                        ))
            st.session_state.comparisons = comps

        comps = st.session_state.comparisons
        if comps:
            rows_c = []
            for c in comps:
                rows_c.append({
                    "Model A":       c["label_a"],
                    "Model B":       c["label_b"],
                    "Δ mean (ms)":   f"{c['delta_ms']:+.3f}",
                    "Δ (%)":         f"{c['delta_pct']:+.1f}%",
                    "Welch t":       f"{c['welch_t']:.3f}",
                    "Welch p":       f"{c['welch_p']:.4f}",
                    "Significant?":  "✅ Yes" if c["significant"] else "❌ No",
                    "MW p":          f"{c['mw_p']:.4f}",
                    "Cohen's d":     f"{c['cohens_d']:.3f}",
                    "Effect":        c["effect_mag"],
                })
            st.dataframe(rows_c, use_container_width=True, hide_index=True)
            st.caption("α = 0.05 for all tests. Welch's t-test does not assume equal variance.")
        else:
            st.info("Click **📊 Compute Comparisons** in the sidebar.")


# ─── Tab: Export ──────────────────────────────────────────────────────────────
with tab_export:
    results = st.session_state.results
    comps   = st.session_state.comparisons
    si      = st.session_state.system_info

    if not results:
        st.info("Run a benchmark first to export results.")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        col_a, col_b = st.columns(2)

        # ── Full JSON ─────────────────────────────
        def safe_dump(res):
            out = []
            for r in res:
                rc = {k: v for k, v in r.items() if k != "inf_series"}
                out.append(rc)
            return out

        json_str = json.dumps({
            "system":       si,
            "results":      safe_dump(results),
            "comparisons":  comps,
            "exported_at":  datetime.now().isoformat(),
        }, indent=2)

        col_a.download_button(
            "📦 Download Full JSON",
            data=json_str,
            file_name=f"benchmark_{ts}.json",
            mime="application/json",
            use_container_width=True,
        )

        # ── CSV ───────────────────────────────────
        rows = build_summary_rows(results)
        buf  = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        col_b.download_button(
            "📄 Download CSV",
            data=buf.getvalue(),
            file_name=f"benchmark_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.divider()
        st.markdown("#### 📋 Preview")
        st.dataframe(rows, use_container_width=True, hide_index=True)
