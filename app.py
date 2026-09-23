"""
YOLO Live Detection — Streamlit UI
====================================
Run:  streamlit run app.py
"""

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="YOLO Live Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark app background */
.stApp { background-color: #0d1117; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] .stMarkdown h3 { color: #58a6ff; margin-top: 0; }

/* Metric cards */
.det-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #161b22;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 5px 0;
    border-left: 4px solid #30363d;
}
.det-card .label  { font-size: 0.88em; font-weight: 600; }
.det-card .count  { font-size: 1.4em;  font-weight: 700; }

/* Info table */
.info-row {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid #21262d;
    font-size: 0.85em;
}
.info-row .key   { color: #8b949e; }
.info-row .value { color: #c9d1d9; font-weight: 600; }

/* FPS badge */
.fps-badge {
    display: inline-block;
    background: #238636;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 700;
}

/* Feed placeholder */
.feed-placeholder {
    background: #161b22;
    border: 2px dashed #30363d;
    border-radius: 12px;
    text-align: center;
    padding: 80px 20px;
    color: #8b949e;
}

/* Remove default streamlit padding on images */
[data-testid="stImage"] img { border-radius: 10px; }

/* Toggle styling */
.stToggle > label { font-weight: 600; }

/* Divider color */
hr { border-color: #30363d !important; }

/* Button */
.stButton > button {
    width: 100%;
    border-radius: 8px;
    font-weight: 700;
    border: none;
}
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

CLASSES = {
    0: "aluminium_foil",
    1: "bottle",
    2: "cement_dust",
    3: "floor",
    4: "obstacle",
    5: "paper",
    6: "wooden_dust",
    7: "wrapper",
}

# (hex, BGR) per class
CLASS_PALETTE = {
    0: ("#FFD700", (0,   215, 255)),  # gold          aluminium_foil
    1: ("#FFA500", (0,   165, 255)),  # orange        bottle
    2: ("#FF4444", (71,   99, 255)),  # red           cement_dust
    3: ("#32CD32", (50,  205,  50)),  # lime          floor
    4: ("#CC44FF", (200,  0,  200)),  # violet        obstacle
    5: ("#1E90FF", (255, 144,  30)),  # blue          paper
    6: ("#20B2AA", (170, 178,  32)),  # teal          wooden_dust
    7: ("#3CB371", (60,  179, 113)),  # green         wrapper
}

IOU_THRESHOLD = 0.45


# ── Detector ───────────────────────────────────────────────────────────────────
class Detector:
    def __init__(self, model_name: str, precision: str):
        try:
            import onnxruntime as ort
        except ImportError:
            st.error("onnxruntime not installed. Run:  pip install onnxruntime")
            st.stop()

        self.model_name = model_name
        self.precision  = precision
        cfg             = MODELS[model_name]
        self.input_w, self.input_h = cfg["input_size"]
        model_path      = str(cfg[precision])

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session     = ort.InferenceSession(model_path, sess_opts, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.provider    = self.session.get_providers()[0].replace("ExecutionProvider", "")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.expand_dims(img.transpose(2, 0, 1), 0)   # NCHW

    def postprocess(self, raw, orig_w: int, orig_h: int, conf_thres: float):
        pred = raw[0]                                       # (1, 4+nc, N)  or  (1, N, 4+nc)
        if pred.ndim == 3:
            pred = pred[0].T if pred.shape[1] < pred.shape[2] else pred[0]

        nc         = len(CLASSES)
        boxes      = pred[:, :4]
        class_conf = pred[:, 4:4 + nc]
        conf       = class_conf.max(axis=1)
        class_ids  = class_conf.argmax(axis=1)

        mask = conf >= conf_thres
        boxes, conf, class_ids = boxes[mask], conf[mask], class_ids[mask]
        if len(boxes) == 0:
            return []

        x1 = (boxes[:, 0] - boxes[:, 2] / 2) * orig_w / self.input_w
        y1 = (boxes[:, 1] - boxes[:, 3] / 2) * orig_h / self.input_h
        x2 = (boxes[:, 0] + boxes[:, 2] / 2) * orig_w / self.input_w
        y2 = (boxes[:, 1] + boxes[:, 3] / 2) * orig_h / self.input_h
        xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        results = []
        for cid in np.unique(class_ids):
            idx  = class_ids == cid
            keep = cv2.dnn.NMSBoxes(
                xyxy[idx].tolist(), conf[idx].tolist(), conf_thres, IOU_THRESHOLD
            )
            if len(keep) == 0:
                continue
            for k in keep.flatten():
                b = xyxy[idx][k]
                results.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]),
                                 float(conf[idx][k]), int(cid)))
        return results

    def infer(self, frame: np.ndarray, conf_thres: float):
        h, w  = frame.shape[:2]
        blob  = self.preprocess(frame)
        t0    = time.perf_counter()
        out   = self.session.run([self.output_name], {self.input_name: blob})
        ms    = (time.perf_counter() - t0) * 1000
        dets  = self.postprocess(out, w, h, conf_thres)
        return dets, ms


# ── Drawing ────────────────────────────────────────────────────────────────────
def draw_frame(frame, dets, show_boxes: bool, show_labels: bool) -> np.ndarray:
    if not show_boxes:
        return frame
    for (x1, y1, x2, y2, conf, cid) in dets:
        _, bgr = CLASS_PALETTE.get(cid, ("#888", (200, 200, 200)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        if show_labels:
            label = f"{CLASSES.get(cid, cid)}  {conf:.0%}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 1)
            cv2.rectangle(frame, (x1, y1 - th - bl - 6), (x1 + tw + 8, y1), bgr, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - bl - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (10, 10, 10), 1, cv2.LINE_AA)
    return frame


# ── Session state defaults ─────────────────────────────────────────────────────
_defaults = {
    "cap":          None,
    "detector":     None,
    "detector_key": None,
    "fps":          0.0,
    "inf_ms":       0.0,
    "frame_count":  0,
    "last_dets":    [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎯 YOLO Detection")
    st.divider()

    # ── Model ──────────────────────────────────
    st.markdown("### 🤖 Model")
    model_name = st.radio(
        "model_radio", ["YOLOv8", "YOLOv11"],
        horizontal=True, label_visibility="collapsed"
    )

    # ── Precision ──────────────────────────────
    st.markdown("### ⚙️ Precision")
    precision = st.radio(
        "precision_radio", ["fp32", "fp16", "int8"],
        horizontal=True, label_visibility="collapsed"
    )

    st.divider()

    # ── Detection settings ─────────────────────
    st.markdown("### 🎚️ Detection Settings")
    conf_pct = st.slider(
        "Confidence Threshold", 5, 95, 40, 5, format="%d%%",
        help="Minimum confidence score to show a detection"
    )
    conf_thres = conf_pct / 100.0

    iou_pct = st.slider(
        "IOU Threshold (NMS)", 10, 90, 45, 5, format="%d%%",
        help="Overlap threshold for non-max suppression"
    )
    iou_thres = iou_pct / 100.0

    st.divider()

    # ── Display toggles ────────────────────────
    st.markdown("### 🖥️ Display")
    show_fps    = st.toggle("Show FPS & Inference time", value=True)
    show_boxes  = st.toggle("Show Bounding Boxes",       value=True)
    show_labels = st.toggle("Show Class Labels",         value=True)

    st.divider()

    # ── Camera ─────────────────────────────────
    st.markdown("### 📷 Camera")
    cam_idx = st.selectbox("Camera Source", options=[0, 1, 2, 3], index=0,
                           format_func=lambda x: f"Camera {x}")

    st.divider()

    # ── Start / Stop ───────────────────────────
    live = st.toggle("▶  Live Detection", value=False, key="live_toggle")

    if live:
        st.success("🟢  Detection running", icon="✅")
    else:
        st.info("Toggle **Live Detection** to start", icon="💡")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📹 Live Feed")

col_feed, col_info = st.columns([3, 1], gap="medium")

with col_feed:
    feed_ph = st.empty()

with col_info:
    st.markdown("#### 📊 Detections")
    dets_ph = st.empty()

    st.markdown("#### ℹ️ Model Info")
    info_ph = st.empty()

    st.markdown("#### 📈 Performance")
    perf_ph = st.empty()


# ── Always show model info ─────────────────────────────────────────────────────
def render_info(model_name, precision, provider="–", inp_size=(0, 0)):
    cfg = MODELS[model_name]
    w, h = cfg["input_size"]
    html = f"""
    <div style="background:#161b22;border-radius:8px;padding:12px;">
      <div class="info-row"><span class="key">Model</span>
        <span class="value">{model_name}</span></div>
      <div class="info-row"><span class="key">Precision</span>
        <span class="value">{precision}</span></div>
      <div class="info-row"><span class="key">Input</span>
        <span class="value">{w} × {h}</span></div>
      <div class="info-row" style="border:none;"><span class="key">Device</span>
        <span class="value">{provider}</span></div>
    </div>"""
    return html


info_ph.markdown(render_info(model_name, precision), unsafe_allow_html=True)


# ── Load / reload detector ─────────────────────────────────────────────────────
detector_key = (model_name, precision)
if st.session_state.detector_key != detector_key:
    with st.spinner(f"Loading {model_name} ({precision}) …"):
        st.session_state.detector     = Detector(model_name, precision)
        st.session_state.detector_key = detector_key
    info_ph.markdown(
        render_info(model_name, precision, st.session_state.detector.provider),
        unsafe_allow_html=True
    )

detector = st.session_state.detector
if detector:
    info_ph.markdown(
        render_info(model_name, precision, detector.provider),
        unsafe_allow_html=True
    )


# ── Camera lifecycle ───────────────────────────────────────────────────────────
if live and st.session_state.cap is None:
    cap = cv2.VideoCapture(int(cam_idx))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        st.error(f"❌ Camera {cam_idx} not accessible. Try a different source.")
        st.stop()
    st.session_state.cap = cap

if not live and st.session_state.cap is not None:
    st.session_state.cap.release()
    st.session_state.cap = None


# ── Placeholder when idle ──────────────────────────────────────────────────────
if not live:
    feed_ph.markdown("""
    <div class="feed-placeholder">
      <div style="font-size:3em;">📷</div>
      <h3 style="color:#8b949e; margin:12px 0 8px;">Camera Off</h3>
      <p>Enable <b>▶ Live Detection</b> in the sidebar to start streaming.</p>
    </div>
    """, unsafe_allow_html=True)

    dets_ph.markdown("""
    <div style="color:#8b949e; text-align:center; padding:20px; font-size:0.9em;">
        No active detections
    </div>
    """, unsafe_allow_html=True)

    perf_ph.markdown("""
    <div style="color:#8b949e; text-align:center; padding:10px; font-size:0.9em;">
        –
    </div>
    """, unsafe_allow_html=True)

    st.stop()   # Nothing more to do while idle


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE DETECTION LOOP
# ══════════════════════════════════════════════════════════════════════════════
cap      = st.session_state.cap
t_prev   = time.perf_counter()
fps_smooth = st.session_state.fps

while st.session_state.get("live_toggle", False):

    ret, frame = cap.read()
    if not ret:
        feed_ph.warning("⚠️ Failed to read frame from camera.")
        time.sleep(0.05)
        continue

    # ── Inference ──────────────────────────────
    dets, inf_ms = detector.infer(frame, conf_thres)

    # ── FPS ────────────────────────────────────
    t_now      = time.perf_counter()
    fps_smooth = 0.85 * fps_smooth + 0.15 * (1.0 / max(t_now - t_prev, 1e-6))
    t_prev     = t_now
    st.session_state.fps    = fps_smooth
    st.session_state.inf_ms = inf_ms

    # ── Annotate frame ─────────────────────────
    annotated = draw_frame(frame.copy(), dets, show_boxes, show_labels)

    if show_fps:
        fps_text = f"FPS: {fps_smooth:.1f}  |  {inf_ms:.1f} ms"
        cv2.putText(annotated, fps_text, (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, fps_text, (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 230, 80), 2, cv2.LINE_AA)

    # ── Display frame ──────────────────────────
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    feed_ph.image(rgb, use_container_width=True)

    # ── Detection stats panel ──────────────────
    counts = {}
    for (*_, cid) in dets:
        counts[cid] = counts.get(cid, 0) + 1

    if counts:
        cards = ""
        for cid, cnt in sorted(counts.items()):
            hex_col, _ = CLASS_PALETTE.get(cid, ("#888", None))
            cards += (
                f'<div class="det-card" style="border-left-color:{hex_col};">'
                f'  <span class="label" style="color:{hex_col};">'
                f'    {CLASSES.get(cid, f"class {cid}")}'
                f'  </span>'
                f'  <span class="count" style="color:{hex_col};">{cnt}</span>'
                f'</div>'
            )
        dets_ph.markdown(cards, unsafe_allow_html=True)
    else:
        dets_ph.markdown(
            '<p style="color:#8b949e;text-align:center;padding:20px;">'
            'No detections</p>',
            unsafe_allow_html=True
        )

    # ── Performance panel ──────────────────────
    perf_html = f"""
    <div style="background:#161b22;border-radius:8px;padding:12px;">
      <div class="info-row">
        <span class="key">FPS</span>
        <span class="value" style="color:#2ea043;">{fps_smooth:.1f}</span>
      </div>
      <div class="info-row">
        <span class="key">Inference</span>
        <span class="value">{inf_ms:.1f} ms</span>
      </div>
      <div class="info-row" style="border:none;">
        <span class="key">Objects</span>
        <span class="value">{len(dets)}</span>
      </div>
    </div>
    """
    perf_ph.markdown(perf_html, unsafe_allow_html=True)
