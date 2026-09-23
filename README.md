# Real-Time Edge Vision Benchmark: YOLOv8 & YOLOv11-seg

[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA_Jetson_Orin_Nano-green.svg)](https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit)
[![Platform](https://img.shields.io/badge/Platform-Raspberry_Pi_--_Wi--Fi-blue.svg)](https://www.raspberrypi.com/)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.3.0-76B900.svg)](https://developer.nvidia.com/tensorrt)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg)](https://www.python.org/)

An end-to-end, scientifically rigorous benchmarking and evaluation suite for real-time computer vision (YOLOv8 & YOLOv11-seg) deployed across an edge architecture combining a **Raspberry Pi** (video transmitter & UI display) and an **NVIDIA Jetson Orin Nano** (edge AI inference engine).

---

## 🌟 Key Features

- **⚡ 30 FPS Hardware Streaming:** Unlocked full 30 FPS camera capture at 720p by utilizing hardware-accelerated MJPEG encoding (`cv2.CAP_PROP_FOURCC = MJPG`), eliminating USB 2.0 uncompressed YUYV bus bottlenecks.
- **🔄 Low-Latency Bidirectional Feedback Channel (`DETS`):** Jetson runs inference and sends normalized detection bounding box coordinates back to the RPi over TCP in real-time, overlaying live detections on the RPi Streamlit preview.
- **📊 4-Stage Wireless Latency Decomposition:** Measures and isolates:
  1. Network Transit Latency (Wi-Fi RTT / 2)
  2. Receiver-side JPEG Decompression (`cv2.imdecode`)
  3. Tensor Preprocessing & Normalization
  4. TensorRT GPU Model Inference
- **⚡ Real-Time Deadline Miss Analysis:** Evaluates Deadline Miss Rates (DMR @ 33ms, 50ms, 100ms, 200ms) for deterministic real-time SLAs.
- **🔋 Tegrastats Hardware & Energy Efficiency:** Real-time hardware sampling at 1000ms intervals measuring CPU/GPU utilization, RAM usage, thermals, power draw (Watts), and energy efficiency (**FPS/Watt** and **mJ per inference**).
- **📈 Rigorous Statistical Validation:** Computes Student's $t$ 95% Confidence Intervals, 1000-sample Bootstrap CIs, IQR, P90/P95/P99 tail percentiles, and Shapiro-Wilk normality tests.

---

## 🛠 Architecture Overview

```
 ┌───────────────────────────────────────┐            Wi-Fi TCP            ┌───────────────────────────────────────┐
 │          Raspberry Pi (Edge)          │ ────── JPEG Frames (TCP) ─────> │       NVIDIA Jetson Orin Nano         │
 │  • V4L2 USB Webcam (720p @ 30 FPS)    │                                 │  • Tegrastats Power Monitor (1000ms)  │
 │  • Streamlit Preview + BBox Overlay   │ <───── DETS Feedback BBoxes ─── │  • TensorRT 10.3.0 Engine Execution   │
 └───────────────────────────────────────┘                                 └───────────────────────────────────────┘
```

---

## 📊 Benchmark Results

### Scenario A (Direct Local) vs. Scenario B (Wireless Wi-Fi)

| Model | Quant | Backend | Size | Local Infer (ms) | Local FPS | Wireless E2E (ms) | Wireless FPS | Wi-Fi Goodput | Energy Efficiency |
|---|---|---|---|---|---|---|---|---|---|
| **YOLOv8** | FP32 | ORT (CPU) | 11.7 MB | 187.56 ms | 5.3 FPS | 138.70 ms | 7.2 FPS | 4.52 Mbps | 0.83 FPS/Watt |
| **YOLOv8** | FP16 | TensorRT | 8.6 MB | 15.03 ms | 66.5 FPS | 33.14 ms | 30.2 FPS | 5.15 Mbps | 10.17 FPS/Watt |
| **YOLOv8** | INT8 | TensorRT | 4.8 MB | **12.89 ms** | **77.6 FPS** | **31.51 ms** | **31.7 FPS** | **5.13 Mbps** | **12.51 FPS/Watt** |
| **YOLOv11**| FP32 | ORT (CPU) | 11.1 MB | 186.38 ms | 5.4 FPS | 144.30 ms | 6.9 FPS | 4.31 Mbps | 0.78 FPS/Watt |
| **YOLOv11**| FP16 | TensorRT | 9.0 MB | 17.10 ms | 58.5 FPS | 37.66 ms | 26.6 FPS | 4.79 Mbps | 7.08 FPS/Watt |

*Note: YOLOv11-seg INT8 quantization is explicitly unsupported on TensorRT 10.3.0 due to missing INT8 kernels for the prototype mask generation head.*

---

## 🚀 Quick Start

### Dependencies

#### NVIDIA Jetson Orin Nano:
```bash
pip install opencv-python numpy onnxruntime
```

#### Raspberry Pi:
```bash
pip install streamlit opencv-python numpy
```

---

### Running Scenario B (Wireless Wi-Fi Mode)

1. **Start TCP Server on Jetson Orin Nano:**
   ```bash
   python3 cli_benchmark.py --wireless --port 9876 --frames 150 --runs 3 --warmup 10
   ```

2. **Start Video Sender on Raspberry Pi:**
   ```bash
   streamlit run rpi_sender.py
   ```
   Open the Streamlit UI in browser, input Jetson's IP address, and click **▶ Connect & Stream**.

---

### Running Scenario A (Direct Local Benchmark)

```bash
python3 cli_benchmark.py --frames 150 --runs 3 --warmup 10
```

---

## 📁 Repository Structure

```
.
├── cli_benchmark.py                  # Main headless scientific benchmark CLI
├── rpi_sender.py                     # Streamlit camera sender & UI preview for Raspberry Pi
├── frame_protocol.py                 # Shared binary transport protocol (FRAME, PING, PONG, DETS)
├── trt_inference.py                  # Custom TensorRT 10.3 engine builder & session manager
├── stats_utils.py                    # Pure numpy statistical utility module
├── app.py / benchmark.py / run.sh    # Streamlit interactive UI launchers
├── PROJECT_CONTEXT.md                # Full technical report & journal documentation
├── benchmark_results_scenario_a.csv  # Direct Local execution results (CSV)
├── benchmark_results_scenario_b.csv  # Wireless Wi-Fi execution results (CSV)
├── yolov8/                           # YOLOv8 models & TensorRT engines (.onnx, .trt, .pt)
└── yolov11/                          # YOLOv11-seg models & TensorRT engines (.onnx, .trt)
```

---

## 📄 License & Citation

Developed for academic research and publication in IEEE / Q1 Scientific Reports.
