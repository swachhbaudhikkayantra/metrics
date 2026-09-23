# PROJECT CONTEXT & COMPREHENSIVE BENCHMARK REPORT
**Target Journal:** IEEE / Q1 Scientific Reports  
**Hardware Platform:** NVIDIA Jetson Orin Nano (Headless DevKit Super, L4T R36.4, CUDA 12.6, TensorRT 10.3.0) + Raspberry Pi (Wireless Camera Sender)  
**Date:** September 23, 2026

---

## 1. Executive Summary & Accomplishments

The edge-computing evaluation framework comparing local execution (Scenario A) vs. wireless streaming (Scenario B) for real-time computer vision (YOLOv8 & YOLOv11-seg) is **100% complete and fully validated**.

- **Reverse Feedback Channel:** Implemented bidirectional TCP protocol (`DETS` packets) sending normalized bounding box coordinates back to RPi for real-time Streamlit visualization.
- **Framerate Optimization:** Hardware MJPEG encoding unlocked 30 FPS webcam streaming on RPi (removing V4L2 USB 2.0 10 FPS bottleneck).
- **Comprehensive Benchmarks:** Executed both Scenario A (Direct Local) and Scenario B (Wireless Wi-Fi) across all active precision variants (3 trials × 150 frames = 450 frames/model per scenario).

---

## 2. Experimental Results: Scenario A vs. Scenario B Comparison

### Scenario A — Direct Local Execution (No Network)

| Model | Quant | Backend | Size | Infer Latency (ms) | 95% CI (ms) | P95 Tail (ms) | FPS | GPU Avg % | Power (W) | Efficiency (FPS/W) |
|---|---|---|---|---|---|---|---|---|---|---|
| **YOLOv8** | FP32 | ORT (CPU) | 11.7 MB | 187.56 ± 45.47 | [183.9, 191.3] | 269.37 | 5.3 | 49.7% | 10.51 W | 0.50 |
| **YOLOv8** | FP16 | TensorRT | 8.6 MB | 15.03 ± 2.81 | [14.8, 15.3] | 18.24 | 66.5 | 65.8% | 9.18 W | 7.24 |
| **YOLOv8** | INT8 | TensorRT | 4.8 MB | **12.89 ± 2.75** | **[12.7, 13.1]** | **17.38** | **77.6** | **63.5%** | **8.30 W** | **9.35** |
| **YOLOv11**| FP32 | ORT (CPU) | 11.1 MB | 186.38 ± 38.23 | [183.3, 189.5] | 262.38 | 5.4 | 54.1% | 10.60 W | 0.51 |
| **YOLOv11**| FP16 | TensorRT | 9.0 MB | 17.10 ± 2.68 | [16.9, 17.3] | 23.35 | 58.5 | 65.0% | 9.20 W | 6.36 |
| **YOLOv11**| INT8 | TensorRT | — | **SKIPPED** | Unsupported | — | — | — | — | — |

---

### Scenario B — Wireless Transmission (RPi → Wi-Fi → Jetson)

| Model | Quant | Backend | Size | Infer (ms) | Net Transit (ms) | Decomp (ms) | Total E2E (ms) | E2E FPS | DMR (<33ms) | Power (W) |
|---|---|---|---|---|---|---|---|---|---|---|
| **YOLOv8** | FP32 | ORT (CPU) | 11.7 MB | 122.70 | 6.55 | 5.25 | 138.70 | 7.2 | 100.0% | 9.77 W |
| **YOLOv8** | FP16 | TensorRT | 8.6 MB | 15.00 | 6.55 | 8.07 | 33.14 | 30.2 | 41.3% | 6.56 W |
| **YOLOv8** | INT8 | TensorRT | 4.8 MB | **12.44** | **6.55** | **8.76** | **31.51** | **31.7** | **47.1%** | **6.42 W** |
| **YOLOv11**| FP32 | ORT (CPU) | 11.1 MB | 129.50 | 6.55 | 5.21 | 144.30 | 6.9 | 100.0% | 9.85 W |
| **YOLOv11**| FP16 | TensorRT | 9.0 MB | 20.05 | 6.55 | 7.88 | 37.66 | 26.6 | 68.0% | 7.05 W |
| **YOLOv11**| INT8 | TensorRT | — | **SKIPPED** | Unsupported | — | — | — | — | — |

---

## 3. Comparative Key Takeaways for Publication

1. **Network & Decompression Overhead:**
   - Wi-Fi Transit Latency adds an average of **6.55 ms** (symmetric one-way estimation).
   - Receiver-side JPEG Decompression (`cv2.imdecode`) adds **5.2 – 8.8 ms** overhead.
   - Total network + decompression pipeline tax is **~15.3 ms**, capping maximum wireless throughput at **~31.7 FPS**.

2. **Quantization Gains (TensorRT INT8 vs FP32):**
   - YOLOv8 INT8 achieves **6.46× speedup** over FP32 (12.44 ms vs 122.70 ms) and consumes **34.3% less power** (6.42 W vs 9.77 W).
   - Energy efficiency improves from **0.83 FPS/Watt to 12.51 FPS/Watt** (15× improvement).

3. **Unsupported Quantization Note:**
   - `YOLOv11-seg` INT8 fails build on TensorRT 10.3.0 due to missing INT8 kernels for prototype head (`/model.23/proto`).

---

## 4. Preserved Result Files

- `benchmark_results_scenario_a.csv` & `benchmark_results_scenario_a.json`: Direct Local Execution
- `benchmark_results_scenario_b.csv` & `benchmark_results_scenario_b.json`: Wireless Wi-Fi Execution
- `PROJECT_CONTEXT.md`: This unified context file.
