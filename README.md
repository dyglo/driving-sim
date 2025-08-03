# 🚗 Autonomous Driving Simulation & Visualization Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow?logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Overview

**Autonomous Driving Simulation & Visualization** is a modular, real-time platform for simulating, visualizing, and researching the perception and decision-making stack of self-driving cars.  
It combines state-of-the-art deep learning (YOLOv8), classical computer vision, and modern UI engineering to process driving videos, detect objects and lanes, predict safe trajectories, and render all results in a professional dashboard overlay.

---

## ✨ Features

- **YOLOv8 Object Detection**: Real-time detection of vehicles, pedestrians, and traffic signs.
- **Lane Detection**: Robust lane line extraction using Canny edge detection and Hough transform.
- **Trajectory & Path Planning**: Fan-shaped projection, obstacle avoidance, and intelligent maneuver suggestions.
- **Safety & Warnings**: Collision prediction, lane departure, and following distance indicators.
- **Modern Dashboard UI**: Glassmorphism panels, real-time stats, and company branding.
- **Performance Metrics**: Live FPS, detection accuracy, and system health.
- **Extensible & Modular**: Easy to swap models, add new sensors, or customize the UI.

---

## 🏗️ Architecture

- **Modular Perception Pipeline**: YOLOv8 for object detection, OpenCV for lane detection.
- **Sensor Fusion**: Combines detection and lane data for unified scene understanding.
- **Path Planning**: Geometric and rule-based logic for trajectory and maneuvering.
- **Visualization**: OpenCV overlays, adaptive scaling, and modern UI design.
- **Callback-based Video Processing**: Efficient, real-time frame handling.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/dyglo/driving-sim.git
cd self-driving
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

- Python 3.8+
- OpenCV 4.x
- Ultralytics YOLOv8 (`pip install ultralytics`)
- NumPy

### 3. Download YOLOv8 Weights

Place your YOLOv8 weights (e.g., `yolov8n.pt`) in the project root or `models/` directory.

### 4. Add Input Videos

Place your driving videos in the `videos/` directory (e.g., `urbanroad.mp4`).

### 5. Run the Simulation

```bash
python main.py
```

- Output video will be saved to `output/processed_output.mp4`
- Live feed will display if enabled in `src/utils/config.py`

---

## 🧑‍💻 Contributing

We welcome contributions! To get started:

1. **Fork this repository**
2. **Create a new branch** for your feature or bugfix
3. **Write clear, well-documented code**
4. **Submit a pull request** with a detailed description

**Ideas for contribution:**
- Add new perception modules (e.g., semantic segmentation, depth estimation)
- Improve lane detection (e.g., deep learning-based)
- Enhance the UI/UX or add web/AR dashboards
- Integrate with real vehicle sensors or robotics platforms
- Add more datasets or video sources


---

## 📚 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- All open-source contributors and the self-driving research community

---

> **This project is a robust foundation for research, prototyping, and community-driven development in autonomous driving and computer vision. Fork it, build on it, and let’s drive the future together!**

