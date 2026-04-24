# 🍎 Fruit Freshness Classifier

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-2.13%2B-D00000?style=flat&logo=keras&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker&logoColor=white)

A real-time fruit and vegetable freshness detector using a webcam and a pretrained Keras model. It draws bounding boxes around detected produce and classifies it as **Fresh** or **Rotten** across 14 categories.

---

## Supported Classes

| Fruit / Vegetable | Fresh | Rotten |
|---|---|---|
| Apple | ✅ | ✅ |
| Banana | ✅ | ✅ |
| Bell Pepper | ✅ | ✅ |
| Carrot | ✅ | ✅ |
| Cucumber | ✅ | ✅ |
| Grape | ✅ | ✅ |
| Guava | ✅ | ✅ |
| Jujube | ✅ | ✅ |
| Mango | ✅ | ✅ |
| Orange | ✅ | ✅ |
| Pomegranate | ✅ | ✅ |
| Potato | ✅ | ✅ |
| Strawberry | ✅ | ✅ |
| Tomato | ✅ | ✅ |

---

## Requirements

- Python 3.8+
- A webcam
- A trained model file: `fruit_freshness_classifier.h5`
- *(Optional)* `fresh.jpeg` and `rotten.jpeg` for visual overlays

---

## Installation

```bash
git clone https://github.com/your-username/fruit-freshness-classifier.git
cd fruit-freshness-classifier
pip install -r requirements.txt
```

> **Tip:** Use a virtual environment to avoid dependency conflicts:
> ```bash
> python -m venv venv
> source venv/bin/activate   # Windows: venv\Scripts\activate
> pip install -r requirements.txt
> ```

---

## Usage

### Basic (uses defaults)
```bash
python live_fruit_classifier.py
```

### With custom options
```bash
python live_fruit_classifier.py \
  --model path/to/your_model.h5 \
  --fresh-img assets/fresh.jpeg \
  --rotten-img assets/rotten.jpeg \
  --camera 0 \
  --confidence-label 70 \
  --confidence-high 80 \
  --confidence-overlay 87
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `fruit_freshness_classifier.h5` | Path to trained `.h5` model |
| `--fresh-img` | `fresh.jpeg` | Overlay image shown for fresh produce |
| `--rotten-img` | `rotten.jpeg` | Overlay image shown for rotten produce |
| `--camera` | `0` | Camera device index |
| `--confidence-label` | `70.0` | Min confidence (%) to display a label |
| `--confidence-high` | `80.0` | Confidence (%) to switch box color to green |
| `--confidence-overlay` | `87.0` | Confidence (%) to trigger the overlay image |

Press **`q`** to quit.

---

## Docker Usage

### Build the image
```bash
docker build -t fruit-freshness-classifier .
```

### Run with webcam (Linux)
```bash
docker run --rm \
  --device /dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  fruit-freshness-classifier
```

### Run with GPU support (requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
```bash
docker run --rm \
  --gpus all \
  --device /dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  fruit-freshness-classifier
```

### Pass custom arguments
```bash
docker run --rm \
  --device /dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  fruit-freshness-classifier \
  python live_fruit_classifier.py --confidence-overlay 85 --camera 0
```

> **macOS / Windows:** Docker does not support direct webcam passthrough on these platforms. Use the script natively with `python live_fruit_classifier.py` instead.

> **CPU-only:** Change the base image in `Dockerfile` from `tensorflow/tensorflow:2.13.0-gpu` to `tensorflow/tensorflow:2.13.0`.

---

## How It Works

1. Each frame is converted to grayscale and blurred, then thresholded to find contours of objects against the background.
2. Contours are sorted by area (largest first). The most prominent object that passes a minimum size filter is cropped out as an ROI.
3. The ROI is resized to `224×224`, normalised, and passed to the Keras model.
4. If the predicted confidence exceeds the label threshold, the class name and confidence are drawn on screen.
5. When confidence exceeds the overlay threshold, a fresh/rotten image is shown in the corner for 3 seconds.

> **Note:** The contour-based detector works best against a plain or contrasting background (e.g. a white table). Detection accuracy on cluttered backgrounds will be lower.

---

## Project Structure

```
fruit-freshness-classifier/
├── live_fruit_classifier.py   # Main script
├── fruit_freshness_classifier.h5  # Your trained model (not included)
├── fresh.jpeg                 # Overlay image (optional)
├── rotten.jpeg                # Overlay image (optional)
├── requirements.txt
└── README.md
```

---

## Model

The model is not included in this repo due to file size. Train your own using a dataset such as:
- [Fruit and Vegetable Disease (Healthy vs Rotten) – Kaggle](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten)

The model expects `224×224` RGB input images normalised to `[0, 1]`, and outputs a softmax over 28 classes.

---

## License

MIT License. See `LICENSE` for details.
