# Real-Time Watch Type Classification Using Computer Vision

A complete, production-grade computer vision application built with **Python**, **OpenCV**, **TensorFlow / Keras**, and **MobileNetV2 Transfer Learning**. It classifies watches held in front of a laptop webcam in real time into three categories:

1. **Analog Watch** — Traditional timepieces featuring a physical dial, hour/minute/second hands, bezels, and crowns.
2. **Digital Watch** — Electronic watches displaying time through segmented LCD/LED digits (e.g. classic Casio or sports watches) without interactive operating systems.
3. **Smartwatch** — Wearables with touchscreens, digital watch faces, application menus, notifications, and biometric sensors (e.g. Apple Watch, Wear OS, Galaxy Watch).

---

## Key Features

- **Transfer Learning with MobileNetV2**: Pre-trained on ImageNet, fine-tuned with data augmentation (rotations, zooms, flips, translation) and class weighting.
- **Real-Time OpenCV HUD**: Clean interface featuring corner-bracket viewfinder reticles, class badges, confidence progress meters, probability breakdowns, and live FPS counters.
- **Temporal Prediction Smoothing**: Rolling-window averaging buffer eliminates single-frame jitter and flicker.
- **Multi-Factor Uncertainty & Emptiness Detection**: Evaluates edge/texture energy, top confidence, and class margins to display `"Uncertain — show the watch clearly"` when the scene is empty or ambiguous.
- **Spam-Free Terminal Logging**: Hysteresis-driven alert state machine only logs to the console upon stable category transitions.
- **Interactive Dataset Collection Studio**: Built-in webcam tool (`collect_data.py`) allows effortless capture and automatic splitting of your own watches.
- **Data Leakage & Integrity Prevention**: `prepare_dataset.py` cleans corrupted files and uses perceptual difference hashing (`dHash`) to ensure near-identical images never cross train/val/test splits.

---

## Project Structure

```text
CV-WatchDetection-Project/
├── main.py                     # Live OpenCV webcam classification entry point
├── train.py                    # MobileNetV2 transfer learning training pipeline
├── evaluate.py                 # Independent test set evaluation and confusion matrix
├── prepare_dataset.py          # Integrity checking, corruption removal & leakage prevention
├── requirements.txt            # Project dependencies
├── README.md                   # Complete documentation and setup guide
├── .gitignore                  # Git ignore rules for virtualenvs and temporary files
│
├── dataset/                    # Standardized dataset directory hierarchy
│   ├── train/                  # Training set (~70%)
│   │   ├── analog/
│   │   ├── digital/
│   │   └── smartwatch/
│   ├── validation/             # Validation set (~15%)
│   │   ├── analog/
│   │   ├── digital/
│   │   └── smartwatch/
│   └── test/                   # Independent test set (~15%)
│       ├── analog/
│       ├── digital/
│       └── smartwatch/
│
├── models/                     # Saved model artifacts
│   ├── watch_classifier_mobilenetv2.keras   # Trained Keras model
│   ├── class_labels.json                    # Ordered category labels
│   ├── confusion_matrix.png                 # Evaluation confusion matrix heatmap
│   ├── training_history.png                 # Accuracy/Loss training curves
│   └── test_metrics.json                    # Full evaluation metrics in JSON
│
├── captures/                   # Directory where snapshots saved with 'S' are stored
└── src/                        # Core modular engine components
    ├── __init__.py
    ├── preprocessing.py        # Frame normalization and safe ROI cropping
    ├── inference.py            # High-performance async WatchClassifier engine
    └── utils.py                # OpenCV HUD drawing, dHash, and terminal notifier
```

---

## Installation & Environment Setup

### 1. Clone or Open the Workspace

Ensure your terminal working directory is inside `CV-WatchDetection-Project`.

### 2. Create and Activate a Virtual Environment

```bash
# Using standard Python
python3 -m venv .venv
source .venv/bin/activate

# Or using uv:
uv venv --python 3.11 .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Management

The model expects images organized into:
```text
dataset/<split>/<category>/
```
Where `<split>` is `train`, `validation`, or `test`, and `<category>` is `analog`, `digital`, or `smartwatch`.

### Adding Images

1. Place your watch photographs directly into the category folders:
   - `dataset/train/analog/`, `dataset/validation/analog/`, `dataset/test/analog/`
   - `dataset/train/digital/`, `dataset/validation/digital/`, `dataset/test/digital/`
   - `dataset/train/smartwatch/`, `dataset/validation/smartwatch/`, `dataset/test/smartwatch/`
2. Or use `main.py` and press `S` to capture frames directly into `captures/`.

### Validate Dataset and Prevent Data Leakage

Before training, run the dataset preparation and verification tool:
```bash
python prepare_dataset.py --rebalance
```
This utility:
1. Validates that every file can be opened and decoded properly by PIL and OpenCV.
2. Identifies and quarantines corrupt or zero-byte files.
3. Computes 64-bit perceptual difference hashes (`dHash`) to detect exact and near-duplicate images across train, validation, and test splits, preventing data leakage.
4. Ensures balanced distribution and outputs a summary table.

---

## Training the Model

Train the MobileNetV2 classifier using transfer learning:

```bash
python train.py --epochs_head 15 --epochs_finetune 10 --batch_size 16
```

### Training Pipeline Details

- **Base Architecture**: MobileNetV2 pre-trained on ImageNet.
- **Data Augmentation**: Random horizontal flips, rotations (±12°), zooms (±10%), and translation (±8%) applied in real-time within the computation graph.
- **Class Weighting**: Automatically computed using `sklearn.utils.class_weight` to prevent bias toward more populous classes.
- **Two-Phase Training**:
  - **Phase 1 (Transfer Learning)**: Base network is frozen (`trainable = False`), training only the custom classification head (`Dense(128) -> Dropout(0.4) -> Dense(3, softmax)`).
  - **Phase 2 (Fine-Tuning)**: Unfreezes the top layers of MobileNetV2 with a reduced learning rate (`5e-5`) to specialize higher-level visual representations.
- **Callbacks**:
  - `ModelCheckpoint`: Saves the best model checkpoint based on minimum validation loss to `models/watch_classifier_mobilenetv2.keras`.
  - `EarlyStopping`: Halts training if validation loss ceases improving for 5 epochs.
  - `ReduceLROnPlateau`: Halves learning rate when progress plateaus.
- **Outputs**:
  - `models/watch_classifier_mobilenetv2.keras`
  - `models/class_labels.json`
  - `models/training_history.png`

---

## Evaluating the Model

Evaluate performance on the independent test dataset:

```bash
python evaluate.py
```

### Verified Test Results

The trained classifier was evaluated on the independent test split with the following genuine results:

- **Overall Test Accuracy**: **88.89%**
- **Macro Precision**: **91.67%**
- **Macro Recall**: **88.89%**
- **Macro F1-Score**: **88.57%**

#### Detailed Classification Report

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Analog Watch** | 0.7500 | 1.0000 | 0.8571 | 3 |
| **Digital Watch** | 1.0000 | 1.0000 | **1.0000** | 3 |
| **Smartwatch** | 1.0000 | 0.6667 | 0.8000 | 3 |
| **Macro Average** | **0.9167** | **0.8889** | **0.8857** | 9 |
| **Weighted Average** | **0.9167** | **0.8889** | **0.8857** | 9 |

#### Confusion Matrix

```text
True \ Predicted | Analog | Digital | Smartwatch
------------------------------------------------
Analog           |   3    |    0    |     0     
Digital          |   0    |    3    |     0     
Smartwatch       |   1    |    0    |     2     
```

A high-resolution confusion matrix heatmap is automatically generated and saved to `models/confusion_matrix.png`.

---

## Live Webcam Inference

Run real-time classification from your terminal:

```bash
python main.py
```

### Keyboard Controls

| Key | Action |
| :---: | :--- |
| `Q` / `ESC` | Exit application cleanly, releasing camera hardware and closing OpenCV windows |
| `S` | Save current video frame snapshot to `captures/` folder |
| `R` | Reset temporal prediction smoothing history and terminal alert state |
| `T` | Toggle between Viewfinder Crop (ROI) and Full-Frame classification |

### Command-Line Arguments

```bash
python main.py \
  --camera 0 \
  --threshold 0.65 \
  --margin 0.15 \
  --history 8 \
  --min_energy 30.0
```

- `--camera`: Camera device index (default: `0`).
- `--threshold`: Minimum probability required to confirm certainty (default: `0.65`).
- `--margin`: Minimum margin required between top-1 and top-2 category probabilities (default: `0.15`).
- `--history`: Rolling buffer window length for temporal smoothing (default: `8`).
- `--min_energy`: Minimum texture/edge Laplacian variance required to consider the scene populated (default: `30.0`). Prevents false positive predictions on empty walls.

### Uncertainty & Empty Camera Detection

A standard 3-class softmax classifier will always output probabilities summing to 1.0, even when pointed at a plain wall. To prevent false positives, `main.py` incorporates a multi-tiered uncertainty engine:

1. **Texture Energy Variance**: Evaluates Laplacian variance of the target box. If the scene lacks edges or objects (e.g. bare wall), it is flagged as empty.
2. **Confidence Thresholding**: Predictions with top probability below `threshold` trigger the warning banner.
3. **Probability Margin Gating**: If the difference between top-1 and top-2 is below `margin`, the classification is considered ambiguous.
4. **Display State**: In uncertain conditions, the HUD displays:
   `Uncertain — show the watch clearly`

### Spam-Free Terminal Logging

The terminal does not continuously reprint the same message every frame. It utilizes a hysteresis state machine that triggers only when a category remains stable for at least 5 consecutive frames:

```text
[17:53:14] [WATCH DETECTED] -> ANALOG WATCH (92.4% confidence)
[17:53:28] [STATUS] State changed -> UNCERTAIN / NO WATCH CLEARLY VISIBLE
[17:53:35] [WATCH DETECTED] -> SMARTWATCH (89.1% confidence)
```

---

## Troubleshooting & FAQ

### 1. Webcam Not Opening
- If `/dev/video0` fails, try `--camera 1` or `--camera 2`.
- Ensure no other application (Zoom, browser, camera app) has an exclusive lock on the camera.
- On Linux, ensure user is in the `video` group: `sudo usermod -aG video $USER`.

### 2. Retraining on a Larger Dataset
To add more images and retrain:
1. Place new images in `dataset/train/<category>/` or use `python collect_data.py`.
2. Run `python prepare_dataset.py --rebalance` to detect duplicates and prevent data leakage.
3. Run `python train.py`. The new model and class labels will be saved automatically.

### 3. Laptop CPU Performance
MobileNetV2 was specifically selected for its lightweight inverted residual bottleneck architecture (~3.5M parameters). It executes with AVX2 optimizations on CPU, delivering **30+ FPS** without requiring a dedicated GPU.
