# GuidedDecoding – Full Setup & Webcam Depth

This repository contains a complete setup workflow for running the GuideDepth model from the GuidedDecoding project on a Windows machine, inside a Python virtual environment, with GPU acceleration.

## Features

- Setting up environment & dependencies
- Downloading required model weights
- Running evaluation on the NYU test dataset
- Running real-time webcam depth estimation
- Fixing common Windows issues (Tkinter, TensorRT, etc.)

---

## 🔧 1. Setup Python Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv percep
percep\Scripts\Activate.ps1
```

If blocked by PowerShell execution policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
percep\Scripts\Activate.ps1
```

---

## ⚡ 2. Install PyTorch (GPU)

Inside the activated venv:

```bash
pip install torch torchvision torchaudio
```

Verify CUDA is available:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

---

## 📦 3. Install Other Dependencies

```bash
pip install opencv-python numpy matplotlib tqdm h5py
```

---

## 📁 4. Required Folder Structure

Create these folders in your project root:

```bash
mkdir weights
mkdir datasets
mkdir model\weights
mkdir output
```

Your folder should look like:

```
GuidedDecoding/
    weights/
    datasets/
    model/
        weights/
    output/
```

---

## 📥 5. Download Required Model Weights

### (A) GuideDepth NYU pretrained weights

Place in: `GuidedDecoding/weights/GuideDepth_NYU.pth`

### (B) DDRNet23s ImageNet backbone (REQUIRED)

Download `DDRNet23s_imagenet.pth`

Place in: `GuidedDecoding/model/weights/DDRNet23s_imagenet.pth`

⚠️ **The model WILL NOT RUN without this backbone file.**

Final structure:

```
model/weights/DDRNet23s_imagenet.pth
weights/GuideDepth_NYU.pth
```

---

## 🧪 6. Download NYU Reduced Testset

Download the official NYU reduced testset ZIP from the authors.

Extract into: `GuidedDecoding/datasets/nyu_test/`

Expected contents:

```
datasets/nyu_test/
    nyu_test_001.npz
    nyu_test_002.npz
    ...
```

✅ `.npz` files are correct — this dataset is preprocessed by the authors.

---

## 🛠️ 7. Fix Windows Matplotlib (Tkinter) Error

Windows + venv often throws:

```
_tkinter.TclError: Can't find a usable init.tcl
```

**Fix:** Open `evaluate.py` and at the top add these lines **BEFORE** importing pyplot:

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

This forces a non-GUI backend, required on Windows.

---

## ▶️ 8. Run Depth Evaluation (NYU)

```bash
python main.py --eval --model GuideDepth --resolution half --dataset nyu_reduced --weights_path weights/GuideDepth_NYU.pth --test_path datasets/nyu_test --save_results output/
```

Output files will be saved in:

```
output/
    image_0.png
    gt_0.png
    depth_0.png
    errors_0.png
    results.txt
```

---

## 🎥 9. Real-Time Webcam Depth Estimation

Create a new file: `webcam_depth.py`

Paste this script:

```python
import cv2
import torch
import numpy as np
import torchvision.transforms as T

from model import loader
from data.transforms import ToTensor

MODEL_NAME = "GuideDepth"
MODEL_WEIGHTS = "weights/GuideDepth_NYU.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DEPTH = 10.0
RESOLUTION = (240, 320)

print("Loading model...")
model = loader.load_model(MODEL_NAME, MODEL_WEIGHTS)
model = model.to(DEVICE)
model.eval()

resize = T.Resize(RESOLUTION)
to_tensor = ToTensor(test=True, maxDepth=MAX_DEPTH)

def inverse_depth_norm(depth):
    depth = MAX_DEPTH / depth
    depth = torch.clamp(depth, MAX_DEPTH / 100, MAX_DEPTH)
    return depth

cap = cv2.VideoCapture(0)
print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    packed = {"image": img_rgb, "depth": np.zeros((img_rgb.shape[0], img_rgb.shape[1]))}
    data = to_tensor(packed)
    image = data["image"].unsqueeze(0).to(DEVICE)

    image = resize(image)

    with torch.no_grad():
        inv_pred = model(image)
        pred = inverse_depth_norm(inv_pred)[0, 0].cpu().numpy()

    depth_norm = (pred / pred.max() * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

    cv2.imshow("RGB", frame)
    cv2.imshow("Depth", depth_colored)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Run webcam depth:

```bash
python webcam_depth.py
```

You will see:
- Live RGB feed
- Live depth visualization
- Press `q` to quit

---

## 🔍 10. Final Expected Folder Structure

```
GuidedDecoding/
│
├── weights/
│     └── GuideDepth_NYU.pth
│
├── model/
│     └── weights/
│           └── DDRNet23s_imagenet.pth
│
├── datasets/
│     └── nyu_test/
│           ├── nyu_test_001.npz
│           ├── nyu_test_002.npz
│           └── ...
│
├── output/
│     ├── image_0.png
│     ├── depth_0.png
│     └── results.txt
│
├── webcam_depth.py
├── main.py
├── evaluate.py
└── percep/   (virtual environment)
```

---

## ✅ Summary

This README provides:

- Full PyTorch + GPU setup
- Correct GuidedDecoding weight placement
- Correct NYU testset usage
- Avoids TensorRT dependency (Jetson-only)
- Fixes all Windows Tkinter issues
- Provides real-time webcam depth script
- Fully reproducible end-to-end pipeline

---

## 📝 Additional Resources

Want more? You can extend this with:

- ✅ A "Run-for-any-image" script
- ✅ A point-cloud (.ply) exporter
- ✅ A demo notebook
- ✅ A streamlined publication-ready README

---

## 🐛 Troubleshooting

### CUDA not available
- Ensure you have a CUDA-compatible GPU
- Install the correct PyTorch version with CUDA support

### Missing DDRNet23s backbone
- The model will fail to load without `DDRNet23s_imagenet.pth`
- Download and place in `model/weights/`

### Tkinter errors
- Add `matplotlib.use('Agg')` before importing pyplot
- Ensure it's at the top of `evaluate.py`

---

## 📄 License

This project uses the GuidedDecoding codebase. Please refer to the original repository for license information.
