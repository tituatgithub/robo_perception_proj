import cv2
import torch
import numpy as np
import torchvision.transforms as T
import os

from model import loader
from data.transforms import ToTensor

# ------------------------------
# CONFIG
# ------------------------------
MODEL_NAME = "GuideDepth"
MODEL_WEIGHTS = "weights/NYU_Full_GuideDepth.pth"   # Correct model file
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DEPTH = 10.0
RESOLUTION = (240, 320)

INPUT_DIR = r"C:\Users\hp\Desktop\perception\perception_pictures"   # <--- Process ALL IMAGES here
OUTPUT_DIR = "output2/"                                             # <--- Save depth maps here
# ------------------------------

# Create output folder if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
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

def run_single_image(image_path):
    filename = os.path.basename(image_path)

    print(f"Processing: {image_path}")

    # Load image
    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"Error reading image: {image_path}")
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    packed = {"image": rgb, "depth": np.zeros((rgb.shape[0], rgb.shape[1]))}
    data = to_tensor(packed)
    image = data["image"].unsqueeze(0).to(DEVICE)

    # Resize to model input
    image = resize(image)

    # Model inference
    with torch.no_grad():
        inv_pred = model(image)
        pred = inverse_depth_norm(inv_pred)[0, 0].cpu().numpy()

    # Normalize for display
    depth_norm = (pred / pred.max() * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

    # Save result
    base = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base}_depth.png")

    cv2.imwrite(output_path, depth_colormap)
    print(f"Saved depth → {output_path}")


# -----------------------------------------
# MAIN – process ALL IMAGES in folder
# -----------------------------------------
if __name__ == "__main__":
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp")

    images = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(supported_ext)
    ]

    print(f"Found {len(images)} images.")

    for img_path in images:
        run_single_image(img_path)

    print("\n✔️ All images processed successfully!")
