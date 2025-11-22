import cv2
import torch
import numpy as np
import torchvision.transforms as T

from model import loader
from data.transforms import ToTensor

# ------------------------------
# CONFIG
# ------------------------------
MODEL_NAME = "GuideDepth"
MODEL_WEIGHTS = "weights/NYU_Full_GuideDepth.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DEPTH = 10.0
RESOLUTION = (240, 320)
IP_CAM = "http://100.66.254.222:8080/video"

DISPLAY_SCALE = 0.6   # shrink output windows
# ------------------------------


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


cap = cv2.VideoCapture(IP_CAM)

if not cap.isOpened():
    print("❌ Error: Cannot open IP webcam")
    exit()

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    packed = {
        "image": img_rgb,
        "depth": np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
    }

    data = to_tensor(packed)
    image = data["image"].unsqueeze(0).to(DEVICE)
    image = resize(image)

    # ------------------------------
    # INFERENCE
    # ------------------------------
    with torch.no_grad():
        inv_prediction = model(image)
        prediction = inverse_depth_norm(inv_prediction)
        depth_map = prediction[0, 0].cpu().numpy()

    # ------------------------------
    # VISUALIZATION
    # ------------------------------
    depth_norm = (depth_map / depth_map.max() * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

    # Resize depth to webcam frame size
    depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))

    # --------------------------------
    # Shrink both RGB & Depth windows
    # --------------------------------
    new_w = int(frame.shape[1] * DISPLAY_SCALE)
    new_h = int(frame.shape[0] * DISPLAY_SCALE)

    frame_small = cv2.resize(frame, (new_w, new_h))
    depth_small = cv2.resize(depth_colored, (new_w, new_h))

    cv2.imshow("Webcam (RGB)", frame_small)
    cv2.imshow("Depth Prediction", depth_small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
