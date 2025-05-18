# USED TO MEASURE METRICS FOR EACH VIDEO

import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import pypiqe
import warnings

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ========== CONFIG ==========
BASE_PATH = r"E:\bakis\stable-diffusion-webui\outputs\img2img-images\text2video"
device = "cuda" if torch.cuda.is_available() else "cpu"
# ============================

# Load CLIP model once
clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

def calculate_clip_score(frames, prompt="a photo"):
    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    inputs = clip_processor(text=[prompt] * len(imgs), images=imgs, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        scores = outputs.logits_per_image.softmax(dim=1)[:, 0]
    return scores.mean().item()


def calculate_niqe(frames):
    try:
        scores = []
        for frame in frames:
            # Convert to grayscale as PyPIQE works with grayscale images
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate PIQE score
            score, activityMask, noticeableArtifactMask, noiseMask = pypiqe.piqe(gray)
            
            scores.append(score)
            
        return float(np.mean(scores))
    except Exception as e:
        print(f"PIQE calculation error: {e}")
        return 5.0

def calculate_ssim(frames):
    if len(frames) < 2:
        return None
    ssim_scores = []
    for i in range(1, len(frames)):
        grayA = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        score = ssim(grayA, grayB)
        ssim_scores.append(score)
    return float(np.mean(ssim_scores))

def calculate_optical_flow(frames):
    if len(frames) < 2:
        return None
    flows = []
    for i in range(1, len(frames)):
        prev = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flows.append(float(mag.mean()))
    return float(np.mean(flows))

def calculate_warping_error(frames):
    if len(frames) < 2:
        return None
    errors = []
    for i in range(1, len(frames)):
        prev = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = flow.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(prev, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        error = np.mean((warped - next) ** 2)
        errors.append(error)
    return float(np.mean(errors))

def calculate_flicker_index(frames):
    if len(frames) < 2:
        return None
    intensities = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in frames]
    diffs = [abs(intensities[i + 1] - intensities[i]) for i in range(len(intensities) - 1)]
    return float(np.mean(diffs))

def calculate_motion_histogram(frames, bins=10):
    if len(frames) < 2:
        return None
    hist_accum = np.zeros(bins)
    for i in range(1, len(frames)):
        prev = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hist, _ = np.histogram(mag, bins=bins, range=(0, 20))
        hist_accum += hist
    if np.sum(hist_accum) > 0:
        hist_norm = (hist_accum / np.sum(hist_accum)).tolist()
        return hist_norm
    return [0.0] * bins

def load_frames(folder):
    images = []
    try:
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        for f in files:
            img = cv2.imread(os.path.join(folder, f))
            if img is not None:
                images.append(img)
        return images
    except Exception as e:
        print(f"Error loading frames from {folder}: {e}")
        return []

def process_video_folder(folder):
    try:
        print(f"Processing: {folder}")
        
        # Load metadata.json
        metadata_path = os.path.join(folder, "metadata.json")
        if not os.path.exists(metadata_path):
            print(f"Missing metadata.json in {folder}")
            return
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Load frames
        frames = load_frames(folder)
        if len(frames) < 2:
            print(f"Not enough frames in {folder}, skipping...")
            return
            
        # Run metrics
        print("Calculating CLIP score...")
        data["clip_score"] = calculate_clip_score(frames, prompt=data.get("prompt", "a photo"))
        
        print("Calculating NIQE...")
        data["niqe"] = calculate_niqe(frames)
        
        print("Calculating SSIM...")
        data["ssim"] = calculate_ssim(frames)
        
        print("Calculating optical flow consistency...")
        data["optical_flow_consistency"] = calculate_optical_flow(frames)
        
        print("Calculating warping error...")
        data["warping_error"] = calculate_warping_error(frames)
        
        print("Calculating flicker index...")
        data["flicker_index"] = calculate_flicker_index(frames)
        
        print("Calculating motion magnitude histogram...")
        data["motion_magnitude_histogram"] = calculate_motion_histogram(frames)
        
        # Save results
        metrics_path = os.path.join(folder, "metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"✔ Saved metrics to {metrics_path}")
        
    except Exception as e:
        print(f"Error processing folder {folder}: {e}")

def main():
    try:
        folders = [os.path.join(BASE_PATH, f) for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))]
        print(f"Found {len(folders)} folders to process")
        
        for folder in tqdm(folders, desc="Processing folders"):
            process_video_folder(folder)
            
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()