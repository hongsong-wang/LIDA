import os
import cv2
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

# Global Dataset Configuration
SUBSETS = ["BigGAN", "Midjourney", "Wukong", "Stable_Diffusion_v1.4",
           "Stable_Diffusion_v1.5", "ADM", "GLIDE", "VQDM"]
CLASS_NAMES = ["real"] + SUBSETS
NUM_CLASSES = len(CLASS_NAMES)

# Preprocessing Pipeline
PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def compute_patch_complexity(patch):
    """Calculates local variation (gradients) within an image patch."""
    patch = np.array(patch).astype(np.int64)
    diff_h = np.sum(np.abs(patch[:, :-1, :] - patch[:, 1:, :]))
    diff_v = np.sum(np.abs(patch[:-1, :, :] - patch[1:, :, :]))
    diff_d = np.sum(np.abs(patch[:-1, :-1, :] - patch[1:, 1:, :]))
    diff_d += np.sum(np.abs(patch[1:, :-1, :] - patch[:-1, 1:, :]))
    return (diff_h + diff_v + diff_d).sum()


import random


def extract_bit_patch(img, img_height=256, patch_size=32, patch_mode="max", output_mode="full"):
    """
    Performs low-bit plane extraction and forensic selection.

    Args:
        img: Input PIL Image or ndarray.
        img_height: Target dimension for the output.
        patch_size: Size of the square patches to extract.
        patch_mode: Selection strategy for patches ("max", "min", or "random").
        output_mode: Switch to determine return value ("full" or "patch").
    """
    # 1. Bit Plane Extraction (Lowest 3 bits)
    img_np = np.array(img)
    mask_low = 0x07
    red_low3 = ((img_np[:, :, 0] & mask_low) * (255 // 7)).astype(np.uint8)
    green_low3 = ((img_np[:, :, 1] & mask_low) * (255 // 7)).astype(np.uint8)
    blue_low3 = ((img_np[:, :, 2] & mask_low) * (255 // 7)).astype(np.uint8)
    combined = cv2.merge([red_low3, green_low3, blue_low3])

    # 2. Resize/Normalization of the combined image
    if min(combined.shape[0], combined.shape[1]) < patch_size:
        combined = cv2.resize(combined, (img_height, img_height))
    else:
        rz = transforms.Resize((img_height, img_height))
        combined = np.array(rz(Image.fromarray(combined)))

    # Return full image immediately if output_mode is "full"
    if output_mode == "full":
        return cv2.resize(combined, (img_height, img_height))

    # 3. Patch Extraction Logic (only runs if output_mode == "patch")
    h, w, _ = combined.shape
    patch_list = []
    # Grid-based sampling for patches
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = combined[y:y + patch_size, x:x + patch_size]
            patch_list.append(patch)

    if not patch_list:  # Fallback if image is too small for a patch
        return cv2.resize(combined, (img_height, img_height))

    # 4. Patch Selection based on patch_mode
    if patch_mode == "max":
        # Return the patch with the highest complexity (gradients)
        patch_list.sort(key=lambda x: compute_patch_complexity(x), reverse=True)
        selected_patch = patch_list[0]
    elif patch_mode == "min":
        # Return the patch with the lowest complexity
        patch_list.sort(key=lambda x: compute_patch_complexity(x), reverse=False)
        selected_patch = patch_list[0]
    else:
        # Default to random selection
        selected_patch = random.choice(patch_list)

    # Return the selected patch, resized to the expected model input size
    return cv2.resize(selected_patch, (img_height, img_height))


def get_image_files(folder_path, max_samples=None, shuffle=True):
    """Retrieves image file paths from the specified directory."""
    exts = ['.png', '.jpg', '.jpeg']
    if not os.path.exists(folder_path):
        return []
    files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in exts]
    if shuffle: random.shuffle(files)
    return files[:max_samples] if max_samples else files


def compute_cosine_similarity(feat1, feat2):
    """Computes the pairwise cosine similarity matrix between two feature sets."""
    feat1_n = feat1 / (np.linalg.norm(feat1, axis=1, keepdims=True) + 1e-8)
    feat2_n = feat2 / (np.linalg.norm(feat2, axis=1, keepdims=True) + 1e-8)
    return np.dot(feat1_n, feat2_n.T)


def evaluate_metrics(similarity, query_labels, gallery_labels):
    """Calculates Rank-1 and mean Average Precision (mAP)."""
    all_cmc, all_ap = [], []
    for i in range(similarity.shape[0]):
        matches = (gallery_labels[np.argsort(similarity[i])[::-1]] == query_labels[i])
        all_cmc.append((matches.cumsum() > 0).astype(float))
        num_rel = matches.sum()
        if num_rel == 0:
            all_ap.append(0.0)
            continue
        rel_indices = np.where(matches)[0]
        ap = sum((j + 1.0) / (pos + 1.0) for j, pos in enumerate(rel_indices)) / num_rel
        all_ap.append(ap)
    return np.mean(all_cmc, axis=0)[0] * 100, np.mean(all_ap) * 100