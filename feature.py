import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from model import SourceSpecificProbing, FeatureExtractor
import utils


def main():
    parser = argparse.ArgumentParser(description='Gallery Feature Extraction')
    parser.add_argument('--dataset_path', type=str, default='/path/to/dataset')
    parser.add_argument('--weight_path', type=str, default='/path/to/weights.pth')
    parser.add_argument('--save_dir', type=str, default='/path/to/features')
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--num_gallery_per_class', type=int, default=10)
    parser.add_argument('--mode', type=str, default='full')
    parser.add_argument('--patch_mode', type=str, default='max')
    parser.add_argument('--patch_size', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")

    # Model initialization
    base = SourceSpecificProbing(pretrain=True)
    if os.path.exists(args.weight_path):
        sd = torch.load(args.weight_path, map_location=device)
        # Weight mapping logic
        mapped_sd = {}
        for k, v in sd.items():
            if k.startswith('fc.'): continue
            new_k = k.replace('backbone.', 'disc.backbone.') if k.startswith('backbone.') else 'disc.' + k
            mapped_sd[new_k] = v
        base.load_state_dict(mapped_sd, strict=False)

    model = FeatureExtractor(base).to(device).eval()

    gallery_features, gallery_labels = [], []

    # 1. Real Images
    path = os.path.join(args.dataset_path, utils.SUBSETS[0], "train", "nature")
    files = utils.get_image_files(path, max_samples=args.num_gallery_per_class)
    for f in tqdm(files, desc="Processing Real Gallery"):
        feat = extract_single(args, os.path.join(path, f), model, device)
        gallery_features.append(feat)
        gallery_labels.append(0)

    # 2. Fake Images
    for idx, subset in enumerate(utils.SUBSETS):
        path = os.path.join(args.dataset_path, subset, "train", "ai")
        files = utils.get_image_files(path, max_samples=args.num_gallery_per_class)
        for f in tqdm(files, desc=f"Processing {subset} Gallery"):
            feat = extract_single(args, os.path.join(path, f), model, device)
            gallery_features.append(feat)
            gallery_labels.append(idx + 1)

    np.save(os.path.join(args.save_dir, 'gallery_feat.npy'), np.array(gallery_features))
    np.save(os.path.join(args.save_dir, 'gallery_lb.npy'), np.array(gallery_labels))
    print(f"Features saved to {args.save_dir}")


def extract_single(args, path, model, device):
    img = Image.open(path).convert("RGB")
    processed = utils.extract_bit_patch(img, patch_size=args.patch_size, patch_mode=args.patch_mode, output_mode=args.mode)
    tensor = utils.PREPROCESS(Image.fromarray(processed)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tensor).flatten().cpu().numpy()


if __name__ == "__main__":
    main()