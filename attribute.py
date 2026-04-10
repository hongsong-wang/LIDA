import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from model import SourceSpecificProbing, FeatureExtractor
import utils

def main():
    parser = argparse.ArgumentParser(description='Source Tracing and Evaluation')
    parser.add_argument('--dataset_path', type=str, default='/path/to/dataset')
    parser.add_argument('--weight_path', type=str, default='/path/to/weights.pth')
    parser.add_argument('--save_dir', type=str, default='/path/to/features')
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--num_query_per_class', type=int, default=2000)
    parser.add_argument('--mode', type=str, default='full')
    parser.add_argument('--patch_mode', type=str, default='max')
    parser.add_argument('--patch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")

    # Load Stored Gallery
    gallery_feat = np.load(os.path.join(args.feature_dir, 'gallery_feat.npy'))
    gallery_lb = np.load(os.path.join(args.feature_dir, 'gallery_lb.npy'))

    # Model Setup (Same as feature.py)
    base = SourceSpecificProbing(pretrain=True)
    sd = torch.load(args.weight_path, map_location=device)
    mapped_sd = { (k.replace('backbone.', 'disc.backbone.') if k.startswith('backbone.') else 'disc.'+k): v
                 for k, v in sd.items() if not k.startswith('fc.') }
    base.load_state_dict(mapped_sd, strict=False)
    model = FeatureExtractor(base).to(device).eval()

    query_features, query_labels = [], []

    # Extract Query Features
    # Real Queries
    path = os.path.join(args.dataset_path, utils.SUBSETS[0], "val", "nature")
    for f in tqdm(utils.get_image_files(path, args.num_query_per_class), desc="Real Queries"):
        query_features.append(extract_single(args, os.path.join(path, f), model, device))
        query_labels.append(0)

    # Fake Queries
    for idx, subset in enumerate(utils.SUBSETS):
        path = os.path.join(args.dataset_path, subset, "val", "ai")
        for f in tqdm(utils.get_image_files(path, args.num_query_per_class), desc=f"{subset} Queries"):
            query_features.append(extract_single(args, os.path.join(path, f), model, device))
            query_labels.append(idx + 1)

    # Evaluate
    similarity = utils.compute_cosine_similarity(np.array(query_features), gallery_feat)
    r1, mAP = utils.evaluate_metrics(similarity, np.array(query_labels), gallery_lb)

    print(f"\nEvaluation Results:")
    print(f"Overall Rank-1: {r1:.2f}%")
    print(f"Overall mAP:    {mAP:.2f}%")

def extract_single(args, path, model, device):
    try:
        img = Image.open(path).convert("RGB")
        processed = utils.extract_bit_patch(img, patch_size=args.patch_size, patch_mode=args.patch_mode, output_mode=args.mode)
        tensor = utils.PREPROCESS(Image.fromarray(processed)).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(tensor).flatten().cpu().numpy()
    except:
        return np.zeros(2048)

if __name__ == "__main__":
    main()
