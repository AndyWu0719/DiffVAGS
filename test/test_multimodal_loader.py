import os
import sys

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# 引入你定义的 MultiModalEncoder 和 MultiModalDataLoader
from input_encoder.multimodal_encoder import MultiModalEncoder
from dataloader.multimodal_loader import MultiModalDataLoader, MultiModalDataset
from input_encoder.text_encoder import TextEncoder
from input_encoder.image_encoder import ImageEncoder


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    text_encoder = TextEncoder()
    image_encoder = ImageEncoder()
    
    encoder = MultiModalEncoder(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        fusion_outchannels=768,
        fusion_outfeatures=512
    )
    
    caption_path = "/home/andywu/Documents/dongjun/LightDiffGS/process_data/step1/texts/captions.txt"
    image_path = "/home/andywu/Documents/dongjun/LightDiffGS/process_data/step1/images"
    dataset = MultiModalDataset(caption_file=caption_path, image_dir=image_path)
    print("Number of samples in dataset:", len(dataset))
    
    fused_loader = MultiModalDataLoader(dataset, encoder, device=device)
    print("Number of fused features:", len(fused_loader))
    
    for i in range(len(fused_loader)):
        fused_feature = fused_loader[i]
        print(f"Sample {i} fused feature shape: {fused_feature.shape}")
        assert fused_feature.shape == (768, 64, 64), f"Shape mismatch: got {fused_feature.shape}"
    
    print("Test passed!")

if __name__ == '__main__':
    main()