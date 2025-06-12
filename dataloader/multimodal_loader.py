import os
import sys
import torch
from PIL import Image
from torch.utils.data import Dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from input_encoder.multimodal_encoder import MultiModalEncoder




class MultiModalDataset(Dataset):
    def __init__(self, caption_file, image_dir):
        """
        参数:
            text_dir: 文本存放目录，每个文件存放一个文本内容
            image_dir: 图像存放目录，支持常见格式（如 .jpg, .png 等）
        """
        self.image_dir = image_dir
        self.caption_file = caption_file

        self.captions = {}
        with open(caption_file, "r", encoding="utf-8") as f:
            for line in f:
                if ": " not in line:
                    continue
                parts = line.strip().split(": ", 1)
                if len(parts) == 2:
                    img_name, caption = parts
                    self.captions[img_name.strip()] = caption.strip()

        self.image_files = []
        for f in os.listdir(image_dir):
            if os.path.isfile(os.path.join(image_dir, f)) and f in self.captions:
                self.image_files.append(os.path.join(image_dir, f))
        self.image_files = sorted(self.image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        image_name = os.path.basename(image_path)
        caption = self.captions.get(image_name, "")
        return {"text": caption, "image": image}    # {"text": str, "image": PIL.Image}
    

class MultiModalDataLoader(Dataset):
    """
    该类包装了 MultiModalDataset 和 MultiModalEncoder，
    用于生成编码后的融合向量，方便后续训练中直接使用融合后的特征。
    """
    def __init__(self, dataset: Dataset, encoder: MultiModalEncoder, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            dataset (Dataset): 原始 MultiModalDataset，返回格式为 {"text": str, "image": PIL.Image}
            encoder (MultiModalEncoder): 多模态编码器，将文本和图像特征融合
            device (torch.device): 指定设备，例如 torch.device("cuda") 或 "cpu"
        """
        self.dataset = dataset
        self.encoder = encoder.to(device)
        self.fused_features = []

        self.encoder.eval()
        with torch.no_grad():
            for idx in range(len(self.dataset)):
                data = self.dataset[idx]
                fused = self.encoder(data)
                self.fused_features.append(fused.cpu())

    def __len__(self):
        return len(self.fused_features)

    def __getitem__(self, idx):
        return self.fused_features[idx]