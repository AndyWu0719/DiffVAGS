import torch
import open_clip
import torch.nn as nn
from torchvision import transforms

class ImageEncoder(nn.Module):
    """
    Image Encoder using MobileCLIP-S1
    Input:
        image_input: torch.Tensor or str, input image or image path
    Output:
        image_features: torch.Tensor, encoded image features
    Shape:
        image_input: [B, 3, 224, 224] or [B]
        image_features: [B, 512]
    """
    def __init__(self, model_name: str = 'MobileCLIP-S1', embed_dim: int = 512):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained='datacompdr')
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.out_features = embed_dim
        self.image_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, image_input):
        # Preprocess the input image
        # Check if the input is a tensor, if not, preprocess it
        if not torch.is_tensor(image_input):
            image_input = self.image_preprocess(image_input).unsqueeze(0)  # [B, 3, 224, 224]
        image_input = image_input.to(self.device)
        # Encode the image
        image_features = self.model.encode_image(image_input)  # [B, 512]
        # Normalize the features
        image_features = nn.functional.normalize(image_features, dim=-1)  # [B, 512]
        return image_features