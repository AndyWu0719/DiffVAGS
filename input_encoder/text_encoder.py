import torch
import open_clip
import torch.nn as nn

class TextEncoder(nn.Module):
    """
    Text Encoder using MobileCLIP-S1
    Input:
        text_input: str, input text
    Output:
        text_features: torch.Tensor, encoded text features
    Shape:
        text_input: [B]
        text_features: [B, 512]
    """
    def __init__(self, model_name: str = 'MobileCLIP-S1', embed_dim: int = 512):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained='datacompdr')
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.out_features = embed_dim

        # freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, text_input):
        text_input = text_input
        # Tokenize the input text
        tokens = self.tokenizer([text_input]).to(self.device)   # [B, 77]
        # Encode the text
        text_features = self.model.encode_text(tokens)  # [B, 512]
        # Normalize the features
        text_features = nn.functional.normalize(text_features, dim=-1)  # [B, 512]
        return text_features