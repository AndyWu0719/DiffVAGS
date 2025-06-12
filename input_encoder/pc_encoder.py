import torch
import torch.nn as nn
import torch.nn.functional as F

class PointCloudEncoder(nn.Module):
    """
    Point Cloud Encoder using PointNet Lite
    Input:
        pc_input: torch.Tensor, input point cloud data
    Output:
        pc_output: torch.Tensor, encoded point cloud features
    Shape:
        pc_input: [B, num_points, 3]
        pc_ouput: [B, 512]
    """
    # PointNet Lite
    def __init__(self, input_dim: int = 3):
        # input_dim: 3 for XYZ coordinates
        # output_features: 512 for final output features
        super().__init__()
        # use 1D convolutional layers to process point cloud data
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(256)
        # fully connected layer to map to output features
        self.fc = nn.Linear(256, 512)

    def forward(self, pc_input):
        # pc_input: [B, num_points, input_dim]
        # return: [B, 512]
        # transpose to [B, input_dim, num_points]
        pc_input = pc_input.transpose(1, 2)
        pc_input = F.relu(self.bn1(self.conv1(pc_input)))      # [B, 64, num_points]
        pc_input = F.relu(self.bn2(self.conv2(pc_input)))      # [B, 128, num_points]
        pc_input = F.relu(self.bn3(self.conv3(pc_input)))      # [B, 256, num_points]
        
        # max pooling
        # [B, 256, num_points] -> [B, 256]
        pc_input = torch.max(pc_input, 2)[0]                   # [B, 256]
        # fully connected layer
        # [B, 256] -> [B, 512]
        pc_input = self.fc(pc_input)                           # [B, 512]
        # normalize
        pc_input = F.normalize(pc_input, p=2, dim=1)
        return pc_input