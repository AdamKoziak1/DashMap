import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalCNN(nn.Module):
    def __init__(self, num_channels, height, width):
        super(TemporalCNN, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(num_channels, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Dropout3d(0.2)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv3d(num_channels, 16, kernel_size=(5, 5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Dropout3d(0.2)
        )
        
        self.fusion = nn.Conv3d(32, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        #self.fc_common = nn.Linear(64 * num_frames//4 * height//4 * width//4, 1024) # TODO adjust based on pooling and convolutions
        self.fc_common = nn.Linear(1624448, 1024) # TODO adjust based on pooling and convolutions
        self.dropout = nn.Dropout(0.5)

        self.fc_depth = nn.Linear(1024, height * width) # turn into output mage dimensions
        self.fc_ego_motion = nn.Linear(1024, 12) # turn into flat 3x4 matrix for ego motion output

    def forward(self, x):
        print("x", x.shape)

        x1 = self.branch1(x)
        print("x1", x1.shape)

        x2 = self.branch2(x)
        print("x2", x2.shape)

        x_fused = torch.cat((x1, x2), dim=1) # Fusion of branches
        print("x_fused", x_fused.shape)

        x_fused = self.fusion(x_fused)
        print("x_fused", x_fused.shape)
        
        # Flatten for FC layers
        x_flat = torch.flatten(x_fused, start_dim=1)
        print("x_flat", x_flat.shape)

        x_common = self.dropout(self.fc_common(x_flat))
        print("x_common", x_common.shape)
        
        # Depth map refinement path
        depth_map = self.fc_depth(x_common)
        depth_map = depth_map.view(-1, 1, height, width) # Reshape to image format
        
        # Ego-motion estimation path
        ego_motion = self.fc_ego_motion(x_common)
        
        return depth_map, ego_motion


# Example usage
if __name__ == "__main__":
    num_frames = 7 
    height, width = 392, 518
    batch_size = 1
    D = 1

    model = TemporalCNN(num_frames, height, width)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    input_tensor = torch.randn(batch_size, num_frames, D, height, width)
    print("Input tensor shape:", input_tensor.shape)  # Expected shape: [batch_size, H, W]

    depth_map, ego_motion = model(input_tensor)
    print("Depth output shape:", depth_map.shape)  # Expected shape: [batch_size, H, W]
    print("motion output shape:", ego_motion.shape)  # Expected shape: [batch_size, 12]
