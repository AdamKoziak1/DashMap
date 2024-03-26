import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Simplified3DCNN(nn.Module):
    def __init__(self):
        super(Simplified3DCNN, self).__init__()
        # Initial convolution layer that starts to mix temporal information
        # Kernel size is chosen to modestly capture temporal patterns while focusing on spatial features
        self.conv1 = nn.Conv3d(4, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(64)

        
        # Further processing to deepen feature extraction, focusing more on spatial detail
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(64)
        
        # Final layer to reduce temporal dimension to 1, essentially summarizing the temporal information
        # and preparing to output a single depth map
        self.conv3 = nn.Conv3d(64, 1, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
                
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout3d(0.4)

        # Adaptive pooling to ensure output depth map matches desired spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 392, 518))

    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.conv1(x))))
        #print(x.shape)
        x = self.dropout1(self.relu(self.bn2(self.conv2(x))))
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        x = self.adaptive_pool(x)
        #print(x.shape)
        # Removing the temporal dimension, which is now reduced to 1
        x = x.squeeze(2)
        #print(x.shape)
        return x

# Example instantiation and forward pass
if __name__ == "__main__":
    # Input tensor shape: [Batch Size, Channels, Depth (frames), Height, Width]
    # Assuming a single channel (depth map), stack of 7 frames, with each frame of size 392x518
    input_tensor = torch.randn(1, 1, 7, 392, 518)
    model = Simplified3DCNN()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    output = model(input_tensor)
    print("Output shape:", output.shape)  # Expected output shape: [Batch Size, 1, Height, Width]
