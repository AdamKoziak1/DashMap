import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CustomModelV2(nn.Module):
    def __init__(self, num_frames, num_channels, img_height, img_width):
        super(CustomModelV2, self).__init__()
        self.num_frames = num_frames
        self.img_height = img_height
        self.img_width = img_width

        # Initial 2D convolution layer to reduce spatial resolution, applied frame by frame
        self.conv2d_initial = nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn2d_initial = nn.BatchNorm2d(16)

        # 3D convolution layer to process the frames after initial 2D convolutions
        self.conv3d = nn.Conv3d(16, 48, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn3d = nn.BatchNorm3d(48)

        # Fully connected layers for intermediate representations
        self.fc_intermediate_size = img_height // 2 * img_width // 2 * 48  # Example calculation, adjust as needed
        self.fc_intermediate = nn.Linear(self.fc_intermediate_size, 512)

        # Final output fully connected layer
        self.fc_final = nn.Linear(512, img_height * img_width * num_channels)

    def forward(self, x):
        batch_size = x.size(0)

        # Apply 2D convolutions frame by frame
        x = x.view(-1, x.size(2), x.size(3), x.size(4))  # Flatten frames for batch-wise 2D conv
        x = F.relu(self.bn2d_initial(self.conv2d_initial(x)))

        # Reshape for 3D convolution
        x = x.view(batch_size, self.num_frames, 16, self.img_height, self.img_width)
        x = F.relu(self.bn3d(self.conv3d(x)))

        # Flatten for fully connected layers
        x = x.view(batch_size, -1)
        x = F.relu(self.fc_intermediate(x))

        # Final output layer
        output = self.fc_final(x)
        output = output.view(batch_size, self.num_frames, self.img_height, self.img_width, -1)
        return output

# Example usage
if __name__ == "__main__":
    num_frames = 7
    num_channels = 4
    img_height = 392
    img_width = 518
    model = CustomModelV2(num_frames, num_channels, img_height, img_width)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    input_tensor = torch.randn(1, num_frames, num_channels, img_height, img_width)
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Expected output shape: [Batch Size, Num Frames, Height, Width, Num Channels]

