import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

class KittiTemporalRGBDDataset(Dataset):
    def __init__(self, file_list_path, n_frames=5):
        self.n_frames = n_frames
        self.data = self._parse_file_list(file_list_path)
        self.indices = self._prepare_indices()

        self.raw_path = '../data/Kitti/raw_data'
        self.inference_path = '../data/Kitti/inferences'
        self.gt_path = '../data/Kitti/data_depth_annotated_zoedepth'

    def _parse_file_list(self, file_list_path):
        # Parse the list and group by drive
        data = {}
        with open(file_list_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                drive_id = parts[0].split('/')[2]  # Assuming drive ID is the 3rd element in the path
                if drive_id not in data:
                    data[drive_id] = []
                data[drive_id].append(parts)
        return data

    def _prepare_indices(self):
        # Prepare indices ensuring that n consecutive frames come from the same drive
        indices = []
        for drive_id, frames in self.data.items():
            for i in range(len(frames) - self.n_frames + 1):
                indices.append((drive_id, i))
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        drive_id, start_idx = self.indices[idx]
        frames = self.data[drive_id][start_idx:start_idx + self.n_frames]
        # Load images and depth maps

        crop_size = (392, 518)  # Example crop size, you can change this as needed


        
        # 
        # resized_pred = Image.fromarray(pred).resize((FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST)

        def import_frame(frame):
            color_image = Image.open(os.path.join(self.raw_path, frame[0])).convert('RGB')
            # Apply resize transformation
            resized_image = TF.resize(color_image, (392, 518))
            image_tensor = TF.to_tensor(resized_image)
            return image_tensor
            #transforms.ToTensor()(transforms.CenterCrop(crop_size)(Image.open(os.path.join(self.raw_path, frame[0])).convert('RGB'))).to('cpu')

        rgb_images = [import_frame(frame) for frame in frames]

        depth_maps = [torch.load(os.path.join(self.inference_path, frame[4]), map_location='cpu').squeeze().unsqueeze(0) for frame in frames]  # Assuming .pt files contain torch tensors

        # Stack images to create RGBD frames
        rgbd_images = [torch.cat((rgb, depth), dim=0) for rgb, depth in zip(rgb_images, depth_maps)]
        
        # Create the stacked input tensor (N, C, H, W)
        x = torch.stack(rgbd_images).permute(1,0,2,3)

        # Load ground truth for the middle frame
        middle_frame = frames[self.n_frames // 2]




        #y = transforms.ToTensor()(transforms.CenterCrop(crop_size)(Image.open(os.path.join(self.gt_path, middle_frame[1])))).to(dtype=torch.float)

        middle_frame = frames[self.n_frames // 2]
        gt_path = os.path.join(self.gt_path, middle_frame[1])
        gt_image = Image.open(gt_path).convert('L')  # Assuming ground truth is a single-channel image
        resized_gt_image = TF.resize(gt_image, (392, 518))
        y = TF.to_tensor(resized_gt_image)

        return x, y
        
if __name__ == "__main__":
    # Usage example
    file_list_path = '../train_test_inputs/test_files_custom.txt'
    dataset = KittiTemporalRGBDDataset(file_list_path, n_frames=7)  # For n consecutive frames
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size set to 1 due to stacking

    # To get a single batch
    for x, y in dataloader:
        print(x.shape, y.shape)
    
