import argparse
import glob
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize
from evaluate import infer

DATASET = 'nyu' # Lets not pick a fight with the model's dataloader

def process_images_depth(model, image_path, output_dir, focal_length):
    try:
        # Load the color image and convert it to a tensor
        color_image = Image.open(image_path).convert('RGB')
        image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

        # Infer the depth map using the model
        focal = torch.Tensor([715.0873]).cuda()  # Focal length for the model, 'magic number'. Reluctantly using this for consistency with provided eval script.
        #focal = torch.Tensor([focal_length]).cuda()  # Focal length for the model
        pred = infer(model, image_tensor, focal=focal)

        # Save the raw tensor immediately after inference
        torch.save(pred, os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".pt"))

        # Save a visualization
        # pred_np = pred.squeeze().detach().cpu().numpy()
        # p = colorize(pred_np, 0, 30, cmap='magma_r')
        # os.makedirs(os.path.join(output_dir, 'colorized'), exist_ok=True)
        # Image.fromarray(p).save(os.path.join(output_dir, 'colorized', os.path.splitext(os.path.basename(image_path))[0] + ".png"))

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def infer_all(directory_path):
    with open(directory_path, 'r') as file:
        for line in tqdm(file, desc="Processing Images"):
            parts = line.strip().split(' ')
            drive_name = parts[0].split('/')[1]  # Extract the drive name
            image_path = parts[0]  # The full image path
            focal_length = float(parts[-1])

            image_path = os.path.join('./data/Kitti/raw_data', image_path)
            output_dir = os.path.join('./data/Kitti/inferences', drive_name)
            os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
            
            # Call the process_images_depth function for each image
            process_images_depth(model, image_path, output_dir, focal_length)


def create_dataloader_list(directory_path, new_dir_path):
    with open(directory_path, 'r') as original_file, open(new_dir_path, 'w') as new_file:
        for line in original_file:
            parts = line.strip().split(' ')
            drive_name = parts[0].split('/')[1] 
            image_path = parts[0] 
            
            drive_dir = parts[0][:-29]
            image_num = os.path.splitext(os.path.basename(image_path))[0]

            velodyne_path = os.path.join(drive_dir, 'oxts', 'data', image_num + '.txt')
            parts.append(velodyne_path)

            inference_path = os.path.join(drive_name, image_num + ".pt")
            parts.append(inference_path)

            new_line = f"{' '.join(part for part in parts)}\n" # raw image path, gt path, focal length, pos data, inference path
            print(new_line)
            new_file.write(new_line)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", type=str, default='zoedepth', help="Name of the model to test")
    # parser.add_argument("-p", "--pretrained_resource", type=str, default='local::./checkpoints/depth_anything_metric_depth_outdoor.pt', help="Pretrained resource to use for fetching weights.")

    # args = parser.parse_args()

    # config = get_config(args.model, "eval", DATASET)
    # config.pretrained_resource = args.pretrained_resource
    # model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
    # model.eval()

    # Call infer_all for both training and validation splits
    train_dir_file_path = 'train_test_inputs/kitti_train_files_full.txt'
    val_dir_file_path = 'train_test_inputs/kitti_test_files_full.txt'
    
    # infer_all(train_dir_file_path)
    # infer_all(val_dir_file_path)

    create_dataloader_list(val_dir_file_path, "train_test_inputs/test_files_custom.txt")
    create_dataloader_list(train_dir_file_path, "train_test_inputs/train_files_custom.txt")



