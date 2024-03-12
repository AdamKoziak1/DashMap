import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import open3d as o3d

def depth_to_pointcloud(depth_image, color_image, intrinsics):
    # Assuming depth_image is a 2D numpy array with depth values
    # color_image is the RGB image corresponding to the depth image
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) # 0 is furthest, 1 is closest
    print(depth_image.min(), depth_image.max())

    # depth_inverted = depth_image/255
    depth_inverted = 1 - depth_image     # 1 is furthest, 0 is closest
    print(depth_inverted.min(), depth_inverted.max())
    depth_inverted = np.power(depth_inverted, 4)  
    print(depth_inverted.min(), depth_inverted.max())

    depth_inverted *= 1000                    # 100 is furthest, rough estimate
    print(depth_inverted.min(), depth_inverted.max())
    depth_inverted += 10                     # rough estimate for hood distance
    print(depth_inverted.min(), depth_inverted.max())

    
    #depth_inverted = np.clip(depth_inverted.cpu(), 0, 1)

    #depth_uint8 = depth_inverted.cpu().numpy().astype(np.uint8)

    height, width = depth_inverted.shape
    index_x, index_y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert depth image to 3D point coordinates
    z = depth_inverted / intrinsics['scale']  # Adjust scale if necessary
    x = (index_x - cx) * z / fx
    y = (index_y - cy) * z / fy
    
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color_image.reshape(-1, 3) / 255.0  # Normalize colors to [0, 1]
    valid_points = points[z.reshape(-1) < depth_inverted.max()]
    valid_colors = colors[z.reshape(-1) < depth_inverted.max()]
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(valid_points)
    point_cloud.colors = o3d.utility.Vector3dVector(valid_colors)
    
    return point_cloud


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--outdir', type=str, default='./output')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    
    args = parser.parse_args()
    
    margin_width = 50

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = os.listdir(args.video_path)
        filenames = [os.path.join(args.video_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    for k, filename in enumerate(filenames):
        #intrinsics = {'fx': 525.0, 'fy': 525.0, 'cx': 1280, 'cy': 720, 'scale': 1.0}
        intrinsics = {'fx': 474.62, 'fy': 355.96, 'cx': 1280, 'cy': 720, 'scale': 1.0}

        print('Progress {:}/{:},'.format(k+1, len(filenames)), 'Processing', filename)
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        output_width = frame_width * 2 + margin_width
        
        filename = os.path.basename(filename)
        output_path = os.path.join(args.outdir, filename[:filename.rfind('.')] + '_video_depth.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        i = 0
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break

            color_frame = raw_frame.copy()
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)

            frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
            
            frame = transform({'image': frame})['image']
            frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                depth = depth_anything(frame)

            depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
            depth_raw = depth
            depth_raw = depth_raw.cpu().numpy()

            depth = (depth - depth.min()) / (depth.max() - depth.min())*255

            depth = depth.cpu().numpy().astype(np.uint8)

            if i % 180 == 0:
                point_cloud = depth_to_pointcloud(depth_raw, color_frame, intrinsics)
                
                point_cloud_path = os.path.join(args.outdir, f"clouds/frame_{i}_point_cloud.ply")
                o3d.io.write_point_cloud(point_cloud_path, point_cloud)

            depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            
            split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            combined_frame = cv2.hconcat([raw_frame, split_region, depth_color])
            
            out.write(combined_frame)
            i+=1
            break
        raw_video.release()
        out.release()
