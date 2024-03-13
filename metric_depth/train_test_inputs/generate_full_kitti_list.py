import os

def get_focal_lengths():
    def read_focal_lengths(file_path):
        focal_lengths = {}
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(' ')
                drive_name = parts[0].split('/')[1]  # Extract the drive name
                focal_length = parts[-1]
                focal_lengths[drive_name] = focal_length
        return focal_lengths

    train_dir_file_path = 'kitti_eigen_train_files_with_gt_sorted.txt'
    val_dir_file_path = 'kitti_eigen_test_files_with_gt_sorted.txt'

    # Read the focal lengths for both train and val sets
    focal_lengths = {}
    focal_lengths.update(read_focal_lengths(train_dir_file_path))
    focal_lengths.update(read_focal_lengths(val_dir_file_path))

    return focal_lengths

def write_valid_pairs(depth_drives_path, image_date_path, focal_lengths, out_name):
    output_lines = [] 
    for drive in os.listdir(depth_drives_path):
        if drive not in focal_lengths:
            print("Warning: ", drive, "does not have an associated focal length. It will be omitted from the output list.")
            continue

        #print(drive) # "2011_xx_xx_drive_xxxx_sync"
        drive_date = drive[:10]

        depth_images_path = os.path.join(drive, 'proj_depth', 'groundtruth', 'image_02')
        rgb_images_path = os.path.join(drive_date, drive, 'image_02', 'data')


        image_depth_names = set(os.listdir(os.path.join(depth_drives_path, drive, 'proj_depth', 'groundtruth', 'image_02')))
        image_names = set(os.listdir(os.path.join(image_date_path, drive_date, drive, 'image_02', 'data')))

        valid_image_names = image_depth_names & image_names

        for image_name in valid_image_names:
            #print(image_name)
            rgb_image_path = os.path.join(rgb_images_path, image_name)
            depth_image_path = os.path.join(depth_images_path, image_name)
            focal_length = focal_lengths[drive]
            line = f"{rgb_image_path} {depth_image_path} {focal_length}"
            output_lines.append(line)
            #print(line)

    with open(out_name, 'w') as file:
        for line in output_lines:
            file.write(line + '\n')


focal_lengths = get_focal_lengths()

image_folders_path = '../data/Kitti/raw_data'

depth_drives_path = '../data/Kitti/data_depth_annotated_zoedepth'
depth_drives_path_train = os.path.join(depth_drives_path, "train")
depth_drives_path_val = os.path.join(depth_drives_path, "val")

write_valid_pairs(depth_drives_path_val, image_folders_path, focal_lengths, out_name= "val.txt")
write_valid_pairs(depth_drives_path_train, image_folders_path, focal_lengths, out_name= "train.txt")
print("Make sure to sort these, there's a bash script included in this folder.")