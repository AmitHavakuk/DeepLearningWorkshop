import os
import shutil
import time

# Paths
base_dir = os.getcwd()
images_dir = os.path.join(base_dir, "./img_align_celeba_full/eval_partition/0")
organized_dir = os.path.join(base_dir, "./img_align_celeba_full/eval_partition/0_identity")

# Ensure the destination directory exists
os.makedirs(organized_dir, exist_ok=True)

# Limit the number of lines to read
max_lines = 10  # make sure the IF-BREAK is not commented down below
start_id = 1
#start_id = 182638
#start_id = 162771

start_line = start_id
stop_line = start_line+max_lines

# Read the identity file
identity_file = os.path.join(base_dir, "./img_align_celeba_full/identity_CelebA.txt")
with open(identity_file, "r") as file:
    # for _ in range(1,start_line):
    #     next(file)

    for line_num, line in enumerate(file, start=start_line):
        if line_num%100==0:
            print(line_num)
        # if line_num >= stop_line:
        #     break  # Stop reading after the first 10 lines

        image_name, identity = line.strip().split()
        # print(image_name)
        # print(identity)

        # Create identity folder if it doesn't exist
        identity_folder = os.path.join(organized_dir, identity)
        os.makedirs(identity_folder, exist_ok=True)

        # Copy the image to the identity folder
        source_image_path = os.path.join(images_dir, image_name)
        destination_image_path = os.path.join(identity_folder, image_name)
        shutil.copy2(source_image_path, destination_image_path)

print("Images copied successfully to identity-based folders.")
time.sleep(2)
