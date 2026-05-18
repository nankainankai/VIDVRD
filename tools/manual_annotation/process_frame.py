import os

folder = "D:\\dataset\\addKITTI\\kitti_0006_1"
files = os.listdir(folder)
for file in files:
    file_name = file.split("_")[1]
    os.rename(os.path.join(folder, file), os.path.join(folder, file_name))