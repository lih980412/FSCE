import glob
import shutil

jpg_dirs = [r'D:\UserD\Li\FSCE-1\datasets\my_dataset_22.01.23\22.01.23_train',
            r'D:\UserD\Li\FSCE-1\datasets\my_dataset_22.01.23\22.01.23_val']
target_dir = r'D:\UserD\Li\FSCE-1\datasets\my_dataset_22.01.23\image'
jpg_list = []
for jpg_dir in jpg_dirs:
    temp = []
    temp = glob.glob(jpg_dir + "/*.jpg")
    jpg_list += temp

for jpg in jpg_list:
    shutil.move(jpg, target_dir)
    print("move " + jpg)