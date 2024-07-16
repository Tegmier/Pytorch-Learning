import torch
# print(dir(torch.cuda.is_available))
# print(help(torch.cuda.is_available))

from torch.utils.data import Dataset
from PIL import Image
# print(help(dataset))
img_path = "E:\\code\\Pytorch-Learning\\dataset\\train\\ants\\0013035.jpg"
img = Image.open(img_path)
img.show()

import os
dir_path = "dataset\\train\\ants"

image_path_list = os.listdir(dir_path)

root_dir = "dataset\\train"
ants_label_dir = "ants"
bees_label_dir = "bees"

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        print(img_name)
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path)

ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset

print(len(train_dataset))
