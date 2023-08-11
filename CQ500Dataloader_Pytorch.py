# 导入必要的库
import glob
import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 定义数据集类
class CT_Dataset(Dataset):
    def __init__(self, data_dir, ID_list,transform=None):
    # 初始化参数
        self.slices_dir = data_dir+"/Slices/" # Slices文件夹路径
        self.labels_dir = data_dir+"/Labels/" # Labels文件夹路径
        self.transform = transform # 可选的数据转换函数
        self.file_names = [] # 存储文件名的列表
        self.ID_list = ID_list

        for item in ID_list:
        #file_name = f"CQ500-CT-{i}_Slice{j}.npy" # 文件名格式
            self.file_names.append(item) # 添加到列表

    def __len__(self):
        # 返回数据集的长度，即文件数量
        return len(self.file_names)

    def __getitem__(self, index):
    # 根据索引返回一个数据样本，包括图片和标签
        print("reading:" + self.file_names[index])
        file_name = self.file_names[index] # 获取文件名
        slice_path = self.slices_dir + file_name+".npy" # 拼接图片文件路径
        label_path = self.labels_dir + file_name+".npy" # 拼接标签文件路径

# 从文件中读取图片和标签
        image = np.load(slice_path) # 读取图片为numpy数组
        label = np.load(label_path, allow_pickle=True) # 读取标签为字典

        data_dict = label.item()
        label = int(data_dict['label'])#提取标签值


        # 如果有转换函数，对图片进行转换
        # if self.transform:
        #     image = self.transform(image)

        # 将图片和标签转换为PyTorch张量
        image = torch.from_numpy(image).float() # 转换为浮点型张量
        label = torch.tensor(label).long() # 转换为长整型张量

        # 返回图片和标签
        return image, label

# 定义数据转换函数，例如将图片归一化或增加维度等
#     def transform(image):
#         image = image / 255.0 # 将像素值归一化到[0,1]区间
#         image = np.expand_dims(image, axis=0) # 在第一个维度上增加一个维度，用于表示通道数
#         return image

# 创建数据集对象，传入文件夹路径和转换函数



num_slices_original = 28
num_slices_per_subject = 24       # always using 16 slices per subject
start_slice = (num_slices_original - num_slices_per_subject)/2
end_slice = start_slice + num_slices_per_subject

start_slice = int(start_slice)
end_slice = int(end_slice)

data_dir = './'
all_IDs = set()
all_Slices = glob.glob(r".\Slices\CQ500-CT-*")

for item in all_Slices:
    subj_match = re.match(r".*CQ500-CT-([0-9]+)_Slice[0-9]+\.npy", item)
    subj_id = subj_match.group(1)
    #all_IDs= all_IDs.union(subj_id)
    all_IDs.add(subj_id)


# use half of the IDs for testing
# print(all_IDs.__sizeof__())

all_IDs = list(all_IDs)
half = int(np.floor(len(all_IDs)/10*9))
all_IDs = all_IDs[0:half]

all_IDs_slices = list()
for subj_id in all_IDs:
    for slice_num in range(start_slice, end_slice):
        all_IDs_slices.append("CQ500-CT-"+subj_id + "_Slice" + str(slice_num)) #一个元素样例 5_Slice22


# # create a dict of labels for all slices
# all_labels = dict()
# label_files = glob.glob(data_dir + "Labels/CQ500-CT-*")
# for item in label_files:
#     slice_match = re.match(data_dir + "Labels/(CQ500-CT-[0-9]+)_Slice([0-9]+).npy", item)
#     subj_id = slice_match.group(1)
#     slice_num = slice_match.group(2)
#     # data_obj = np.load(item)
#     data_dict = data_obj.item()
#     all_labels[subj_id + "_Slice" + slice_num] = int(data_dict["label"])    # store labes as 1 or 0 for True or False


# divide list into train and validation
percentage_to_train = 0.8
cutoff_index = int(np.floor(len(all_IDs) * percentage_to_train)) * num_slices_per_subject
training_IDs = all_IDs_slices[0:cutoff_index]
validation_IDs  = all_IDs_slices[cutoff_index:]


dataset = CT_Dataset(".",all_IDs_slices)
# 创建数据加载器对象，传入数据集对象和其他参数，例如批量大小，是否打乱顺序等
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 使用数据加载器对象进行迭代，获取批量的图片和标签
for images, labels in dataloader:
    print(images.shape) # 打印图片张量的形状，例如(32, 1, 512, 512)
    print(labels.shape) # 打印标签张量的形状，例如(32,)
    break # 退出循环
