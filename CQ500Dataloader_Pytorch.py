# 导入必要的库
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 定义数据集类
class CT_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
    # 初始化参数
        self.slices_dir = data_dir+"/Slices/" # Slices文件夹路径
        self.labels_dir = data_dir+"/Labels/" # Labels文件夹路径
        self.transform = transform # 可选的数据转换函数
        self.file_names = [] # 存储文件名的列表

        # 遍历Slices文件夹，获取所有文件名
        for i in range(491): # [0-490]
            for j in range(28): # [0-27]
                file_name = f"CT-500-{i}_Slice{j}.npy" # 文件名格式
                self.file_names.append(file_name) # 添加到列表

    def __len__(self):
        # 返回数据集的长度，即文件数量
        return len(self.file_names)

    def __getitem__(self, index):
    # 根据索引返回一个数据样本，包括图片和标签
        file_name = self.file_names[index] # 获取文件名
        slice_path = self.slices_dir + file_name # 拼接图片文件路径
        label_path = self.labels_dir + file_name # 拼接标签文件路径

# 从文件中读取图片和标签
        image = np.load(slice_path) # 读取图片为numpy数组
        label = np.load(label_path, allow_pickle=True) # 读取标签为字典
        label = label["label"] # 提取标签值

        # 如果有转换函数，对图片进行转换
        if self.transform:
            image = self.transform(image)

        # 将图片和标签转换为PyTorch张量
        image = torch.from_numpy(image).float() # 转换为浮点型张量
        label = torch.tensor(label).long() # 转换为长整型张量

        # 返回图片和标签
        return image, label

# 定义数据转换函数，例如将图片归一化或增加维度等
    def transform(image):
        image = image / 255.0 # 将像素值归一化到[0,1]区间
        image = np.expand_dims(image, axis=0) # 在第一个维度上增加一个维度，用于表示通道数
        return image

# 创建数据集对象，传入文件夹路径和转换函数
dataset = CT_Dataset(".")

# 创建数据加载器对象，传入数据集对象和其他参数，例如批量大小，是否打乱顺序等
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 使用数据加载器对象进行迭代，获取批量的图片和标签
for images, labels in dataloader:
    print(images.shape) # 打印图片张量的形状，例如(32, 1, 512, 512)
    print(labels.shape) # 打印标签张量的形状，例如(32,)
    break # 退出循环
