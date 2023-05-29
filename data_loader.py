import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T


def npy_to_tensor(data, label):
    data = torch.from_numpy(data).unsqueeze(dim=1).float()
    label = torch.from_numpy(label).long()
    # data, label -> dataset
    dataset = torch.utils.data.TensorDataset(data, label)

    return dataset


def train_test_split(data, label, ratio=0.8):
    # samples number
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.seed(99)
    np.random.shuffle(arr)
    arr_data = data[arr]
    arr_label = label[arr]
    s = int(num_example * ratio)

    data_train = arr_data[:s]
    label_train = arr_label[:s]
    data_val = arr_data[s:]
    label_val = arr_label[s:]

    return data_train, label_train, data_val, label_val


def load_dataset(data, label):
    data_aug = []

    img_trans = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5), 
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), 
        T.RandomRotation((0, 360)), 
        T.ToTensor()
    ])

    for i in range(len(data)):
        fig = np.uint8(np.interp(data[i], (data[i].min(), data[i].max()), (0, 255)))
        fig = Image.fromarray(fig)
        fig = np.asarray(img_trans(fig))
        data_aug.append(fig)

    data_aug = torch.Tensor(np.asarray(data_aug)).float()
    label = torch.from_numpy(label).long()
    dataset = torch.utils.data.TensorDataset(data_aug, label)
    
    return dataset
        
