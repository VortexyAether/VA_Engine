import torch
from torch.utils.data import TensorDataset, Dataset
import torchvision.transforms as transforms
import torch.nn.functional as f
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

class loadDataset():
    def __init__(self, file_name, input_frames=1, target_frames=10, normalize=False):
        self.file_name = file_name
        self.input_frames = input_frames
        self.target_frames = target_frames
        self.normalize = normalize
        self.data = torch.tensor(np.load(self.file_name), dtype=torch.float32)

    def get_dataset(self):
        num_samples = self.data.shape[0] - (self.input_frames + self.target_frames) + 1

        input_data = torch.stack([self.data[i:i + self.input_frames] for i in range(num_samples)], dim=0)
        target_data = torch.stack([self.data[i + self.input_frames:i + self.input_frames + self.target_frames] for i in range(num_samples)], dim=0)

        input_data = input_data.permute(0, 2, 1, 3, 4)
        target_data = target_data.permute(0, 2, 1, 3, 4)

        if self.normalize:
            input_data = input_data / 255
            target_data = target_data / 255

        dataset = TensorDataset(input_data, target_data)

        return dataset


class imgloadDataset(Dataset):
    def __init__(self, image_folder, image_size=(32, 32), input_frames=10, target_frames=10, input_channel=3,
                 target_channel=3, normalization=True):
        self.image_folder = image_folder
        self.channelMode = {
                            1 : 'L',
                            3 : 'RGB',
                            4 : 'RGBA'
                            }
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.input_frames = input_frames
        self.target_frames = target_frames
        self.in_channel = self.channelMode[input_channel]
        self.out_channel = self.channelMode[target_channel]
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
        self.normalization = normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_files) - (self.input_frames + self.target_frames) + 1

    def __getitem__(self, idx):
        input_images = []
        target_images = []

        # 입력 프레임 가져오기
        for i in range(self.input_frames):
            image_path = os.path.join(self.image_folder, self.image_files[idx + i])
            image = Image.open(image_path)

            image = image.convert(self.in_channel)

            if self.transform:
                image = self.transform(image)
            if self.normalization:
                image = self.normalize(image)
            input_images.append(image)

        # 예측할 타겟 프레임 가져오기
        for i in range(self.target_frames):
            image_path = os.path.join(self.image_folder, self.image_files[idx + i + self.input_frames])
            image = Image.open(image_path)

            image = image.convert(self.out_channel)

            if self.transform:
                image = self.transform(image)
            if self.normalization:
                image = self.normalize(image)
            target_images.append(image)

        # 입력 및 타겟 프레임을 텐서로 변환하고 (channels, frame, height, width) 순서로 정렬
        input_tensor = torch.stack(input_images, dim=1)  # (channels, frames, height, width)
        target_tensor = torch.stack(target_images, dim=1)  # (channels, frames, height, width)

        return input_tensor, target_tensor

def dataToImage(data, path, order):
    channel = data.shape[0]

    for i in range(channel):
        if i == 0:
            c = 'u'
        elif i == 1:
            c = 'v'
        else:
            c = 'p'

        if not os.path.exists(f'{path}/{c}'):
            os.makedirs(f'{path}/{c}')

        plt.imshow(data[i, :, :], cmap='viridis')
        plt.title(f'{c}_order')
        plt.savefig(f'{path}/{c}/{order}.png')
        plt.close()
