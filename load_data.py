# https://deep-learning-study.tistory.com/470
import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, data_dir, labels_dir, transform=None):
        self.data_dir = data_dir # 이미지 데이터 경로
        self.labels_dir = labels_dir # 레이블 데이터 경로
        
        self.image_names = os.listdir(self.data_dir) # 이미지 데이터 내의 모든 파일(디렉토리) 리스트를 리턴한다. -> ['000019.jpg', '000002.jpg', ...]
        self.full_image_names = [os.path.join(self.data_dir, f) for f in self.image_names] # ['./data/train/image/000019.jpg', './data/train/image/000002.jpg', ...]
        self.labels_df = pd.read_csv(self.labels_dir, index_col='image') # csv 파일 읽어들이기        
        self.labels = [self.labels_df.loc[image_name].values[0] for image_name in self.image_names]
#         print(self.labels)
    
        self.transform = transform
    
    def __len__(self):
        return len(self.full_image_names)

    def __getitem__(self, idx):
        image = Image.open(self.full_image_names[idx])
        image = self.transform(image)
        return image, self.labels[idx]

def load_dataset():
    
    data_transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    
    train_datasets = CustomDataset(data_dir='./data/train/image', labels_dir='./data/train/label.csv', transform=data_transformer)

    valid_datasets = CustomDataset(data_dir='./data/valid/image', labels_dir='./data/valid/label.csv', transform=data_transformer)

    test_datasets = CustomDataset(data_dir='./data/test/image', labels_dir='./data/test/label.csv', transform=data_transformer)

    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=4, shuffle=True, num_workers=4)
    valid_dataloader = torch.utils.data.DataLoader(valid_datasets, batch_size=4, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=4, shuffle=True, num_workers=4)

    dataloaders = {
        'train': train_dataloader,
        'valid': valid_dataloader,
        'test': test_dataloader
    }

    dataset_sizes = {
        'train': len(train_datasets),
        'valid': len(valid_datasets),
        'test': len(test_datasets)
    }

    return dataloaders, dataset_sizes
    
def imsave(input):
    # torch.Tensor를 numpy 객체로 변환
    input = input.numpy().transpose((1, 2, 0))
    # 이미지 출력
    plt.imshow(input)
    plt.savefig('test.png')

if __name__ == "__main__":
    data_transformer = transforms.Compose([transforms.ToTensor()])
    custom_dataset = CustomDataset(data_dir='./data/train/image', labels_dir='./data/train/label.csv', transform=data_transformer)
    # print('length of custom dataset is ', len(custom_dataset))
    image, label = custom_dataset[2]
    imsave(image)