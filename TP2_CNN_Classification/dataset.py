import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import random

POC_LABEL_MAP = {
    0: "Chorionic_villi",
    1: "Decidual_tissue",
    2: "Hemorrhage",
    3: "Trophoblastic_tissue"
}

class POCDataset(Dataset):
    def __init__(self, data_dir, data_type="Training", transform=None, split=None, split_ratio=0.8):
        super().__init__()
        self.data_dir = data_dir
        self.data_type = data_type
        self.transform = transform
        self.split = split
        self.split_ratio = split_ratio

        image_names, labels = self._load_data()

        # training data -> split 수행
        if self.data_type == "Training" and self.split is not None:
            random.seed(42) # 항상 동일한 split 유지

            combined = list(zip(image_names, labels)) # 이미지 이름, 레이블 쌍
            random.shuffle(combined)

            split_point = int(len(combined) * self.split_ratio)

            train_data = combined[:split_point]
            val_data = combined[split_point:]


            if self.split == "train":
                selected = train_data
            elif self.split == "val":
                selected = val_data
            else:
                raise ValueError("split must be 'train' or 'val'")

            self.image_names, self.labels = zip(*selected)

        # testing data -> 전체 사용. split 없음
        else:
            self.image_names = image_names
            self.labels = labels

    def _load_data(self):
        full_path = os.path.join(self.data_dir, self.data_type)

        image_list = []
        label_list = []

        # POC_LABEL_MAP의 각 클래스 폴더 이미지 읽기
        for label_idx, folder_name in POC_LABEL_MAP.items():
            folder_path = os.path.join(full_path, folder_name)

            if not os.path.isdir(folder_path):
                continue

            files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]

            for f in files:
                image_list.append(f)
                label_list.append(label_idx)

        return image_list, label_list

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        label = self.labels[idx]
        folder_name = POC_LABEL_MAP[label]

        img_path = os.path.join(self.data_dir, self.data_type, folder_name, img_name)
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            # 사용자 지정 transform이 있으면 적용
            img = self.transform(img)
        else:
            # transform 지정 X -> 기본 전처리 적용
            img = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor()
            ])(img)

        return img, torch.tensor(label).long()
