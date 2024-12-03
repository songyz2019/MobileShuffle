import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from pathlib import Path
from torchvision.datasets import VisionDataset
import numpy as np
import pandas as pd
import unittest


class FusarShip(VisionDataset):
    def __init__(self, root_dir=Path("data/fusar_ship")):
        # 常量
        super().__init__()
        self.TRAIN_TRANSFORMS = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.TEST_TRANSFORMS = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.CLASS_LIST = (
            'Cargo',
            'DiveVessel',
            'Dredger',
            'Fishing',
            'HighSpeedCraft',
            'LawEnforce',
            # 'Other',
            'Passenger',
            'PortTender',
            # 'Reserved',
            'SAR',
            'Tanker',
            'Tug',
            # 'Unspecified',
            'WingInGrnd',
        )

        # 变量
        self.root_dir :Path = root_dir
        info_table :pd.DataFrame = pd.read_csv(root_dir/'meta.csv')
        self.INFO = info_table[ info_table['path'].str.startswith(self.CLASS_LIST)].reset_index(drop=True)



    def __len__(self):
        return len(self.INFO)

    def onehot(self, x):
        return np.eye(len(self.CLASS_LIST))[self.CLASS_LIST.index(x)]

    def __getitem__(self, index):
        class_ = self.INFO['path'][index]
        id_ = self.INFO['id'][index]
        base_class, sub_class = class_.split('\\')

        # SB数据集
        SUB_CLASS_MAPPER = {
            'Tanker\\LPGTanker': ('Tanker', 'Ship_C12S09N0001'),
            'Passenger\\Platform': ('Passenger', 'Ship_C08S02N0001'),
        }
        if class_ in SUB_CLASS_MAPPER.keys():
            sub_class, id_ = SUB_CLASS_MAPPER[class_]

        # SB数据集
        ID_MAPPER = {
            'Ship_C12S09N0020': 'UnidentifiedImageError',
            'Ship_C08S03N0006': 'UnidentifiedImageError',
            'Ship_C12S09N0067': 'UnidentifiedImageError',
            'Ship_C08S02N0018': 'UnidentifiedImageError',
            'Ship_C12S09N0098': 'UnidentifiedImageError',
            'Ship_C12S08N0008': 'UnidentifiedImageError',
            'Ship_C12S05N0009': 'UnidentifiedImageError',
            'Ship_C12S09N0047': 'UnidentifiedImageError',
        }
        if id_ in ID_MAPPER.keys():
            id_ = ID_MAPPER[id_]

        # SB数据集
        img_path = self.root_dir / base_class / sub_class / (id_+'.tiff')
        if not img_path.exists():
            img_path = next((self.root_dir / base_class / sub_class).iterdir())

        img = Image.open( img_path )
        img = img.convert('RGB')
        return self.TRAIN_TRANSFORMS(img), self.onehot(base_class)


class Test(unittest.TestCase):
    def setUp(self):
        pass

    def test_0(self):
        dataset = FusarShip()
        print(f"{dataset.INFO.tail()=}")

        print(f"{len(dataset)=}")
        img, label = dataset[3290]
        print(f"{img=}")
        print(f"{label=}")
        for i in range(len(dataset)):
            try:
                dataset[i]
            except Exception as err:
                print(err)