import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

train_path = "./train.txt"
val_path = "./val.txt"
class LoadData(Dataset):
    def __init__(self, txt_path, train_flag):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag

        self.train_transform = transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])


        
        self.test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split('\t'),imgs_info))
            return imgs_info
        
    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        if self.train_flag:
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)
        label = int(label)

        return img,label
    
    def __len__(self):
        return len(self.imgs_info)
   
if __name__ == "__main__":
    train_dataset = LoadData(txt_path=train_path,train_flag=True)
    print("训练集数据个数：", len(train_dataset))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=128,
                                               num_workers=64,
                                               pin_memory=False,
                                               shuffle=True)
    

    # for image,label in train_loader:
    #     print(image.shape)
    #     print(image)
    #     print(label)
    #     break




    val_dataset = LoadData(txt_path=val_path,train_flag=False)
    print("验证集数据个数：", len(val_dataset))

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=128,
                                               num_workers=64,
                                               pin_memory=True,
                                               shuffle=True)
    # for image,label in val_loader:
    #     print(image.shape)
    #     print(image)
    #     print(label)