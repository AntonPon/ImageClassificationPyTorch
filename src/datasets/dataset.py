from torch.utils.data import Dataset
import os
import csv
import cv2
import numpy as np
import torch
#from torch import FloatTensor

class CustomDataset(Dataset):

    def get_annot_list(self, path_to_annot):
        final_list = []
        ignore_elements = ['2017_24086860.jpg', '2017_17704629.gif', '2017_79148221.gif', '2017_38643152.gif']
        with open(path_to_annot) as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] not in ignore_elements:
                    final_list.append(row)
        return final_list

    def __init__(self, path_to_data, path_to_annot, transforms=None):
        super(CustomDataset, self).__init__()
        self.annot_list = self.get_annot_list(path_to_annot)[:100]
        self.data_path = path_to_data
        self.transforms = transforms

    def __len__(self):
        return len(self.annot_list)

    def __getitem__(self, item):
        item_list = self.annot_list[item]

        image_path = os.path.join(self.data_path, item_list[0])
        labels = [float(label) for label in item_list[1:]]
        image = self.get_image(image_path, self.transforms)
        return {'image': image, 'labels': torch.from_numpy(np.array(labels)).type(torch.FloatTensor)}


    def get_image(self, img_path, transforms):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError('cannot open the image: {}'.format(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        img = normalize(img)


        if transforms is not None:
            augmented = transforms(image=img)
            img = augmented['image']
        img = np.transpose(img, (-1, 0, 1))
        img = torch.from_numpy(img).type(torch.FloatTensor)
        return img



def normalize(img):
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])
    return (img.astype(np.float32)-mean)/std

if __name__ == '__main__':
    path_to_annot = '../../data/csv/val_0_4167.csv'
    path_to_data = '../../data/val2017'
    dataset = CustomDataset(path_to_data=path_to_data, path_to_annot=path_to_annot)
    print(dataset[3])