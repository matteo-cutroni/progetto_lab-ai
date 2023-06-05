import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 as cv
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)



class GrapeDataset100(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Inputs:
            root_dir - path della directory in cui si trovano le cartelle delle classi
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        #lista con nomi delle classi
        self.classes = sorted(os.listdir(root_dir))

        #per creare le liste di immagini, label
        self.load_images()


    def load_images(self):
        data = []
        label = []

        #ad ogni classe è assegnato l'indice corrispondente
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root_dir, class_name)

            #per ogni immagine nella cartella aggiunge nelle liste il path dell'immagine e la sua label (indice della classe)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                data.append(img_path)
                label.append(class_idx)


        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = cv.imread(self.data[idx])
        data_label = self.label[idx]


        if self.transform is not None:
            data_point = self.transform(data_point)

        return data_point, data_label

# Fissare il seed per la riproducibilità
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

root_dir = '/home/vboxuser/Desktop/Progetto LAB-AI/data'
transform = transforms.Compose({
    transforms.Resize((224,224)),
    transforms.ToTensor(),
})

dataset = GrapeDataset100(root_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=2000, shuffle=True)

imgs, _ = next(iter(data_loader))
mean = torch.mean(imgs, dim=[0,2,3])
std = torch.std(imgs, dim=[0,2,3])
print(f"Mean: {mean}")
print(f"Standard deviation: {std}")





