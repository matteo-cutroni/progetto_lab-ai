import torch
from torch.utils.data import DataLoader
from train import GrapeDataset100
from tqdm import tqdm
from torchvision import transforms
import numpy as np

test_path = '/home/vboxuser/Desktop/Progetto LAB-AI/test_set'
model_path = 'models/resnet18_new32.pt'

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)

# Fissare il seed per la riproducibilità
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

#stessa transform di train
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3149, 0.2998, 0.2792], std=[0.1808, 0.1984, 0.2022])
])

#full dataset già diviso in train.py
dataset = GrapeDataset100(test_path, transform=transform)

batch_size = 32

data_loader= DataLoader(dataset, batch_size=batch_size, shuffle=False)


#modello con pesi addestrati da me
model = torch.load(model_path, map_location=device)

#rimuove ultimo layer fc e logsoftmax
model.fc = model.fc[:2]

model.to(device)

model.eval()

all_embeddings = []
all_labels = []

with torch.no_grad(), tqdm(total=len(data_loader), desc='Calcolo gli embeddings', unit='batch') as progress_bar:
    for images, labels in data_loader:
        images = images.to(device) 
        labels = labels.to(device)

        embeddings = model(images)

        all_embeddings.append(embeddings)
        all_labels.append(labels)

        progress_bar.update(1)

all_embeddings = torch.cat(all_embeddings, dim=0)
all_labels = torch.cat(all_labels, dim=0)


torch.save({'embeddings': all_embeddings, 'labels': all_labels}, 'embeddings/outputs.pth')