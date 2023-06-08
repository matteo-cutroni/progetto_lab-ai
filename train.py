import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

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

root_dir = '/home/vboxuser/Desktop/Progetto LAB-AI/data'


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
        data_point = Image.open(self.data[idx])
        data_label = self.label[idx]

        if self.transform is not None:
            data_point = self.transform(data_point)

        return data_point, data_label


#nuova transform con normalizzazione (valori di mean e std già calcolati)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3149, 0.2998, 0.2792], std=[0.1808, 0.1984, 0.2022])
])

#full dataset con immagini normalizzate
full_dataset = GrapeDataset100(root_dir, transform=transform)

# divisione del dataset in parte per il training con classificazione
class_dataset_size = int(0.8 * len(full_dataset))
re_id_dataset_size = len(full_dataset) - class_dataset_size
class_dataset, re_id_dataset = torch.utils.data.random_split(full_dataset, [class_dataset_size, re_id_dataset_size])


def train(loss_vector, log_interval=100, logging_dir='runs/our_experiment'):
    writer = SummaryWriter(logging_dir)

    model.train()

    train_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):

        data= data.to(device)
        target = target.to(device)

        output = model(data)

        loss = criterion(output, target)

        optimizer.zero_grad()

        loss.backward()
        
        optimizer.step()

        # per calcolo loss media 
        train_loss += loss.item() * data.size(0)

        #ogni tanto stampa loss di questo batch
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.data.item():.4f}')
    
    #registra loss media alla fine dell'epoch
    train_loss /= len(train_loader.dataset)
    loss_vector.append(train_loss)
    writer.add_scalar('Loss/Training', train_loss, epoch)

    writer.close()


def validate(loss_vector, logging_dir='runs/our_experiment'):
    writer = SummaryWriter(logging_dir)

    model.eval()

    val_loss, correct = 0, 0

    # pesi non devono essere aggiornati in validation
    with torch.no_grad():
        for data, target in val_loader:
          
            data, target = data.to(device), target.to(device)

            output = model(data)
            
            loss = criterion(output, target)
            
            #per calcolo loss media
            val_loss += loss.item() * data.size(0)
            
            #estrae la predizione dal tensore con probabilità per ogni classe (output.data)
            _, predicted = torch.max(output.data, 1)
            
            #per calcolo accuracy
            correct += (predicted == target).sum().item()
    
    #registra loss media e accuracy
    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)
    
    loss_vector.append(val_loss)

    # Scrivi la perdita e l'accuratezza su TensorBoard
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
    writer.close()
    
    print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({val_accuracy:.0f}%)\n')


def test(logging_dir='runs/our_experiment'):
    writer = SummaryWriter(logging_dir)

    model.eval()

    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            loss = criterion(output, target)
            
            test_loss += loss.item() * data.size(0)
            
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)

    writer.add_scalar('Loss/Test', test_loss, epoch)
    writer.add_scalar('Accuracy/Test', test_accuracy, epoch)
    writer.close()
    
    return test_loss, test_accuracy

#per non eseguire con import
if __name__ == "__main__":


    #divisione di dataset in training set, valildation set e test set
    train_size = int(0.7 * len(class_dataset))
    val_size = int((len(class_dataset) - train_size)/2)
    test_size = len(class_dataset) - (train_size + val_size)
    print(f"dataset: {len(class_dataset)} immagini")
    print(f"training set: {train_size} immagini")
    print(f"validation set: {val_size} immagini")
    print(f"test set: {test_size} immagini")

    train_set, val_set, test_set = torch.utils.data.random_split(class_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader= DataLoader(test_set, batch_size=32, shuffle=False)


    #carica modello con pesi preaddestrati
    model = models.resnet18(weights='IMAGENET1K_V1')

    num_features = model.fc.in_features


    # Freeze the parameters 
    for param in model.parameters():
        param.requires_grad = False 

    #modifica ultimo layer fc con il sequential
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Linear(128,100),
        nn.LogSoftmax(dim=1)
    )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    

    num_epochs = 20

    #per matplotlib
    train_loss, val_loss = [],[]


    for epoch in range(1, num_epochs + 1):
        train(train_loss)
        validate(val_loss)

    # matplotlib plot
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    #salva modello
    torch.save(model, f"resnet18_tb{train_loader.batch_size}.pt")
    


    test_loss, test_acc = test()
    print('Test Loss = ', test_loss, 'Test Accuracy = ', test_acc)


    #plt 3x3 con immagini e predizione
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(test_set), size=(1,)).item()
        img, label = test_set[sample_idx]
        label = torch.tensor(label)

        output = model(img.unsqueeze(0).to(device))
        _,predicted = torch.max(output.data, 1)
        figure.add_subplot(rows, cols, i)
        if (label == predicted.item()):
            plt.title(predicted.item(), color='green')
        else:
            plt.title(f"{label}" +' != '+ f"{predicted.item()}", color='red')
            plt.axis("off")
            img = img.swapaxes(0,1)
            img = img.swapaxes(1,2)
            plt.imshow(img)
    plt.show()