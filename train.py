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
from sklearn import metrics
import copy


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



def train(loss_vector, acc_vector, log_interval=200, logging_dir='runs/our_experiment'):
    writer = SummaryWriter(logging_dir)

    model.train()

    train_loss, correct = 0.0, 0

    for batch_idx, (data, target) in enumerate(train_loader):

        data= data.to(device)
        target = target.to(device)

        output = model(data)

        loss = criterion(output, target)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        #estrae la predizione dal tensore con probabilità per ogni classe (output.data)
        _, predicted = torch.max(output.data, 1)

        #per calcolo accuracy
        correct += (predicted == target).sum().item()

        # per calcolo loss media
        train_loss += loss.item() * data.size(0)

        #ogni tanto stampa loss di questo batch
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.data.item():.4f}')

    #registra loss media alla fine dell'epoch
    train_loss /= len(train_loader.dataset)
    loss_vector.append(train_loss)
    writer.add_scalar('Loss/Training', train_loss, epoch)

    train_accuracy = 100. * correct / len(train_loader.dataset)
    acc_vector.append(train_accuracy)

    writer.close()



def validate(loss_vector, acc_vector_reid, acc_vector_cl, logging_dir='runs/our_experiment'):
    writer = SummaryWriter(logging_dir)

    model.eval()

    val_loss, correct = 0, 0


    true_y = []
    pred_cos = []

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

            emb_model = copy.deepcopy(model)
            emb_model.fc = emb_model.fc[:2]

            # divide il batch in due parti da confrontare
            images1 = data[:half_batch]
            images2 = data[half_batch:]

            labels1 = target[:half_batch]
            labels2 = target[half_batch:]

            embeddings1 = emb_model(images1)
            embeddings2 = emb_model(images2)

            cosine_similarities = F.cosine_similarity(embeddings1, embeddings2)

            y_true = (labels1 == labels2).float()

            # cambiare soglie in base a metriche
            y_pred_cos = (cosine_similarities > 0.7).float()


            for i in range(half_batch):
                    true_y.append(y_true[i].item())
                    pred_cos.append(y_pred_cos[i].item())

    accuracy_cos = 100. * metrics.accuracy_score(true_y, pred_cos)
    print(f"\nAccuracy Cosine Similarity: {accuracy_cos:.2f}%\n")
    acc_vector_reid.append(accuracy_cos)

    cm_cos = metrics.confusion_matrix(true_y, pred_cos)
    metrics.ConfusionMatrixDisplay(cm_cos).plot()
    plt.show()

    #registra loss media e accuracy
    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)
    loss_vector.append(val_loss)
    acc_vector_cl.append(val_accuracy)

    # Scrivi la perdita e l'accuratezza su TensorBoard
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
    writer.close()

    print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({val_accuracy:.0f}%)\n\n')


#per non eseguire con import
if __name__ == "__main__":


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

    #nuova transform con normalizzazione (valori di mean e std già calcolati)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3149, 0.2998, 0.2792], std=[0.1808, 0.1984, 0.2022])
    ])

    #full dataset con immagini normalizzate
    full_dataset = GrapeDataset100(root_dir, transform=transform)

    # divisione del dataset in training e validation
    train_set_size = int(0.8 * len(full_dataset))
    val_set_size = len(full_dataset) - train_set_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_set_size, val_set_size])

    print(len(train_set), len(val_set))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True, drop_last=True)


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
    train_loss, val_loss, val_acc_reid, train_acc_cl, val_acc_cl = [],[],[],[],[]


    #per validation
    half_batch = val_loader.batch_size // 2


    for epoch in range(1, num_epochs + 1):
        train(train_loss, train_acc_cl)
        validate(val_loss, val_acc_reid, val_acc_cl)

    # matplotlib plot
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    plt.plot(val_acc_reid, label='Validation Accuracy per Re-ID')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

    plt.plot(train_acc_cl, label='Training Accuracy')
    plt.plot(val_acc_cl, label='Validation Accuracy')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

    #salva modello
    torch.save(model, f"resnet18_new_{num_epochs}e.pt")

