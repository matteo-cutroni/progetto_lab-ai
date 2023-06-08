import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import train
from tqdm import tqdm
from sklearn import metrics

#full dataset già diviso in train.py
dataset = train.re_id_dataset

batch_size = 32
half_batch = batch_size // 2

data_loader= DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)


#modello con pesi addestrati da me
model = torch.load("models/resnet18_tb32.pt", map_location=train.device)

#rimuove ultimo layer fc e logsoftmax
model.fc = model.fc[:2]

model.eval()


true_y = []
pred_sq = []
pred_cos = []

with torch.no_grad(), tqdm(total=len(data_loader), desc='Calcolo Metriche Re ID', unit='batch') as progress_bar:
    for images, labels in data_loader:
        images = images.to(train.device) 
        labels = labels.to(train.device)

        # divide il batch in due parti da confrontare
        images1 = images[:half_batch]
        images2 = images[half_batch:]

        labels1 = labels[:half_batch]
        labels2 = labels[half_batch:]

        embeddings1 = model(images1)
        embeddings2 = model(images2)

        squared_distances = (embeddings1 - embeddings2).pow(2).sum(1)
        cosine_similarities = F.cosine_similarity(embeddings1, embeddings2)
        
        y_true = (labels1 == labels2).float()

        # cambiare soglie in base a metriche
        y_pred_sq = (squared_distances < 650).float()  
        y_pred_cos = (cosine_similarities > 0.845).float()


        for i in range(half_batch):
                true_y.append(y_true[i].item())
                pred_sq.append(y_pred_sq[i].item())
                pred_cos.append(y_pred_cos[i].item())

        progress_bar.update(1)


accuracy_sq = 100. * metrics.accuracy_score(true_y, pred_sq)
print(f"\nAccuracy Square Distance: {accuracy_sq:.2f}%")

accuracy_cos = 100. * metrics.accuracy_score(true_y, pred_cos)
print(f"Accuracy Cosine Similarity: {accuracy_cos:.2f}%\n")


precision_sq = 100. * metrics.precision_score(true_y, pred_sq)
print(f"Precision Square Distance = {precision_sq:.2f}")
precision_cos = 100. * metrics.precision_score(true_y, pred_cos)
print(f"Precision Cosine Similarity = {precision_cos:.2f}\n")


recall_sq = 100. * metrics.recall_score(true_y, pred_sq)
print(f"Recall Square Distance = {recall_sq:.2f}")
recall_cos = 100. * metrics.recall_score(true_y, pred_cos)
print(f"Recall Cosine Similarity = {recall_cos:.2f}\n")


f1_sq = 100. * metrics.f1_score(true_y, pred_sq)
print(f"F1-score Square Distance = {f1_sq:.2f}")
f1_cos = 100. * metrics.f1_score(true_y, pred_cos)
print(f"F1-score Cosine Similarity = {f1_cos:.2f}\n")


cm_sq = metrics.confusion_matrix(true_y, pred_sq)
cm_sq_display = metrics.ConfusionMatrixDisplay(cm_sq)
cm_cos = metrics.confusion_matrix(true_y, pred_cos)
cm_cos_display = metrics.ConfusionMatrixDisplay(cm_cos)
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(121)
cm_sq_display.plot(ax=ax1)
ax1.title.set_text("Confusion Matrix Square Distance")
ax2 = fig.add_subplot(122)
cm_cos_display.plot(ax=ax2)
ax2.title.set_text("Confusion Matrix Cosine Similarity")
plt.show()