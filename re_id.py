import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)

#modello con pesi addestrati da me
outputs = torch.load('embeddings/outputs.pth')

embeddings = outputs['embeddings']
labels = outputs['labels']

soglia = 0.85

print(f'\nsoglia per questo test = {soglia}\n')



true_y_list = torch.empty(0, dtype=torch.float)
pred_y_list = torch.empty(0, dtype=torch.float)

with tqdm(total=len(embeddings), desc='Calcolo Metriche Re ID', unit='immagine') as progress_bar:
    for i in range(len(embeddings)):
             
        cosine_similarities = F.cosine_similarity(embeddings[i], embeddings)
        
        y_true = (labels[i] == labels).float()

        # cambiare soglia in base a metriche
        y_pred = (cosine_similarities > soglia).float()

        true_y_list = torch.cat((true_y_list, y_true), dim=0)
        pred_y_list = torch.cat((pred_y_list, y_pred), dim=0)

        progress_bar.update(1)


true_y_list = true_y_list.flatten().tolist()
pred_y_list = pred_y_list.flatten().tolist()

accuracy_cos = 100. * metrics.accuracy_score(true_y_list, pred_y_list)
print(f"Accuracy = {accuracy_cos:.2f}%\n")


precision_cos = 100. * metrics.precision_score(true_y_list, pred_y_list)
print(f"Precision = {precision_cos:.2f}%\n")


recall_cos = 100. * metrics.recall_score(true_y_list, pred_y_list)
print(f"Recall = {recall_cos:.2f}%\n")


f1_cos = 100. * metrics.f1_score(true_y_list, pred_y_list)
print(f"F1-score = {f1_cos:.2f}%\n")


cm_cos = metrics.confusion_matrix(true_y_list, pred_y_list)
cm_cos_display = metrics.ConfusionMatrixDisplay(cm_cos).plot()
plt.savefig(f'conf_matrix_{soglia}.png')
plt.show()