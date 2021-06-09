#%%
from src.models.main import load_checkpoint
from src.data.data import mnist
import torch 
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm


model = load_checkpoint('../../models/checkpoint_07062021173450.pth',get_feature_layer=True)

children_counter = 0
for n,c in model.named_children():
    print("Children Counter: ",children_counter," Layer Name: ",n,)
    children_counter+=1


#feature_extractor = nn.Sequential(*list(model.modules())[:-1])
model.eval()
feature_extractor = model

_, testloader = mnist()

f_data = []
labels_plot = []

for ii, (images, labels) in enumerate(testloader):
    print(ii)
    with torch.no_grad():
        model.eval()
        features = feature_extractor(images)
        f_data.append(features)
        labels_plot.append(labels)
    if ii==5:
        break

f_data = torch.cat(f_data)
labels_plot = torch.cat(labels_plot)

f_emb = TSNE(n_components=2).fit_transform(f_data)

colors = cm.rainbow(np.linspace(0, 1, 10))
plt.scatter(f_emb[:,0],f_emb[:,1],color=colors[labels_plot])
plt.savefig('../../reports/figures/feature_embeddings.png')



