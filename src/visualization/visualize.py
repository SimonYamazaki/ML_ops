#%%
from src.models.main import load_checkpoint
from src.data.data import mnist
import torch 
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm


model = load_checkpoint('../../models/checkpoint07062021131555.pth')

children_counter = 0
for n,c in model.named_children():
    print("Children Counter: ",children_counter," Layer Name: ",n,)
    children_counter+=1

"""
class new_model(nn.Module):
    def __init__(self, model, output_layer):
        super().__init__()
        self.output_layer = output_layer
        self.pretrained = model
        self.children_list = []
        for n,c in self.pretrained.named_children():
            self.children_list.append(c)
            if n == self.output_layer:
                break

        self.net = nn.Sequential(*self.children_list)
        self.pretrained = None
        
    def forward(self,x):
        x = self.net(x)
        print(x.shape)

        return x


feature_extractor = new_model(model,4)"""

feature_extractor = nn.Sequential(*list(model.children())[:-1])

_, testloader = mnist()

emb_data = []
labels_plot = []

for images, labels in testloader:
    with torch.no_grad():
        model.eval()
        features = feature_extractor(images)
        f_emb = TSNE(n_components=2).fit_transform(features)
        emb_data.append(f_emb)
        labels_plot.append(labels)

emb_data = np.array(emb_data[:-1])
emb_data = emb_data.reshape(emb_data.shape[0]*64,2)

labels_plot = np.array(labels_plot[:-1]).reshape(-1)


colors = cm.rainbow(np.linspace(0, 1, 10))
plt.scatter(emb_data[:,0],emb_data[:,1],color=colors[labels_plot])
plt.savefig('../../reports/figures/feature_embeddings.png')



