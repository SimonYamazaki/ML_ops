
#%%
import numpy as np
import torch

data = torch.load('../../data/FashionMNIST/processed/training.pt')

infer_imgs = data[0][:100,:,:].numpy()

np.save('../../data/FashionMNIST/processed/infer_imgs_100', infer_imgs)

