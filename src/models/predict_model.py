#%%
import argparse
import sys

import numpy as np
import torch
from torchvision import transforms

from src.data.data import get_infer_numpy_data
from src.models.main import load_checkpoint

#%%
"""
parser = argparse.ArgumentParser(description='Evaluation arguments')
parser.add_argument('--load_model_from', default="")
parser.add_argument('--load_data_dir_from', default="")

args = parser.parse_args(sys.argv[1:])
print(args)
"""
#%%

input_path = "../../data/FashionMNIST/processed/test.pt"
output_path = "../../data/FashionMNIST/processed/infer_imgs_100"


def get_infer_numpy_data(input_path, output_path):
    _, _ = mnist()
    data = torch.load(input_path)
    infer_imgs = data[0][:100, :, :].numpy()
    np.save(output_path, infer_imgs)


#%%
"""
numpy_data = np.load('../../data/FashionMNIST/processed/infer_imgs_100.npy')
datat = torch.from_numpy(numpy_data)

if len(datat.shape) != 4:
    datat = datat.reshape(numpy_data.shape[0],1,numpy_data.shape[1],numpy_data.shape[2])

model = load_checkpoint('../../models/checkpoint.pth')

print(datat.shape)
print(datat.dtype)

with torch.no_grad():
    model.eval()
    ps = torch.exp(model(datat))
    top_p, top_class = ps.topk(1, dim=1)
"""


model = load_checkpoint("../../models/checkpoint.pth")
from src.data.data import mnist

_, testloader = mnist()

images, labels = next(iter(testloader))
with torch.no_grad():
    model.eval()
    ps = torch.exp(model(images))
    top_p, top_class = ps.topk(1, dim=1)
    classes = list(top_class.numpy().squeeze())

f = open("../../infer_classes.txt", "w")
f.write(str(classes))
f.close()


"""
def infer_model(make_numpy_data=False):
    if make_numpy_data:
        #numpy_data = get_infer_numpy_data(path='../../data/FashionMNIST/processed/training.pt')
        numpy_data = get_infer_numpy_data(path=args.load_data_dir_from)

    #data = torch.load(args.load_data_dir_from)
    numpy_data = np.load(args.load_data_dir_from)
    datat = torch.from_numpy(numpy_data)
    if len(datat.shape) != 4:
        datat = datat.reshape(numpy_data.shape[0],1,numpy_data.shape[1],numpy_data.shape[2])

    model = load_checkpoint(args.load_model_from)

    with torch.no_grad():
        model.eval()
        ps = torch.exp(model(datat))
        top_p, top_class = ps.topk(1, dim=1)


if __name__ == '__main__':
    infer_model()

    """
