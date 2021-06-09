import torch
import numpy as np
from torchvision import datasets, transforms

def mnist(get_dataset=False,batch_size=64):
    
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Download and load the training data
    trainset = datasets.FashionMNIST('/Users/simonyamazaki/ML_ops/data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    #trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    
    # Download and load the test data
    testset = datasets.FashionMNIST('/Users/simonyamazaki/ML_ops/data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    if get_dataset:
        return trainset, testset, trainloader, testloader
    else:
        return trainloader, testloader


def get_infer_numpy_data(input_path='../../data/FashionMNIST/processed/test.pt',
                        output_path='../../data/FashionMNIST/processed/infer_imgs_100'):
    _,_ = mnist()
    data = torch.load(input_path)
    infer_imgs = data[0][:100,:,:].numpy()
    np.save(output_path, infer_imgs)
