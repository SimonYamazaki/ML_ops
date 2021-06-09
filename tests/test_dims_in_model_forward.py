# content of test_sysexit.py
import pytest

from src.data.data import mnist
from src.models.model import MyAwesomeModel

import torch 

trainset,testset,trainloader,testloader = mnist(get_dataset=True)
model = MyAwesomeModel()

def model_forward():
    with torch.no_grad():
        model.eval()
        train_data = next(iter(trainloader))
        input_data = train_data[0]
        input_data = input_data.squeeze()
        model(input_data)

def test_value_error():
    with pytest.raises(ValueError):
        model_forward()

