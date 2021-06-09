
from src.data.data import mnist
from src.models.model import MyAwesomeModel

import torch 

class Test_model:
    trainset,testset,trainloader,testloader = mnist(get_dataset=True)
    model = MyAwesomeModel()

    def test_model_shape(self):
        with torch.no_grad():
            self.model.eval()
            train_data = next(iter(self.trainloader))
            input_data = train_data[0]
            output_data = self.model(input_data)
            assert output_data.shape[0] == input_data.shape[0] and output_data.shape[1] == len(self.testset.classes)


