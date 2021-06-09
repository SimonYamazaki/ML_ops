
import torch

from src.data.data import mnist


class Test_data:
    trainset,testset ,_, _ = mnist(get_dataset=True)
        
    def test_dataset_len(self):
        assert len(self.trainset) == 60000 and len(self.testset) == 10000

    def test_datashape(self):
        assert self.trainset.data.shape[-2:]==torch.Size([28,28]) and self.testset.data.shape[-2:]==torch.Size([28,28])

    def test_n_labels(self):
        assert self.trainset.targets.shape==torch.Size([60000]) and self.testset.targets.shape==torch.Size([10000])

    def test_label_representation(self):
        assert len(torch.unique(self.trainset.targets))==len(self.trainset.classes) and len(torch.unique(self.trainset.targets))==len(self.trainset.classes)
