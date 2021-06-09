

import pytest
import torch

from src.models.main import TrainOREvaluate


@pytest.mark.parametrize("x", [0, 1, 2])

class Test_training:

    def test_training(self,x):
        self.training_model = TrainOREvaluate(manual_parse="train",log_wandb=False,testing_mode=True,test_layer=x)
        self.diff_params = torch.sum(self.training_model.parameter_changes)
        assert self.diff_params == 0
