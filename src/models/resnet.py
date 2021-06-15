import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
print(model)
script_model = torch.jit.script(model)
script_model.save('../../models/deployable_resnet_model.pt')
