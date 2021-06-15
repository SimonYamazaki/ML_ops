from azureml.core import Workspace, Model
from azureml.core.authentication import InteractiveLoginAuthentication
ia = InteractiveLoginAuthentication(tenant_id='f251f123-c9ce-448e-9277-34bb285911d9')

ws = Workspace.get(name='ML_ops',
                     subscription_id='3256dba5-2e98-4c8b-b9a4-f34b68b28b8b',
                     resource_group='Best_recource_group',auth=ia)

#register a model from local file
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

#model = Model.register(ws, model_name="working_MNIST_CNN", model_path=dir_path+"/models/checkpoint_15062021123318.pth")

model = ws.models['working_MNIST_CNN']

