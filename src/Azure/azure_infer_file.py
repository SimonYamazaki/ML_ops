import json
import joblib
import numpy as np
from azureml.core.model import Model
import torch 
from src.models.model import MyAwesomeModel

def load_checkpoint(filepath, get_feature_layer=False):
    checkpoint = torch.load(filepath)
    model = MyAwesomeModel(
        checkpoint["image_dim"],
        checkpoint["kernel_size"],
        checkpoint["filters"],
        checkpoint["fc_features"],
        checkpoint["n_classes"],
        get_feature_layer=get_feature_layer,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path('working_MNIST_CNN')
    model = load_checkpoint(model_path)
    #model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    dataT = torch.from_numpy(data)
    # Get a prediction from the model
    with torch.no_grad():
        model.eval()
        ps = torch.exp(model(dataT))
        top_p, top_class = ps.topk(1, dim=1)
    
    # Get the corresponding classname for each prediction (0 or 1)
    classnames = ['0', '1','2','3','4','5','6','7','8','9']
    predicted_classes = []
    for prediction in top_class:
        predicted_classes.append(classnames[prediction])
    # Return the predictions as JSON
    return json.dumps(predicted_classes)