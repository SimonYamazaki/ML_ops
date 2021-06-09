import sys
import argparse

import torch
from torch import nn,optim

from datetime import datetime

from src.data.data import mnist
from src.models.model import MyAwesomeModel

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
from matplotlib.figure import Figure

import numpy as np
import wandb


with open("../../wandb.key" , "r") as handle:
    wandb_key = handle.readlines()[0]

wandb.login(key=wandb_key)
wandb.init(project='FashionMNIST_CNN', entity=None)

config = wandb.config
config.learning_rate = 1e-4
config.filters = 32
config.epochs = 1
config.kernel_size = 5
config.fc_features = 128
config.image_dim = 28 
config.n_classes = 10

def load_checkpoint(filepath,get_feature_layer=False):
    checkpoint = torch.load(filepath)
    model = MyAwesomeModel(checkpoint['image_dim'],checkpoint['kernel_size'],checkpoint['filters'],
                           checkpoint['fc_features'],checkpoint['n_classes'],get_feature_layer=get_feature_layer)
    model.load_state_dict(checkpoint['state_dict'])
    return model



class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        image_dim = config.image_dim
        kernel_size = config.kernel_size
        filters = config.filters
        fc_features = config.fc_features
        n_classes = config.n_classes
        
        model = MyAwesomeModel(image_dim,kernel_size,filters,fc_features,n_classes)
        trainloader, _ = mnist()
        n_batches = len(trainloader)
        model.train()
        
        criterion = nn.NLLLoss()  
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        epochs = config.epochs
        
        train_losses = []
        train_accs = [] 

        for e in range(epochs):
            print('epoch {0}/{1}'.format(e,epochs))
            running_loss = 0
            t_acc= 0
            n_train = 0
            
            for images, labels in trainloader:
                print('batch {0}/{1}'.format(n_train,n_batches))
                optimizer.zero_grad()

                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)   
                train_acc = torch.mean(equals.type(torch.FloatTensor))

                t_acc += train_acc
                n_train += 1
            else:
                train_losses.append(running_loss)
                t_acc = t_acc/n_batches
                train_accs.append(t_acc)
                wandb.log({"loss": running_loss})
                wandb.log({"accuracy": t_acc})
                print('Loss: {0} Accuracy: {1} '.format(running_loss,t_acc))
        
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y%H%M%S")

        y_axis = ['Loss', 'Accuracy']
        measures = [train_losses, train_accs]
        """"with PdfPages('../../reports/figures/train_measures'+dt_string+'.pdf') as pages:
            for i in range(2):
                fig = Figure()
                ax = fig.gca()
                ax.plot(measures[i])
                ax.set_xlabel('epoch')
                ax.set_ylabel(y_axis[i])
                canvas = FigureCanvasPdf(fig)
                canvas.print_figure(pages)"""

        for i in range(len(measures)):
            plt.figure()
            plt.plot(measures[i])
            plt.xlabel('epoch')
            plt.ylabel(y_axis[i])
            wandb.log({f"Train {y_axis[0]}": wandb.Image(plt)})
        #wandb.log({"img": [wandb.Image(im, caption="Cafe")]})

        checkpoint = {'image_dim': image_dim,
              'kernel_size': kernel_size,
              'filters': filters,
              'fc_features': fc_features,
              'n_classes': n_classes,
              'state_dict': model.state_dict()}

        torch.save(checkpoint, '../../models/' + 'checkpoint_' + dt_string + '.pth')

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Evaluation arguments')
        parser.add_argument('--load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        #if args.load_model_from:
        
        model = load_checkpoint(args.load_model_from)
        #model = torch.load(args.load_model_from)
        _, testloader = mnist()

        n_batches = len(testloader)
        val_acc = 0
        n_val = 0

        with torch.no_grad():

            model.eval()

            for images, labels in testloader:
                print('batch {0}/{1}'.format(n_val,n_batches))
                ps = torch.exp(model(images))

                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)   
                accuracy_val = torch.mean(equals.type(torch.FloatTensor))
                val_acc += accuracy_val
                n_val += 1
            else:    
                print(f'val_Accuracy: {val_acc.item()*100/n_batches}%')
            
if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    