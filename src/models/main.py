import argparse
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
from matplotlib.figure import Figure
from torch import nn, optim

from src.data.data import mnist
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


class TrainOREvaluate(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(
        self, manual_parse=None, log_wandb=True, testing_mode=False, test_layer=3
    ):
        self.learning_rate = 1e-4
        self.filters = 32
        self.epochs = 2
        self.kernel_size = 5
        self.fc_features = 128
        self.image_dim = 28
        self.n_classes = 10

        self.testing_mode = testing_mode
        self.test_layer = test_layer
        self.log_wandb = log_wandb

        if self.log_wandb:
            with open("../../wandb.key", "r") as handle:
                wandb_key = handle.readlines()[0]

            wandb.login(key=wandb_key)
            wandb.init(project="FashionMNIST_CNN", entity=None)

            config = wandb.config
            config.learning_rate = self.learning_rate
            config.filters = self.filters
            config.epochs = self.epochs
            config.kernel_size = self.kernel_size
            config.fc_features = self.fc_features
            config.image_dim = self.image_dim
            config.n_classes = self.n_classes

        if manual_parse == None:
            parser = argparse.ArgumentParser(
                description="Script for either training or evaluating",
                usage="python main.py <command>",
            )
            parser.add_argument("command", help="Subcommand to run")
            args = parser.parse_args(sys.argv[1:2])
            if not hasattr(self, args.command):
                print("Unrecognized command")

                parser.print_help()
                exit(1)
            # use dispatch pattern to invoke method with same name
            getattr(self, args.command)()
        else:
            getattr(self, manual_parse)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement training loop here
        model = MyAwesomeModel(
            self.image_dim,
            self.kernel_size,
            self.filters,
            self.fc_features,
            self.n_classes,
        )
        trainloader, _ = mnist()
        n_batches = len(trainloader)
        model.train()

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        train_losses = []
        train_accs = []

        for e in range(self.epochs):
            print("epoch {0}/{1}".format(e, self.epochs))
            running_loss = 0
            t_acc = 0
            batch_ii = 0

            for ii, (images, labels) in enumerate(trainloader):
                print("batch {0}/{1}".format(batch_ii, n_batches))
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
                batch_ii += 1

                if self.testing_mode:
                    if ii == 0:
                        self.b_weights = model.parameters()
                    elif ii == 4:
                        break

            else:
                train_losses.append(running_loss)
                t_acc = t_acc / n_batches
                train_accs.append(t_acc)
                wandb.log({"loss": running_loss})
                wandb.log({"accuracy": t_acc})
                print("Loss: {0} Accuracy: {1} ".format(running_loss, t_acc))

        if self.testing_mode:
            self.init_weights = iter(self.b_weights)
            self.later_weights = iter(model.parameters())
            for ii in range(self.test_layer + 1):
                self.layer_init_weights = next(self.init_weights)
                self.layer_later_weights = next(self.later_weights)

            self.parameter_changes = self.layer_init_weights - self.layer_later_weights
            return self.parameter_changes
        else:
            now = datetime.now()
            dt_string = now.strftime("%d%m%Y%H%M%S")

            y_axis = ["Loss", "Accuracy"]
            measures = [train_losses, train_accs]

            for i in range(len(measures)):
                plt.figure()
                plt.plot(measures[i])
                plt.xlabel("epoch")
                plt.ylabel(y_axis[i])
                if self.log_wandb:
                    wandb.log({f"Train {y_axis[0]}": wandb.Image(plt)})
            # wandb.log({"img": [wandb.Image(im, caption="Cafe")]})

            checkpoint = {
                "image_dim": self.image_dim,
                "kernel_size": self.kernel_size,
                "filters": self.filters,
                "fc_features": self.fc_features,
                "n_classes": self.n_classes,
                "state_dict": model.state_dict(),
            }

            torch.save(checkpoint, "../../models/" + "checkpoint_" + dt_string + ".pth")

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Evaluation arguments")
        parser.add_argument("--load_model_from", default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement evaluation logic here
        # if args.load_model_from:

        model = load_checkpoint(args.load_model_from)
        # model = torch.load(args.load_model_from)
        _, testloader = mnist()

        n_batches = len(testloader)
        val_acc = 0
        n_val = 0

        with torch.no_grad():

            model.eval()

            for images, labels in testloader:
                print("batch {0}/{1}".format(n_val, n_batches))
                ps = torch.exp(model(images))

                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy_val = torch.mean(equals.type(torch.FloatTensor))
                val_acc += accuracy_val
                n_val += 1
            else:
                print(f"val_Accuracy: {val_acc.item()*100/n_batches}%")


if __name__ == "__main__":
    TrainOREvaluate()
