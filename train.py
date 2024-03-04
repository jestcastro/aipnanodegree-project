import argparse
from PIL import Image
import numpy as np
import pathlib

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.vgg import VGG16_Weights
import time
from utils import CNNArch, print_validation_progress


class Train:
    def __init__(
        self,
        data_dir,
        save_dir,
        arch,
        gpu,
        load_dir,
        learning_rate,
        hidden_units,
        epochs,
    ):
        self.data_dir = data_dir.rstrip("/")
        self.save_dir = save_dir.rstrip("/")
        self.arch = arch
        self.gpu = gpu
        self.load_dir = load_dir
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.epochs = epochs

    def set_device(self):
        if not self.gpu:
            self.device = torch.device("cpu")
            return
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            return
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            return

    def load_data(self):
        train_dir = self.data_dir + "/train"
        valid_dir = self.data_dir + "/valid"
        test_dir = self.data_dir + "/test"
        normalization_mean = [0.485, 0.456, 0.406]
        normalization_std = [0.229, 0.224, 0.225]
        validation_test_transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std),
        ]
        dataloaders_batch_size = 64
        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomRotation(30),
                    transforms.RandomResizedCrop(224, scale=(0.05, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=normalization_mean, std=normalization_std
                    ),
                ]
            ),
            "validation": transforms.Compose(validation_test_transforms),
            "test": transforms.Compose(validation_test_transforms),
        }
        self.image_datasets = {
            "train": datasets.ImageFolder(
                train_dir, transform=data_transforms.get("train")
            ),
            "validation": datasets.ImageFolder(
                valid_dir, transform=data_transforms.get("validation")
            ),
            "test": datasets.ImageFolder(
                test_dir, transform=data_transforms.get("test")
            ),
        }
        self.dataloaders = {
            "train": torch.utils.data.DataLoader(
                self.image_datasets.get("train"),
                batch_size=dataloaders_batch_size,
                shuffle=True,
            ),
            "validation": torch.utils.data.DataLoader(
                self.image_datasets.get("validation"), batch_size=dataloaders_batch_size
            ),
            "test": torch.utils.data.DataLoader(
                self.image_datasets.get("test"), batch_size=dataloaders_batch_size
            ),
        }

    def load_model(self):
        if self.arch == CNNArch.VGG16:
            self.model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.input_size = 25088
        else:
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.input_size = 512

    def freeze_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def create_classifier(self):
        classifier = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_units, 102),
            nn.LogSoftmax(dim=1),
        )
        if self.arch == CNNArch.VGG16:
            self.model.classifier = classifier
        else:
            self.model.fc = classifier

    def get_classifier(self):
        if self.arch == CNNArch.VGG16:
            return self.model.classifier
        else:
            return self.model.fc

    def run_epochs(self):
        steps = 0
        running_loss = 0
        print_frequency = 5
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(
            self.get_classifier().parameters(), lr=self.learning_rate
        )
        header_already_printed = False
        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.epochs):
            for inputs, labels in self.dataloaders["train"]:
                steps += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                logps = self.model.forward(inputs)
                loss = criterion(logps, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_frequency == 0:
                    validation_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in self.dataloaders["validation"]:
                            inputs, labels = inputs.to(self.device), labels.to(
                                self.device
                            )
                            logps = self.model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            validation_loss += batch_loss.item()

                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(
                                equals.type(torch.FloatTensor)
                            ).item()

                    header_already_printed = print_validation_progress(
                        header_already_printed,
                        epoch,
                        self.epochs,
                        running_loss,
                        print_frequency,
                        validation_loss,
                        accuracy,
                        self.dataloaders["validation"],
                    )
                    running_loss = 0
                    self.model.train()

    def save_checkpoint(self):
        self.model.class_to_idx = self.image_datasets["train"].class_to_idx
        checkpoint = {
            "input_size": self.input_size,
            "output_size": 102,
            "arch": self.arch,
            "classifier": (
                self.model.classifier if self.arch == CNNArch.VGG16 else self.model.fc
            ),
            "state_dict": self.model.state_dict(),
            "class_to_idx": self.model.class_to_idx,
        }
        if not pathlib.Path(self.save_dir).exists():
            pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, self.save_dir + "/checkpoint.pth")

    def train_model(self):
        start_time = time.time()
        self.load_data()
        self.load_model()
        self.freeze_parameters()
        self.create_classifier()
        self.run_epochs()
        self.save_checkpoint()
        print(f"Training time: {(time.time() - start_time)/60:.3f} minutes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to the folder of images")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=".",
        help="Path to the folder to save the checkpoint",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=CNNArch.VGG16,
        choices=[CNNArch.VGG16, CNNArch.RESNET],
        help="CNN Architecture",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument(
        "--load_dir",
        type=str,
        default=".",
        help="Path to the folder to load the checkpoint",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--hidden_units", type=int, default=512, help="Number of hidden units"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    args = parser.parse_args()

    training_instance = Train(
        args.data_dir,
        args.save_dir,
        args.arch,
        args.gpu,
        args.load_dir,
        args.learning_rate,
        args.hidden_units,
        args.epochs,
    )
    training_instance.set_device()
    training_instance.train_model()


if __name__ == "__main__":
    main()
