import argparse
from PIL import Image
import torch
from torchvision import transforms, models
import json
from utils import print_results, CNNArch


class Predict:
    def __init__(self, image_path, load_dir, top_k, category_names, gpu):
        self.image_path = image_path
        self.load_dir = load_dir
        self.top_k = top_k
        self.category_names = category_names
        self.gpu = gpu
        self.prediction_transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

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

    def load_category_names(self):
        with open(self.category_names, "r") as f:
            cat_to_name = json.load(f)
        return cat_to_name

    def load_checkpoint(self):
        checkpoint = torch.load(self.load_dir)
        model = models.get_model(checkpoint["arch"])
        if checkpoint["arch"] == CNNArch.VGG16:
            model.classifier = checkpoint["classifier"]
        else:
            model.fc = checkpoint["classifier"]
        model.load_state_dict(checkpoint["state_dict"])
        model.class_to_idx = checkpoint["class_to_idx"]
        self.model = model
        self.input_size = checkpoint["input_size"]

    def predict(self):
        """Predict the class (or classes) of an image using a trained deep learning model."""
        self.set_device()
        self.load_checkpoint()
        cat_to_name = self.load_category_names()
        self.model.to(self.device)
        self.model.eval()
        img_transformer = transforms.Compose(self.prediction_transforms)
        image = Image.open(self.image_path)
        image = img_transformer(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        output = self.model.forward(image)
        probabilities = torch.exp(output)
        top_probabilities, top_labels = probabilities.topk(self.top_k)
        top_probabilities = top_probabilities.detach().cpu().tolist()
        top_probabilities = top_probabilities[0]
        top_labels = top_labels.cpu().tolist()
        top_labels = top_labels[0]
        idx_to_class = {value: key for key, value in self.model.class_to_idx.items()}
        top_classes = [idx_to_class[label] for label in top_labels]
        print_results(cat_to_name, top_probabilities, top_classes)
        return top_probabilities, top_classes


def main():
    parser = argparse.ArgumentParser(
        description="Predict the class of an image using a trained deep learning model"
    )
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("load_dir", type=str, help="Path to the checkpoint file")
    parser.add_argument(
        "--top_k", type=int, default=5, help="Return the top K most likely classes"
    )
    parser.add_argument(
        "--category_names",
        type=str,
        default="cat_to_name.json",
        help="Path to the file that maps the class values to category names",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    args = parser.parse_args()
    predict = Predict(
        args.image_path, args.load_dir, args.top_k, args.category_names, args.gpu
    )
    predict.predict()


if __name__ == "__main__":
    main()
