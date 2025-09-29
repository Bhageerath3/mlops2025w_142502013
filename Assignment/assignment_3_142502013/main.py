import torch
import urllib.request
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import json
import toml
from itertools import product
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import (
    resnet34, resnet50, resnet101, resnet152,
    ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
)


def main():
    print()
    print("\nFor the question 3a, pre-trained resnet models are used to perform inference on a sample image.")
    print("The sample image has been downloaded from the PyTorch hub.")
    print("The top-5 predictions along with their probabilities are displayed for each ResNet variant.")
    print("ResNet variants used: resnet34, resnet50, resnet101, resnet152")
    print()
    
    resnet_variants = {
        "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
        "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1),
        "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V1),
        "resnet152": (resnet152, ResNet152_Weights.IMAGENET1K_V1),
    }

    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  

    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    urllib.request.urlretrieve(labels_url, "imagenet_classes.txt")

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    for model_name, (model_fn, weights) in resnet_variants.items():
        print(f"\nInference using {model_name.upper()} :- ")
        model = model_fn(weights=weights)
        model.eval()

        if torch.cuda.is_available():
            model.to("cuda")
            batch = input_batch.to("cuda")
        else:
            batch = input_batch

        with torch.no_grad():
            output = model(batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for i in range(top5_prob.size(0)):
            print(f"{categories[top5_catid[i]]}: {top5_prob[i].item()*100:.2f}%")
    print("\n\n")
    print("For 3b and 3c please refer config.json and params.toml files.\n")
    print("For 3e, hyperparameter tuning is performed on a small subset of the MNIST dataset using different combinations of ResNet variants, learning rates, optimizers, and momentums.")

    print("Since the dataset is huge, the training is done for only one batch to keep it quick.")

    print("The best hyperparameters(for 1 batch training) for each ResNet variant based on validation accuracy are displayed at the end.")
    print()

    with open("config.json") as f:
        config = json.load(f)

    params = toml.load("params.toml")

    model_map = {
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152
    }

    # --
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    val_full   = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    
    train_dataset = Subset(train_full, range(500))
    val_dataset   = Subset(val_full, range(100))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def train_evaluate(model_name, lr, optimizer_type, momentum):
        model = model_map[model_name](weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        if optimizer_type.lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type.lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError("Unknown optimizer")

        
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            break 

       
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"{model_name} | lr={lr}, optimizer={optimizer_type}, momentum={momentum} -> val_acc: {acc:.2f}%")
        return acc

    results = []

    for model_name in config["models"]:
        lrs = params[model_name]["learning_rates"]
        opts = params[model_name]["optimizers"]
        moms = params[model_name]["momentums"]
        
        for lr, opt, mom in product(lrs, opts, moms):
            acc = train_evaluate(model_name, lr, opt, mom)
            results.append({
                "model": model_name,
                "learning_rate": lr,
                "optimizer": opt,
                "momentum": mom,
                "val_accuracy": acc
            })

    for model_name in config["models"]:
        best = max([r for r in results if r["model"]==model_name], key=lambda x: x["val_accuracy"])
        print(f"\nBest hyperparameters(1 batch training) and accuracy for {model_name}: {best}")

if __name__ == "__main__":
    main()
