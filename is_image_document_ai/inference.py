#!/usr/bin/env python3
"""
Example inference script that can:
1) Load TinyCNN or MobileNetV2
2) Do single-image or folder-based inference
3) Evaluate on a random-split test subset from data_loader.py
"""

import argparse
import os
import torch
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
from torchvision import models, transforms

# Import your custom TinyCNN
from .model import TinyCNN  # Make sure model.py is in the same directory or installed as a module

def load_tinycnn(model_path=None, num_classes=2, device="cpu"):
    """
    Load a TinyCNN model and weights from the installed package directory.
    """
    if model_path is None:
        # Get the directory of the installed package
        package_dir = Path(__file__).resolve().parent
        model_path = package_dir / "tinycnn_document_vs_image.pth"

    model = TinyCNN(num_classes=num_classes)
    model.to(device)
    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_mobilenet(model_path=None, num_classes=2, device="cpu"):
    """
    Load a MobileNetV2 model and weights from the installed package directory.
    """
    if model_path is None:
        package_dir = Path(__file__).resolve().parent
        model_path = package_dir / "mobilenet_document_vs_image.pth"

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.to(device)
    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_image(image_path, model = None, device="cpu"):
    """
    Predicts if `image_path` is 'document' or 'image' using the given model.
    """

    if model is None:
        model = load_mobilenet()

    # Same transforms as training (excluding random flips)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, dim=1)

    # If subfolders were named alphabetically => 'document'=0, 'image'=1
    return "document" if pred.item() == 0 else "image"

def infer_single_image(image_path, model = None,  device="cpu"):
    """
    Classify a single image file.
    """
    if model is None:
        model = load_mobilenet()
    
    if not os.path.isfile(image_path):
        print(f"Error: '{image_path}' is not a file.")
        return
    label = predict_image(image_path, model=model, device=device)
    print(f"{image_path} => {label}")
    return label

def infer_folder(folder_path, model = None, device="cpu"):
    """
    Classify all images in a folder.
    """

    if model is None:
        model = load_mobilenet()
    
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory.")
        return

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"No images found in '{folder_path}'.")
        return

    labels = [] 

    for f in files:
        image_path = os.path.join(folder_path, f)
        label = predict_image(image_path, model=model, device=device)
        print(f"{f} => {label}")
        labels.append(label)
    return labels

def test_with_dataloader(model, root_dir="images", batch_size=32, device="cpu"):
    """
    Evaluate the model on the test portion of the dataset from data_loader.py.
    """
    from data_loader import get_data_loaders
    # Create new splits (train/val/test). We'll only use test_loader.
    _, _, test_loader = get_data_loaders(
        root_dir=root_dir,
        batch_size=batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )

    total = 0
    correct = 0
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test accuracy (DataLoader-based) on random-split subset: {accuracy:.2f}%")

def main():
    parser = argparse.ArgumentParser(
        description="Inference script for document vs. image classification."
    )
    parser.add_argument("--model-type", type=str, choices=["tinycnn", "mobilenet"], default="tinycnn",
                        help="Which model to load: 'tinycnn' or 'mobilenet'. Default: tinycnn.")
    parser.add_argument("--weights", type=str, default="tinycnn_document_vs_image.pth",
                        help="Path to the saved model weights.")
    parser.add_argument("--mode", type=str, choices=["single", "folder", "testset"], default="single",
                        help="Inference mode: single image, entire folder, or testset.")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to image file or folder (for single/folder modes). Ignored in 'testset' mode.")
    parser.add_argument("--root-dir", type=str, default="images",
                        help="Root directory for the dataset. Used in 'testset' mode.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for dataloader test. Default: 32.")
    args = parser.parse_args()

    # Decide device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the chosen model
    if args.model_type == "tinycnn":
        print(f"Loading TinyCNN from {args.weights}...")
        model = load_tinycnn(args.weights, num_classes=2, device=device)
    else:
        print(f"Loading MobileNetV2 from {args.weights}...")
        model = load_mobilenet(args.weights, num_classes=2, device=device)

    # Inference based on mode
    if args.mode == "single":
        if args.input is None:
            print("Error: --input must be specified for single-image mode.")
        else:
            infer_single_image(model, args.input, device=device)

    elif args.mode == "folder":
        if args.input is None:
            print("Error: --input must be specified for folder mode.")
        else:
            infer_folder(model, args.input, device=device)

    elif args.mode == "testset":
        test_with_dataloader(model, root_dir=args.root_dir, batch_size=args.batch_size, device=device)

if __name__ == "__main__":
    main()
