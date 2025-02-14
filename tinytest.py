# tinytest.py

import torch
import os
from PIL import Image
import torch.nn as nn
from torchvision import transforms

from model import TinyCNN  # your custom CNN

def load_model(model_path="tinycnn_document_vs_image.pth", num_classes=2):
    """
    Loads TinyCNN model with a final layer for `num_classes` (e.g., 2).
    """
    model = TinyCNN(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load saved state
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.eval()
    return model

def predict_image(model, image_path):
    """
    Predict if `image_path` is a 'document' or 'image' using the given model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Same transforms as training except we omit random flips
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
    
    # If subfolders are alphabetical => 'document' = 0, 'image' = 1
    return "document" if pred.item() == 0 else "image"

def test_model_on_folder(model, folder_path="test_images"):
    """
    Runs prediction on all images in `folder_path` and prints results.
    """
    if not os.path.isdir(folder_path):
        print(f"Folder '{folder_path}' does not exist or is not a directory.")
        return
    
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"No image files found in '{folder_path}'.")
        return
    
    for f in files:
        image_path = os.path.join(folder_path, f)
        label = predict_image(model, image_path)
        print(f"{f} => {label}")

def test_model_with_dataloader(model, root_dir="images", batch_size=32):
    """
    Evaluate the model on the 'test' portion of the dataset loaded via DataLoader.
    This requires a 3-way random split from data_loader.get_data_loaders(...).
    """
    # Import inside the function to avoid circular imports
    from data_loader import get_data_loaders
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a new train/val/test split - we'll only use test_loader
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
            _, preds = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    print(f"Test accuracy (DataLoader-based) on random-split subset: {accuracy:.2f}%")

if __name__ == '__main__':
    # Example usage
    model = load_model("tinycnn_document_vs_image.pth", num_classes=2)
    
    # 1) Ad-hoc classification of images in "test_images/"
    # test_model_on_folder(model, "test_images")
    
    # 2) Evaluate on the official test split from data_loader.py
    test_model_with_dataloader(model, root_dir="images", batch_size=32)
