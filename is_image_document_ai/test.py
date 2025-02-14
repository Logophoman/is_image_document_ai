# test.py

import torch
import os
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

def load_model(model_path="mobilenet_document_vs_image.pth", num_classes=2):
    """
    Loads the MobileNetV2 model with a final layer for `num_classes` (e.g., 2).
    
    :param model_path: Path to the saved model state dict (e.g. "mobilenet_document_vs_image.pth")
    :param num_classes: Number of classes in the final classification layer
    :return: A MobileNetV2 model with the custom final layer and loaded weights, set to eval mode.
    """
    # 1. Initialize MobileNetV2 pretrained on ImageNet
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    # 2. Replace final layer for 2-class classification
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    # 3. Load onto CPU (or GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 4. Load the saved state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    # 5. Set model to evaluation mode
    model.eval()
    return model

def predict_image(model, image_path):
    """
    Predict if `image_path` is a 'document' or an 'image' using the given model.
    
    :param model: A trained MobileNetV2 with 2-class head.
    :param image_path: Path to a single image file.
    :return: A string, either "document" or "image".
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Same normalization & resize as in training, but without random flips
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Open the image and convert to RGB just in case
    img = Image.open(image_path).convert('RGB')
    # Apply the same transforms => shape [1, 3, 224, 224]
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        # Get the predicted class index
        _, pred = torch.max(output, dim=1)
    
    # If your dataset subfolders were named alphabetically: 'document' -> class 0, 'image' -> class 1
    if pred.item() == 0:
        return "document"
    else:
        return "image"
    
def test_model_with_dataloader(model, root_dir="images", batch_size=32):
    """
    Evaluate the model on the 'test' portion of the dataset loaded via DataLoader.
    This requires a 3-way random split from data_loader.get_data_loaders(...).
    
    :param model: Trained MobileNet model with 2-class head.
    :param root_dir: Root folder containing subfolders 'document/' and 'image/'.
    :param batch_size: Batch size for loading the data.
    """
    # We import here to avoid circular imports if data_loader also imports test.py
    from data_loader import get_data_loaders
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a new train/val/test split - but we'll only use the test_loader
    train_loader, val_loader, test_loader = get_data_loaders(
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

def test_model_on_folder(model, folder_path="test_images"):
    """
    Runs prediction on all images in `folder_path`.
    Prints the file name and the predicted label for each image.
    
    :param model: The trained classification model.
    :param folder_path: Directory containing images to classify.
    """
    # List all .png/.jpg/.jpeg files
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for f in files:
        image_path = os.path.join(folder_path, f)
        label = predict_image(model, image_path)
        print(f"{f} => {label}")

if __name__ == '__main__':
    # Load the trained model
    model = load_model("mobilenet_document_vs_image.pth")
    
    # Classify all images in the "test_images" folder 
    # (create this folder and add some images before running)
    
    # test_model_on_folder(model, "test_images")

    test_model_with_dataloader(model, root_dir="images", batch_size=32)
