# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from .data_loader import get_data_loaders

def train_model(
    root_dir="images",
    num_classes=2,
    num_epochs=5,
    batch_size=32,
    learning_rate=1e-3
):
    # 1. Get data loaders (3-way split)
    #    Suppose you do 80% train, 10% val, 10% test
    train_loader, val_loader, test_loader = get_data_loaders(
        root_dir=root_dir,
        batch_size=batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # 2. Load a pretrained MobileNet
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    
    # 3. Replace the classifier layer for 2 classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    # 4. Move to CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 5. Define loss/optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 6. Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Optional early stopping
        if val_acc >= 99.0:
            print("Early stopping - high validation accuracy reached.")
            break

    # 7. Save model
    torch.save(model.state_dict(), "mobilenet_document_vs_image.pth")
    print("Model saved to mobilenet_document_vs_image.pth")

    # (Optional) Evaluate on test set
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (preds == labels).sum().item()
    test_acc = 100.0 * test_correct / test_total
    print(f"Final Test Accuracy: {test_acc:.2f}%")

if __name__ == '__main__':
    train_model()
