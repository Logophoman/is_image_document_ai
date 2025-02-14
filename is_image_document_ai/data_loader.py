# data_loader.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(
    root_dir="images",
    input_size=(224, 224),
    batch_size=32,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    num_workers=2
):
    """
    Creates PyTorch DataLoaders for training, validation, and test from a single
    folder structure. Splits the data into (train_ratio, val_ratio, test_ratio).
    
    Folder structure:
        images/
          ├─ document/
          │   ├─ 0.jpg
          │   ├─ 1.jpg
          │   ...
          └─ image/
              ├─ 0.jpg
              ├─ 1.jpg
              ...
    
    Args:
        root_dir (str): Path to the root dataset folder with 2 subfolders: document, image.
        input_size (tuple): (width, height) for resizing images.
        batch_size (int): Batch size for all loaders.
        train_ratio (float): Fraction of total data used for training.
        val_ratio (float): Fraction of total data used for validation.
        test_ratio (float): Fraction of total data used for testing.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Make sure the ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-6, "train_ratio + val_ratio + test_ratio must be 1."

    # 1) Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # typical ImageNet normalization
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 2) Create a single dataset using train_transform by default
    full_dataset = datasets.ImageFolder(root=root_dir, transform=train_transform)
    total_len = len(full_dataset)
    
    # 3) Calculate split lengths
    train_len = int(total_len * train_ratio)
    val_len   = int(total_len * val_ratio)
    test_len  = total_len - train_len - val_len  # leftover

    # 4) Randomly split into three subsets (they reference the same underlying Dataset)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])

    # 5) We want val_dataset and test_dataset to use the "eval_transform" 
    #    instead of the "train_transform" (which was set on full_dataset).
    #    Because random_split subsets share the same Dataset object,
    #    we can override .transform on the *underlying* dataset for
    #    the validation & test portions.
    #    NOTE: random_split returns Subset objects that store indices 
    #    plus a reference to the "full_dataset".
    #    We can do:
    val_dataset.dataset.transform = eval_transform
    test_dataset.dataset.transform = eval_transform
    
    # 6) Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
