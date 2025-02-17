# is_image_document_ai

A lightweight pipeline for classifying whether an image is a **document** (e.g., scanned text, forms, PDFs) or a **generic image** (photographs, art, etc.). This repository provides **two** model options:

1. **MobileNetV2** (pretrained on ImageNet, then fine-tuned)
2. **TinyCNN** (a small custom CNN)

Both achieve ~99% accuracy on our dataset, with MobileNetV2 (recommended & default) slightly higher (~99.8%) and TinyCNN slightly faster (~99.2% accuracy).

## Quickstart:

Install with: 

`pip install is-image-document-ai==0.0.2`

Classify a folder of images of documents or generic images...

```python
from is_image_document_ai import infer_folder

result = infer_folder("testfolder")
print(result)
```

Expected result: 

```
paper.jpg => document
image.png => image
['document', 'image']
```

## Table of Contents

- [Overview](#overview)
- [Data](#data)
- [Installation](#installation)
- [Inference](#inference)
  - [Single Image](#single-image)
  - [Folder of Images](#folder-of-images)
  - [Test-Set Evaluation (DataLoader)](#test-set-evaluation-dataloader)
- [Training](#training)
  - [MobileNetV2 Model](#mobilenetv2-model)
  - [TinyCNN Model](#tinycnn-model)
- [Usage Examples](#usage-examples)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

- **Purpose**: Quickly determine if an image is a “document” vs. a “regular image.” Useful for filtering scanned PDFs, forms, or text-based images from photo-based datasets.
- **Models**:
  - **MobileNetV2** for high accuracy (99.8%).
  - **TinyCNN** for a more lightweight approach (~99.2% accuracy, fewer parameters).

---

## Data

- We downloaded **10,000 document images** from [HuggingFaceM4/DocumentVQA](https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA) and **10,000 generic images** from [jackyhate/text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M), resulting in:
  ```
  images/
   ├─ document/
   │   ├─ 0.jpg
   │   ├─ 1.jpg
   │   ...
   └─ image/
       ├─ 0.jpg
       ├─ 1.jpg
       ...
  ```
- We perform a **3-way random split**: 80% training, 10% validation, 10% test (i.e., `train_ratio=0.8, val_ratio=0.1, test_ratio=0.1`).
- Each image is resized to **224×224** before feeding into the models.

## Installation

## Using the PyPi package:

`pip install is-image-document-ai==0.0.1`

## Using normal python environment:

1. **Clone** this repository:
   ```bash
   git clone https://github.com/logophoman/is_image_document_ai.git
   cd is_image_document_ai
   ```
2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
   This installs `torch`, `torchvision`, `requests`, and `datasets`.

3. (Optional) **Download data** if you haven’t:
   ```bash
   python download_data.py
   ```
   By default, it streams from the Hugging Face datasets to get 10k `document` images and 10k `image` images, saved into `images/`.

   > Note: You will need ~3.8GB of free storage for the image data. Training requires ~4GB of VRAM for mobilenet and ~800MB VRAM for TinyCNN

---

## Inference

## Using the PyPi package: 

Input Folder:

```python
from is_image_document_ai import load_tinycnn, infer_folder

model = load_tinycnn()
result = infer_folder("testfolder", model)
print(result)
```

Expected result: 

```
paper.jpg => document
image.png => image
['document', 'image']
```

Input File:

```python
from is_image_document_ai import load_mobilenet, infer_single_image

model = load_mobilenet()
result = infer_single_image("paper.jpg", model)
print(result)
```

Expected result: 

```
paper.jpg => document
document
```

Or for the lazy ones (defaults to mobilenet):

```python
from is_image_document_ai import infer_folder

result = infer_folder("testfolder")
print(result)
```

## Manual using the scripts yourself:

We provide multiple ways to run inference. You can:

1. Use the **`test.py`** (for MobileNetV2) or **`tinytest.py`** (for TinyCNN) scripts for quick checks.
2. Use the **`inference.py`** script, which supports command-line arguments to load **either** model and do **single, folder, or testset** evaluation.

### Single Image

```bash
python inference.py --model-type mobilenet \
                    --weights mobilenet_document_vs_image.pth \
                    --mode single \
                    --input path/to/your_image.jpg
```

### Folder of Images

```bash
python inference.py --model-type tinycnn \
                    --weights tinycnn_document_vs_image.pth \
                    --mode folder \
                    --input test_images/
```

### Test-Set Evaluation (DataLoader)

```bash
python inference.py --model-type mobilenet \
                    --weights mobilenet_document_vs_image.pth \
                    --mode testset \
                    --root-dir images \
                    --batch-size 32
```

---

## Training

### MobileNetV2 Model

- **File**: [`train.py`](train.py)
- Command:
  ```bash
  python train.py
  ```
- This script:
  1. Loads data from `images/` folder using [`data_loader.py`](data_loader.py).
  2. Initializes a MobileNetV2 (pretrained on ImageNet).
  3. Fine-tunes the final layer for 2 classes: **document** vs. **image**.
  4. Trains until ~99% accuracy or until epochs finish.
  5. Saves the weights to **`mobilenet_document_vs_image.pth`**.
  6. Evaluates on the test set (the 10% split).

### TinyCNN Model

- **File**: [`tinytrain.py`](tinytrain.py)
- Command:
  ```bash
  python tinytrain.py
  ```
- This script:
  1. Loads the same `images/` dataset with an 80/10/10 split.
  2. Initializes our custom **TinyCNN** architecture ([`model.py`](model.py)).
  3. Trains for up to 5 epochs (or early stops at ~99% val accuracy).
  4. Saves weights to **`tinycnn_document_vs_image.pth`**.
  5. Prints final test accuracy.

---

## Results

After training on 20k images (10k document + 10k generic), both models achieve **high accuracy**:

- **MobileNetV2**: ~**99.8%** accuracy on the test set.
- **TinyCNN**: ~**99.2%** accuracy on the test set.

The TinyCNN has fewer parameters and uses less VRAM for the model itself, but PyTorch’s baseline overhead may still be a few hundred MB. MobileNetV2 is slightly more accurate and can converge faster due to ImageNet pretraining.

---

## Contributing

Contributions are welcome! To get started:

1. **Fork** the repo and clone your fork.
2. **Create a new branch** for your feature or bug fix.
3. **Make changes**, add tests, and ensure everything passes.
4. **Open a Pull Request** describing your changes.

Possible improvements:

- **Quantization** or **Half-Precision** (FP16) to reduce model size / VRAM usage.
- Additional **data augmentations** to handle more varied document layouts.
- Implementation of other **lightweight architectures** (e.g. ShuffleNet, EfficientNet-Lite).

---

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this code as permitted by the license.

---

**Enjoy classifying your images!** If you have any questions or issues, feel free to open an issue or pull request.

