import requests
import os
from datasets import load_dataset

def download_images_text2image(save_dir="images/image", limit=10_000):
    """
    Stream the 'jackyhate/text-to-image-2M' dataset, which has a 'jpg' column containing
    a PIL image. We'll save only the first `limit` images to `save_dir`.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Downloading up to {limit} samples from jackyhate/text-to-image-2M (streaming)...")

    ds = load_dataset("jackyhate/text-to-image-2M", split="train", streaming=True)
    # Optional: shuffle, so we randomly sample from the dataset
    ds = ds.shuffle(seed=42, buffer_size=10_000)

    count = 0
    for sample in ds:
        if count >= limit:
            break
        try:
            # 'jpg' is a PIL image
            pil_image = sample["jpg"]
            outpath = os.path.join(save_dir, f"{count}.jpg")
            pil_image.save(outpath, format="JPEG")
            count += 1
        except Exception as e:
            print(f"Failed to save sample {count}: {e}")
    
    print(f"Saved {count} images into '{save_dir}'.")


def download_images_documentvqa(save_dir="images/document", limit=10_000):
    """
    Stream the 'HuggingFaceM4/DocumentVQA' dataset and save
    only the first `limit` images into `save_dir`.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    print(f"Downloading up to {limit} 'document' samples from HuggingFaceM4/DocumentVQA... (streaming)")

    ds = load_dataset("HuggingFaceM4/DocumentVQA", split="train", streaming=True)
    # optional: shuffle
    ds = ds.shuffle(seed=42, buffer_size=10_000)

    count = 0
    for sample in ds:
        if count >= limit:
            break

        try:
            # "image" column is presumably a PIL image
            pil_image = sample["image"]
            outpath = os.path.join(save_dir, f"{count}.jpg")
            pil_image.save(outpath, format="JPEG")
            count += 1
        except Exception as e:
            print(f"Failed to save sample {count}: {e}")

    print(f"Saved {count} images into '{save_dir}'.")


def main():
    # Ensure these folders exist
    os.makedirs("images", exist_ok=True)
    os.makedirs("images/image", exist_ok=True)
    os.makedirs("images/document", exist_ok=True)

    # Download a subset of text-to-image-2M (generic images)
    #download_images_text2image(save_dir="images/image", limit=10_000)
    
    # Download a subset of DocumentVQA (document-like images)
    download_images_documentvqa(save_dir="images/document", limit=10_000)


if __name__ == "__main__":
    main()
