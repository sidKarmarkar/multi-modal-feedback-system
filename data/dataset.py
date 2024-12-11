from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO
import os

def download_images():
    dataset = load_dataset("yuvalkirstain/pickapic_v1", split='train')

    image_dir = '/Users/sidkarmarkar/Desktop/Columbia/Academics/EECS6694/Dynamic User-Guided Refinement for Text-to-Image Generation/downloaded_images'
    os.makedirs(image_dir, exist_ok=True)

    for i, example in enumerate(dataset):
        if i >= 20:  # Stop after downloading 20 images
            break
        
        image_url = example['image_0_url']
        response = requests.get(image_url)
        
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            
            # Save the image to your local machine
            image_path = os.path.join(image_dir, f'image_{i+1}.jpg')
            image.save(image_path)
            print(f"Downloaded {image_path}")
        else:
            print(f"Failed to download image from {image_url}")

if __name__ == "__main__":
    download_images()