import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import BertTokenizer
from datasets import load_dataset
import requests
from io import BytesIO

class RichHF18KDataset(Dataset):
    def __init__(self, tfrecord_path, pickapic_dataset, transform=None):
        self.tfrecord_path = tfrecord_path
        self.dataset = tf.data.TFRecordDataset(tfrecord_path)
        self.pickapic_dataset = pickapic_dataset  # This is a streaming dataset
        self.transform = transform or Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return sum(1 for _ in self.dataset)

    def __getitem__(self, idx):
        raw_record = next(iter(self.dataset.skip(idx).take(1)))
        parsed = tf.train.Example.FromString(raw_record.numpy())
        
        filename = parsed.features.feature['filename'].bytes_list.value[0].decode('utf-8')

        image = self.fetch_image_from_stream(filename)
        
        prompt = parsed.features.feature['prompt_misalignment_label'].bytes_list.value[0].decode('utf-8')
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
        
        scores = {
            'aesthetics_score': torch.tensor(parsed.features.feature['aesthetics_score'].float_list.value[0]),
            'artifact_score': torch.tensor(parsed.features.feature['artifact_score'].float_list.value[0]),
            'misalignment_score': torch.tensor(parsed.features.feature['misalignment_score'].float_list.value[0]),
            'overall_score': torch.tensor(parsed.features.feature['overall_score'].float_list.value[0]),
        }
        
        return image, encoded_prompt['input_ids'].squeeze(0), encoded_prompt['attention_mask'].squeeze(0), scores

    def fetch_image_from_stream(self, filename):
        for example in self.pickapic_dataset['train']:
            if example['image_0_uid'] == filename or example['image_1_uid'] == filename:
                image_url = example['image_0_url'] if example['image_0_uid'] == filename else example['image_1_url']
                response = requests.get(image_url)
                if response.status_code == 200:
                    return self.transform(Image.open(BytesIO(response.content)).convert('RGB'))
                else:
                    raise Exception(f"Failed to download image from URL: {image_url}")
        raise Exception(f"No matching entry found for filename: {filename}")

def get_data_loader(tfrecord_path, batch_size=16, shuffle=True):
    pickapic_dataset = load_dataset("yuvalkirstain/pickapic_v1", split='train', streaming=True)
    dataset = RichHF18KDataset(tfrecord_path, pickapic_dataset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Example Usage:
if __name__ == "__main__":
    tfrecord_path = '/path/to/your/data/train.tfrecord'
    loader = get_data_loader(tfrecord_path, batch_size=8)

    for images, input_ids, attention_masks, scores in loader:
        print(images.shape, input_ids.shape, attention_masks.shape, scores)