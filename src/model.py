import torch
from torch import nn
from transformers import ViTModel, BertModel, BertTokenizer, T5ForConditionalGeneration, T5Tokenizer

class RichFeedbackSystem(nn.Module):
    def __init__(self):
        super(RichFeedbackSystem, self).__init__()
        # Load pre-trained Vision Transformer (ViT)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Load pre-trained BERT model for text embeddings
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Load pre-trained T5 model for generating text sequences
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

        self.self_attention = nn.MultiheadAttention(embed_dim=768, num_heads=12)
        
        self.plausibility_head = nn.Linear(768, 1)
        self.alignment_head = nn.Linear(768, 1)
        self.aesthetic_head = nn.Linear(768, 1)
        self.overall_quality_head = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, text_prompts):
        vit_outputs = self.vit(pixel_values=images)
        image_features = vit_outputs.last_hidden_state

        encoded_text = self.t5_tokenizer(text_prompts, return_tensors='pt', padding=True, truncation=True).to(images.device)
        bert_outputs = self.bert(input_ids=encoded_text.input_ids, attention_mask=encoded_text.attention_mask)
        text_features = bert_outputs.last_hidden_state

        combined_features = torch.cat([image_features, text_features], dim=1)
        attn_output, _ = self.self_attention(combined_features, combined_features, combined_features)
        mean_attn_output = attn_output.mean(dim=1)
        plausibility = self.sigmoid(self.plausibility_head(mean_attn_output))
        alignment = self.sigmoid(self.alignment_head(mean_attn_output))
        aesthetics = self.sigmoid(self.aesthetic_head(mean_attn_output))
        overall_quality = self.sigmoid(self.overall_quality_head(mean_attn_output))

        t5_outputs = self.t5.generate(input_ids=encoded_text.input_ids, attention_mask=encoded_text.attention_mask, max_length=50)

        return {
            'plausibility': plausibility,
            'alignment': alignment,
            'aesthetics': aesthetics,
            'overall_quality': overall_quality
        }, t5_outputs

# Example Usage
if __name__ == "__main__":
    model = RichFeedbackSystem()
    dummy_images = torch.rand(2, 3, 224, 224)  # Simulated batch of 2 images, 224x224 with 3 color channels
    dummy_texts = ["a cat on a mat", "a dog in the fog"]
    scores, modified_texts = model(dummy_images, dummy_texts)
    print("Scores:", scores)