import torch
import torch.nn as nn
from torchvision import models
from transformers import BertLMHeadModel, BertConfig

class XRayReportGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- 1. THE ENCODER (Vision) ---
        # We load a pre-trained DenseNet-121
        print("Loading DenseNet-121 Encoder...")
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        
        # We drop the final classification layer because we aren't classifying cats/dogs.
        # We just want the raw visual feature maps from the lungs.
        self.feature_extractor = densenet.features
        
        # --- 2. THE BRIDGE (Projection) ---
        # DenseNet outputs features with a size of 1024.
        # But our BERT Transformer expects inputs with a size of 768.
        # This linear layer translates the vision features into the transformer's language.
        self.proj = nn.Linear(1024, 768)
        
        # --- 3. THE DECODER (Text) ---
        # We use BERT, but we configure it to act as a text generator (decoder) 
        # and tell it to pay attention to the images (cross-attention).
        print("Loading Transformer Decoder...")
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.is_decoder = True
        config.add_cross_attention = True
        
        self.decoder = BertLMHeadModel.from_pretrained("bert-base-uncased", config=config)

    def forward(self, images, input_ids, attention_mask):
        # Step 1: Pass the X-ray through DenseNet
        # Output shape: [Batch, 1024, 7, 7]
        features = self.feature_extractor(images)
        
        # Step 2: Flatten the 7x7 grid into a sequence of 49 "Visual Tokens"
        # New shape: [Batch, 49, 1024]
        batch_size, channels, h, w = features.shape
        features = features.view(batch_size, channels, h * w).permute(0, 2, 1)
        
        # Step 3: Pass visual tokens through the bridge
        # New shape: [Batch, 49, 768]
        visual_embeds = self.proj(features)
        
        # Step 4: The Transformer looks at the visual embeds and generates the report.
        # By passing labels=input_ids, HuggingFace automatically calculates the loss for us!
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=visual_embeds,
            labels=input_ids 
        )
        
        # Return the loss (for training) and logits (the actual predicted words)
        return outputs.loss, outputs.logits

# --- QUICK TEST ---
if __name__ == "__main__":
    from dataset import XRayDataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    
    print("\nInitializing Model Test...")
    model = XRayReportGenerator()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Load just ONE batch from your local dataset to test the pipes
    dataset = XRayDataset('data/train_metadata.csv', 'images', tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    
    # Feed the batch into our new model
    print("\nFeeding data through the model (this might take a few seconds on CPU)...")
    loss, logits = model(
        images=batch['pixel_values'],
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask']
    )
    
    print("\n✅ Model Forward Pass Successful!")
    print(f"Calculated Loss: {loss.item():.4f}")
    print(f"Logits Shape (Predictions): {logits.shape}")