import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer

class XRayDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, max_length=128):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # REQUIREMENT 1: Format the image exactly how a pre-trained DenseNet expects
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # These are the exact mathematical means/standard deviations from ImageNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- 1. PROCESS THE IMAGE ---
        img_name = str(row['IMAGE_NAMES']).strip()
        img_path = os.path.join(self.img_dir, img_name)
        
        # Convert to RGB because DenseNet expects 3 color channels, even for grayscale
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # --- 2. PROCESS THE TEXT ---
        findings = str(row['FINDINGS']).strip()
        impression = str(row['IMPRESSIONS']).strip()  # <-- Added the S
        
        # Safety check: If the CSV has empty cells, replace them so the code doesn't break
        if pd.isna(row['FINDINGS']) or findings.lower() == 'nan': 
            findings = "No findings."
        if pd.isna(row['IMPRESSIONS']) or impression.lower() == 'nan': # <-- Added the S
            impression = "No impression."            
        # REQUIREMENT 2: Combine into a single target report
        full_report = f"FINDINGS: {findings} IMPRESSION: {impression}"
        
        # REQUIREMENT 3: Tokenize the text (turn words into numbers for the Transformer)
        encodings = self.tokenizer(
            full_report,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Return a clean dictionary to the training loop
        return {
            'pixel_values': image,
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze()
        }

# --- QUICK TEST TO PROVE IT WORKS ---
if __name__ == "__main__":
    print("Initializing Tokenizer and Dataset...")
    
    # We use a standard BERT tokenizer as our baseline text translator
    test_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Let's test it on your local Training split!
    train_dataset = XRayDataset(
        csv_file='data/train_metadata.csv', 
        img_dir='images', 
        tokenizer=test_tokenizer
    )
    
    # Load 4 images at a time (Batch Size = 4)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Grab the very first batch
    batch = next(iter(train_loader))
    
    print("\n✅ Batch Successfully Loaded!")
    print(f"Images Tensor Shape: {batch['pixel_values'].shape}")
    print(f"Text Tokens Shape: {batch['input_ids'].shape}")