import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os
import matplotlib.pyplot as plt
from PIL import Image

def load_and_clean_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Isolate the base image name (Grouping logic)
    # The 'IMAGE_LABELS' column usually holds the base name like 'CXR01.JPEG'
    # The 'IMAGE_NAMES' column holds the long UUID variations
    # We will group by the base 'IMAGE_LABELS' to prevent leakage
    print(f"Total rows found: {len(df)}")
    unique_base_images = df['IMAGE_LABELS'].nunique()
    print(f"Total UNIQUE base images found: {unique_base_images}")
    
    return df

def create_splits(df):
    # 2. Use GroupShuffleSplit to keep identical base images in the same split
    # We use 80% for training and 20% for validation
    print("\nCreating Train/Validation splits based on Image Groups...")
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    
    train_idx, val_idx = next(gss.split(df, groups=df['IMAGE_LABELS']))
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    print(f"Training set size: {len(train_df)} rows")
    print(f"Validation set size: {len(val_df)} rows")
    
    # Check for leakage
    train_bases = set(train_df['IMAGE_LABELS'])
    val_bases = set(val_df['IMAGE_LABELS'])
    leakage = train_bases.intersection(val_bases)
    
    if len(leakage) == 0:
        print("✅ Data Split SUCCESS: No data leakage detected between Train and Val sets.")
    else:
        print(f"❌ WARNING: Leakage detected in {len(leakage)} base images.")
        
    return train_df, val_df

def verify_image_paths(df, images_dir):
    # 3. Since you have 16GB RAM, we can safely check if the files actually exist!
    print(f"\nVerifying image paths in '{images_dir}'...")
    missing_files = 0
    
    for img_name in df['IMAGE_NAMES'].head(100): # Only check first 100 for speed
        path = os.path.join(images_dir, img_name)
        if not os.path.exists(path):
            missing_files += 1
            
    if missing_files == 0:
         print("✅ Path check passed for sample batch.")
    else:
         print(f"⚠️ Missing files detected. Check your folder structure.")

def save_splits(train_df, val_df, data_dir):
    # 4. Export the clean data
    train_path = os.path.join(data_dir, 'train_metadata.csv')
    val_path = os.path.join(data_dir, 'val_metadata.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print(f"\nSaved clean splits to '{data_dir}' folder.")

if __name__ == "__main__":
    # Define paths based on your provided structure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    images_dir = os.path.join(base_dir, 'images')
    csv_path = os.path.join(data_dir, 'KMC_original_csv.csv')
    
    # Execute Pipeline
    df = load_and_clean_data(csv_path)
    train_df, val_df = create_splits(df)
    verify_image_paths(df, images_dir)
    save_splits(train_df, val_df, data_dir)
    
    print("\nPhase 2 Complete. Ready for Phase 3 (README generation).")