import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

def visualize_clones():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'KMC_original_csv.csv')
    images_dir = os.path.join(base_dir, 'images')

    df = pd.read_csv(csv_path)
    
    # Strip hidden spaces/newlines
    df['IMAGE_NAMES'] = df['IMAGE_NAMES'].str.strip()
    df['IMAGE_LABELS'] = df['IMAGE_LABELS'].str.strip()

    print("Scanning for a random patient with a complete set of augmentations...")
    sample_label = None
    image_filenames = []
    
    # Get all patients who have at least 3 images listed in the CSV
    label_counts = df['IMAGE_LABELS'].value_counts()
    potential_labels = label_counts[label_counts >= 3].index.tolist()
    
    # FIX: Shuffle the list so we get a different patient every time we run it!
    random.shuffle(potential_labels)
    
    for label in potential_labels:
        names = df[df['IMAGE_LABELS'] == label]['IMAGE_NAMES'].tolist()
        existing = [name for name in names if os.path.exists(os.path.join(images_dir, name))]
        
        if len(existing) >= 3: 
            sample_label = label
            image_filenames = existing
            break

    if not sample_label:
        print("❌ Could not find any patient with existing duplicate images.")
        return

    print(f"✅ Success! Visualizing patient: {sample_label} ({len(image_filenames)} valid files found)")
    print("\n--- EXACT FILENAMES ---")
    for name in image_filenames:
        print(name)
    print("-----------------------\n")

    # Limit to 5 images for the display
    num_to_show = min(5, len(image_filenames))
    fig, axes = plt.subplots(1, num_to_show, figsize=(15, 5))
    
    if num_to_show == 1:
        axes = [axes] 

    for i in range(num_to_show):
        img_name = image_filenames[i]
        img_path = os.path.join(images_dir, img_name)
        
        img = Image.open(img_path)
        axes[i].imshow(img, cmap='gray')
        
        display_title = "Base Image" if img_name == sample_label else f"Augmentation {i}"
        axes[i].set_title(display_title)
        axes[i].axis('off')

    plt.suptitle(f"Visualizing Data Leakage / Augmentations for: {sample_label}", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_clones()