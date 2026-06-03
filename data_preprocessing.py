import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# 1. Load the original CSV
df = pd.read_csv('data/KMC_original_csv.csv')
img_dir = 'images'

# 2. PRUNING: Only keep rows where the image actually exists on your disk
print(f"Original CSV size: {len(df)}")
df['exists'] = df['IMAGE_NAMES'].apply(lambda x: os.path.exists(os.path.join(img_dir, str(x).strip())))
df = df[df['exists'] == True].copy()
df = df.drop(columns=['exists'])
print(f"Usable rows (matching your 221 images): {len(df)}")

# 3. SPLITTING: Now split ONLY the 221 images
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, val_idx = next(gss.split(df, groups=df['IMAGE_LABELS']))

train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]

# 4. SAVE: Overwrite the old CSVs with the clean, smaller versions
train_df.to_csv('data/train_metadata.csv', index=False)
val_df.to_csv('data/val_metadata.csv', index=False)

print(f"New Train size: {len(train_df)}")
print(f"New Val size: {len(val_df)}")