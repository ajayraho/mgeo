raise Exception('This code is intended to run on colab, please dont run here')

import boto3
import pandas as pd
import gzip
import json
import os
from botocore import UNSIGNED
from botocore.config import Config

# 1. Setup AWS S3 Client (No Sign-in Required)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
BUCKET_NAME = 'amazon-berkeley-objects'

def download_file(s3_key, local_path):
    """Downloads a file from the public S3 bucket if not exists."""
    if not os.path.exists(local_path):
        print(f"Downloading {s3_key}...")
        s3.download_file(BUCKET_NAME, s3_key, local_path)
        print(f"Saved to {local_path}")
    else:
        print(f"Found {local_path}, skipping download.")

def load_abo_dataset(download_dir="abo_data"):
    """Loads and Merges ABO Metadata + Image Paths."""
    os.makedirs(download_dir, exist_ok=True)

    # --- A. Download Metadata (The Text) ---
    # The listings are inside a tar, but we can grab the specific json.gz if we know the key
    # For simplicity, we grab the specific metadata file typically found in the archive
    # Note: AWS structure sometimes changes, but standard path is usually accessible.
    # If direct key fails, we download the mappings first.

    # Let's grab the Image Metadata (CSV) - It's smaller and maps IDs to Paths
    img_meta_key = "images/metadata/images.csv.gz"
    img_meta_path = os.path.join(download_dir, "images.csv.gz")
    download_file(img_meta_key, img_meta_path)

    # Let's grab the Listings Metadata (JSON)
    # This is often large, so for a quick test, we might stream or download the
    # 'listings/metadata/listings_0.json.gz' which acts as a partition.
    listing_key = "listings/metadata/listings_0.json.gz"
    listing_path = os.path.join(download_dir, "listings_0.json.gz")
    download_file(listing_key, listing_path)

    # --- B. Load into Pandas ---
    print("Loading Image Metadata...")
    df_imgs = pd.read_csv(img_meta_path)

    print("Loading Listings Metadata (this may take a moment)...")
    data = []
    with gzip.open(listing_path, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    df_text = pd.DataFrame(data)

    # --- C. Merge Text + Visuals ---
    # We map 'main_image_id' from text to 'image_id' from images
    print("Merging Dataframes...")
    df_merged = df_text.merge(
        df_imgs,
        left_on='main_image_id',
        right_on='image_id',
        how='inner'
    )

    # Filter for columns we actually care about for Attribute Injection
    cols_to_keep = [
        'item_id',
        'item_name',  # The Title
        'bullet_point', # The Features (List)
        'path',       # The S3 Path to the Image
        'brand',      # Good for filtering
        'product_type'
    ]

    # Keep only columns that exist (some might be missing in specific versions)
    final_cols = [c for c in cols_to_keep if c in df_merged.columns]

    return df_merged

def download_specific_image(s3_image_path, local_save_path):
    """Downloads the actual JPG image for the VLM to see."""
    # The dataset path is usually 'images/small/...' or 'images/original/...'
    # The 'path' column in metadata usually looks like '71/7182838.jpg'
    # We need to prepend 'images/small/' to it.

    full_key = f"images/small/{s3_image_path}"
    s3.download_file(BUCKET_NAME, full_key, local_save_path)

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Load the Data
    df = load_abo_dataset()

    print(f"Successfully loaded {len(df)} products with text+images.")

    # 2. Filter for a visual category (e.g., 'SHIRT') to test your pipeline
    # Note: 'product_type' values vary, we search specifically for visual items.
    # Let's just sample 5 random items that have bullet points.
    sample_df = df[df['bullet_point'].notna()].sample(5)

    # 3. Download the Sample Images
    os.makedirs("abo_samples", exist_ok=True)

    print("\n--- Downloading Samples for AutoGEO Phase 1 ---")
    for idx, row in sample_df.iterrows():
        img_name = f"{row['item_id']}.jpg"
        local_path = os.path.join("abo_samples", img_name)

        print(f"Product: {row['item_name'][:30]}...")
        print(f" -> Downloading Image: {row['path']}")

        try:
            download_specific_image(row['path'], local_path)
        except Exception as e:
            print(f" -> Failed to download: {e}")

    print("\nDone! Check the 'abo_samples' folder.")
    print("You can now feed these images + 'row.bullet_point' into your LLaVA model.")