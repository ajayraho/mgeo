raise Exception('This module is intended to be run on colab env. Please dont proceed ahead.')

#@title download image based on `queries.json`

import boto3
import json
import os
import shutil
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm  # Progress bar for sanity

# --- CONFIGURATION ---
INPUT_JSON = "query.json"  # The file generated in the previous step
DOWNLOAD_DIR = "abo_images_download"  # Local folder to store images
OUTPUT_ZIP = "abo_images_dataset"     # Name of the final zip file
BUCKET_NAME = 'amazon-berkeley-objects'

# 1. Setup AWS S3 Client (Public Access)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

def download_image(s3_path, local_save_path):
    """
    Downloads a single image from ABO S3 Bucket.
    ABO structure: 'images/small/' + the path found in the CSV/JSON.
    """
    try:
        # Construct the full S3 key (using 'small' for speed/efficiency)
        full_key = f"images/small/{s3_path}"

        s3.download_file(BUCKET_NAME, full_key, local_save_path)
        return True
    except Exception as e:
        # Returns False if 404 or other error
        return False

def fetch_all_images():
    # 1. Load the Repository
    if not os.path.exists(INPUT_JSON):
        print(f"❌ Error: {INPUT_JSON} not found. Run generate_repository.py first.")
        return

    with open(INPUT_JSON, 'r') as f:
        repo_data = json.load(f)

    print(f"📂 Loaded Repository. Scanning for unique images...")

    # 2. Extract Unique Image Paths
    # We use a dictionary to deduplicate (Item A might appear in 2 different queries)
    # Mapping: item_id -> s3_image_path
    image_targets = {}

    for query_obj in repo_data:
        for result in query_obj['results']:
            item_id = result['item_id']
            # Access 'image_path' key we created in generate_repository.py
            img_path = result.get('image_path')

            if img_path:
                image_targets[item_id] = img_path

    print(f"   Found {len(image_targets)} unique images to download.")

    # 3. Prepare Directory
    if os.path.exists(DOWNLOAD_DIR):
        shutil.rmtree(DOWNLOAD_DIR) # Clean start
    os.makedirs(DOWNLOAD_DIR)

    # 4. Download Loop (with Progress Bar)
    print(f"🚀 Starting Download to '{DOWNLOAD_DIR}/'...")

    success_count = 0
    fail_count = 0

    # Using tqdm for a nice progress bar in Colab
    for item_id, s3_path in tqdm(image_targets.items(), desc="Downloading"):

        # Save as "item_id.jpg" so Agents can find it easily using just the ID
        filename = f"{item_id}.jpg"
        local_path = os.path.join(DOWNLOAD_DIR, filename)

        if download_image(s3_path, local_path):
            success_count += 1
        else:
            fail_count += 1
            print(f"\n⚠️ Failed: {item_id} ({s3_path})")

    print(f"\n✅ Download Complete.")
    print(f"   Success: {success_count}")
    print(f"   Failed:  {fail_count}")

    # 5. Zip it up
    print(f"📦 Zipping dataset...")
    shutil.make_archive(OUTPUT_ZIP, 'zip', DOWNLOAD_DIR)
    print(f"🎉 Created {OUTPUT_ZIP}.zip")
    print(f"   Size: {os.path.getsize(OUTPUT_ZIP + '.zip') / (1024*1024):.2f} MB")

if __name__ == "__main__":
    fetch_all_images()