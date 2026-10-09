import os
import shutil

# Paths
NEW_DATASET_PATH = "new_dataset"
TARGET_DATASET_PATH = "dataset/train"

print("Starting to merge datasets...")
os.makedirs(TARGET_DATASET_PATH, exist_ok=True)

if not os.path.exists(NEW_DATASET_PATH):
    print(f"ERROR: Could not find '{NEW_DATASET_PATH}'.")
    print("Please make sure you extracted the new Kaggle download into a folder named 'new_dataset'.")
    exit()

count = 0
for folder_name in os.listdir(NEW_DATASET_PATH):
    source_dir = os.path.join(NEW_DATASET_PATH, folder_name)
    if not os.path.isdir(source_dir):
        continue
    
    # The new dataset uses names like "Mango__Healthy" and "Mango__Rotten"
    if "__" in folder_name:
        parts = folder_name.split("__")
        fruit_name = parts[0].lower()
        condition = parts[1].lower()
        
        # Map to our existing naming convention
        if condition == "healthy":
            new_cond = "fresh"
        elif condition == "rotten":
            new_cond = "rotten"
        else:
            continue
            
        target_folder = f"{new_cond}{fruit_name}"
        target_dir = os.path.join(TARGET_DATASET_PATH, target_folder)
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy the images over
        print(f"Merging {folder_name} -> {target_folder}...")
        for file in os.listdir(source_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                shutil.copy2(os.path.join(source_dir, file), os.path.join(target_dir, file))
                count += 1

print(f"\n✅ Merging complete! Successfully added {count} new images to your dataset.")
print("You can now safely delete the 'new_dataset' folder to save space.")
