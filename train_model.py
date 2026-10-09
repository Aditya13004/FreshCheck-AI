import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================
IMG_SIZE = 224
BATCH_SIZE = 32
DATASET_PATH = "dataset"  # You should extract your downloaded dataset here

# Determine classes based on your requirement (Apple, Banana, Orange, etc.)
# Using MobileNetV2 for faster training and web app compatibility
print("TF version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# ==============================================================================
# 2. DATA PREPARATION & SPLITTING (FIXED)
# ==============================================================================
# We parse the dataset folder. We expect folders like 'freshapples', 'rottenbanana', etc.
# Or if it's already split into 'train' and 'test', we look inside those.

def gather_image_paths(base_dir):
    all_paths = []
    fruit_labels = []
    cond_labels = []
    
    # Dynamically find fruit names by looking at folder names (removing 'fresh' or 'rotten')
    fruit_set = set()
    for root, dirs, files in os.walk(base_dir):
        folder = os.path.basename(root).lower()
        if folder.startswith('fresh'):
            fruit_set.add(folder.replace('fresh', ''))
        elif folder.startswith('rotten'):
            fruit_set.add(folder.replace('rotten', ''))
            
    fruit_to_idx = {fruit: idx for idx, fruit in enumerate(sorted(list(fruit_set)))}
    cond_to_idx = {"fresh": 0, "rotten": 1}
    
    idx_to_fruit = {v: k for k, v in fruit_to_idx.items()}
    idx_to_cond = {v: k for k, v in cond_to_idx.items()}

    for root, dirs, files in os.walk(base_dir):
        folder_name = os.path.basename(root).lower()
        if folder_name in ['train', 'test', 'dataset']:
            continue
            
        # Extract fruit and condition from folder name
        cond_lbl, fruit_lbl = None, None
        
        for cond, c_idx in cond_to_idx.items():
            if cond in folder_name:
                cond_lbl = c_idx
                break
                
        for fruit, f_idx in fruit_to_idx.items():
            if fruit in folder_name:
                fruit_lbl = f_idx
                break
                
        if cond_lbl is not None and fruit_lbl is not None:
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    full_path = os.path.join(root, file)
                    try:
                        # 1. Strict Magic Byte Check (TensorFlow only likes real JPG/PNG/BMP)
                        with open(full_path, "rb") as f:
                            header = f.read(8)
                        is_jpeg = header.startswith(b'\xff\xd8')
                        is_png = header.startswith(b'\x89PNG\x0d\x0a\x1a\x0a')
                        is_bmp = header.startswith(b'BM')
                        
                        if not (is_jpeg or is_png or is_bmp):
                            continue # Skip WebPs masquerading as JPGs
                            
                        # 2. Deep verification to catch truncated files
                        from PIL import Image
                        with Image.open(full_path) as img:
                            img.load() 
                            
                        all_paths.append(full_path)
                        fruit_labels.append(fruit_lbl)
                        cond_labels.append(cond_lbl)
                    except Exception:
                        pass # Silently skip corrupted images
                    
    return all_paths, fruit_labels, cond_labels, fruit_to_idx, idx_to_fruit, idx_to_cond

all_paths, fruit_labels, cond_labels, fruit_to_idx, idx_to_fruit, idx_to_cond = gather_image_paths(DATASET_PATH)
num_fruits = len(fruit_to_idx)

print(f"Total valid images found: {len(all_paths)}")

# BUG FIX: Correct Data-Splitting Strategy
# 1. Split into (Train + Val) and (Test) - Test is strictly 15% and completely unseen!
tr_va_p, test_p, tr_va_f, test_f, tr_va_c, test_c = train_test_split(
    all_paths, fruit_labels, cond_labels,
    test_size=0.15, stratify=fruit_labels, random_state=42
)

# 2. Split (Train + Val) into Train (80% of remainder) and Val (20% of remainder)
tr_p, val_p, tr_f, val_f, tr_c, val_c = train_test_split(
    tr_va_p, tr_va_f, tr_va_c,
    test_size=0.20, stratify=tr_va_f, random_state=42
)

print(f"Train set: {len(tr_p)} images")
print(f"Validation set (for tuning): {len(val_p)} images")
print(f"Test set (STRICTLY for final evaluation): {len(test_p)} images")

# ==============================================================================
# 3. TF.DATA PIPELINE (MATCHES INFERENCE)
# ==============================================================================
def safe_preprocess(path, fruit_lbl, cond_lbl):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    
    # Using MobileNetV2 preprocessing (scales pixels between -1 and 1)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, {"fruit_output": fruit_lbl, "cond_output": tf.cast(cond_lbl, tf.float32)}

def make_ds(paths, flabels, clabels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, flabels, clabels))
    ds = ds.map(safe_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2048)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(tr_p, tr_f, tr_c, training=True)
val_ds   = make_ds(val_p, val_f, val_c)
test_ds  = make_ds(test_p, test_f, test_c)

# ==============================================================================
# 4. MODEL ARCHITECTURE (MULTI-HEAD LEARNING)
# ==============================================================================
def build_model(num_fruits):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomBrightness(0.2)(x)

    # Replaced EfficientNetB3 with MobileNetV2 (better for beginner/web app requirements)
    backbone = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    backbone.trainable = False

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Head 1 — Fruit Type
    fx = layers.Dense(128, activation="relu")(x)
    fx = layers.Dropout(0.2)(fx)
    fruit_out = layers.Dense(num_fruits, activation="softmax", name="fruit_output")(fx)

    # Head 2 — Condition (Fresh vs Rotten)
    cx = layers.Dense(128, activation="relu")(x)
    cx = layers.Dropout(0.2)(cx)
    cond_out = layers.Dense(1, activation="sigmoid", name="cond_output")(cx)

    return keras.Model(inputs, [fruit_out, cond_out])

model = build_model(num_fruits)

# ==============================================================================
# 5. TRAINING PHASE
# ==============================================================================
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=3, verbose=1),
    keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True, verbose=1)
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss={
        "fruit_output": "sparse_categorical_crossentropy",
        "cond_output":  "binary_crossentropy"
    },
    loss_weights={"fruit_output": 1.0, "cond_output": 0.5},
    metrics={"fruit_output": "accuracy", "cond_output": "accuracy"}
)

print("\n=== Phase 1: Training heads only ===")
# NOTE: Kept epochs low for initial run. You can increase this to 20 for full training.
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

# Optional Phase 2: Unfreeze top layers for fine-tuning... (omitted for brevity, can be enabled later)

# ==============================================================================
# 6. FINAL INDEPENDENT EVALUATION ON TEST SET (NO LEAKAGE)
# ==============================================================================
print("\n=== FINAL EVALUATION ON UNSEEN TEST SET ===")
y_true_f, y_true_c, y_pred_f, y_pred_c = [], [], [], []

for imgs, labels in test_ds:
    fp, cp = model.predict(imgs, verbose=0)
    y_true_f.extend(labels["fruit_output"].numpy())
    y_true_c.extend(labels["cond_output"].numpy())
    y_pred_f.extend(np.argmax(fp, axis=1))
    y_pred_c.extend((cp[:, 0] > 0.5).astype(int))

fruit_names = [idx_to_fruit[i] for i in range(num_fruits)]

print("\n--- Fruit Classification Metrics ---")
print(classification_report(y_true_f, y_pred_f, target_names=fruit_names))

print("\n--- Condition (Fresh vs Rotten) Metrics ---")
print(classification_report(y_true_c, y_pred_c, target_names=["fresh", "rotten"]))

# Save Model
model.save("fruit_classifier_final.keras")

import json
with open("class_names.json", "w") as f:
    json.dump(idx_to_fruit, f)

print("\n✅ Training complete! Model saved as fruit_classifier_final.keras and classes saved to class_names.json")
