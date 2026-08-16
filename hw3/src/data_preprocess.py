import os
import cv2
import random
import pandas as pd
import numpy as np
from tqdm import tqdm


random.seed(42)

DATA_ROOT = "../train"
CSV_PATH = "balanced_data.csv"

label_mapping = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}

data_records = []

class_counts = {}
print("Loading raw data...")
for emotion in os.listdir(DATA_ROOT):
    emotion_dir = os.path.join(DATA_ROOT, emotion)
    if not os.path.isdir(emotion_dir):
        continue
    label = label_mapping[emotion]
    image_files = os.listdir(emotion_dir)
    class_counts[label] = len(image_files)
    for file in image_files:
        data_records.append([os.path.join(emotion_dir, file), label, "original"])

max_count = max(class_counts.values())
print("Original class counts:", class_counts)

print("Augmenting minority classes...")
for label, count in class_counts.items():
    if count >= max_count:
        continue

    needed = max_count - count
    pool = [rec for rec in data_records if rec[1] == label]

    # 擴增的圖像記錄
    augment_records = []
    aug_types = ['hflip', 'vflip', 'rotate']
    k = 0

    for i in range(needed):
        path, _, _ = random.choice(pool)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if aug_types[k] == 'hflip':
            img_aug = cv2.flip(img, 1)
        elif aug_types[k] == 'vflip':
            img_aug = cv2.flip(img, 0)
        else:
            angle = random.uniform(-30, 30)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
            img_aug = cv2.warpAffine(img, M, (w, h))

        k = (k + 1) % len(aug_types)

        # 存到 memory 並記錄
        tmp_path = f"augmented/{label}_{i}.png"
        os.makedirs("augmented", exist_ok=True)
        cv2.imwrite(tmp_path, img_aug)
        augment_records.append([tmp_path, label, "augmented"])

    data_records.extend(augment_records)

print(f"Saving to {CSV_PATH} with {len(data_records)} records")
df = pd.DataFrame(data_records, columns=["path", "label", "source"])
df.to_csv(CSV_PATH, index=False)
print("Done")
