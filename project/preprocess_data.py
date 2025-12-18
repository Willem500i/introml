"""
Preprocess extracted features and prepare image crops for CNN training.

This script takes the extracted vehicle-curb features and creates:
1. Cropped vehicle images saved to disk (memory-efficient)
2. Normalized feature vectors for each sample
3. Train/val/test splits ready for model training

Memory-efficient: Processes images in batches and saves incrementally.
Uses dataset limits to keep training tractable.
"""

import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import gc


# Dataset size limits (adjust based on available memory/time)
MAX_TRAIN_SAMPLES = 50000  # 50K training samples is plenty for this task
MAX_VAL_SAMPLES = 10000
MAX_TEST_SAMPLES = 10000


def load_features(data_dir: str, split: str) -> List[Dict]:
    """Load extracted features for a dataset split."""
    filepath = os.path.join(data_dir, 'processed', f'{split}_features.pkl')
    
    if not os.path.exists(filepath):
        print(f"Features file not found: {filepath}")
        print("Run extract_features.py first!")
        return []
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_image_path(data_dir: str, split: str, image_name: str) -> str:
    """Get full path to an image file."""
    return os.path.join(data_dir, '100k', split, f'{image_name}.jpg')


def crop_vehicle(image_path: str, box2d: Dict, padding: float = 0.1) -> Optional[Image.Image]:
    """Crop vehicle from image with optional padding."""
    if not os.path.exists(image_path):
        return None
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            x1, y1 = box2d['x1'], box2d['y1']
            x2, y2 = box2d['x2'], box2d['y2']
            
            box_width = x2 - x1
            box_height = y2 - y1
            pad_x = box_width * padding
            pad_y = box_height * padding
            
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)
            
            crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
            return crop.copy()
        
    except Exception:
        return None


def resize_and_normalize(img: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize image and convert to normalized numpy array."""
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    
    if len(arr.shape) == 2:
        arr = np.stack([arr] * 3, axis=-1)
    
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    
    return arr


def create_feature_vectors(features: List[Dict]) -> np.ndarray:
    """Create normalized feature vectors for non-image-based classification."""
    scene_encoding = {'city street': 0, 'highway': 1, 'residential': 2, 
                      'parking lot': 3, 'tunnel': 4, 'undefined': 5, 'unknown': 5}
    weather_encoding = {'clear': 0, 'overcast': 1, 'rainy': 2, 
                        'snowy': 3, 'foggy': 4, 'partly cloudy': 5, 'undefined': 6, 'unknown': 6}
    time_encoding = {'daytime': 0, 'night': 1, 'dawn/dusk': 2, 'undefined': 3, 'unknown': 3}
    
    vectors = []
    for f in features:
        vec = [
            f['normalized_curb_distance'],
            f['vehicle_width'] / 200,
            f['vehicle_height'] / 200,
            f['vehicle_area'] / 40000,
            f['vehicle_x_center'] / 1280,
            f['vehicle_y_position'] / 720,
            1.0 if f['occluded'] else 0.0,
            scene_encoding.get(f['scene'], 5) / 5,
            weather_encoding.get(f['weather'], 6) / 6,
            time_encoding.get(f['timeofday'], 3) / 3
        ]
        vectors.append(vec)
    
    return np.array(vectors, dtype=np.float32)


def balance_classes(features: List[Dict], max_samples: int) -> List[Dict]:
    """Balance classes and limit total samples."""
    # Separate by class
    class_0 = [f for f in features if f['label'] == 0]
    class_1 = [f for f in features if f['label'] == 1]
    
    # Shuffle
    np.random.seed(42)
    np.random.shuffle(class_0)
    np.random.shuffle(class_1)
    
    # Balance: take equal from each class up to half of max_samples
    samples_per_class = max_samples // 2
    class_0 = class_0[:samples_per_class]
    class_1 = class_1[:samples_per_class]
    
    # Combine and shuffle
    balanced = class_0 + class_1
    np.random.shuffle(balanced)
    
    return balanced


def prepare_dataset(data_dir: str, split: str, output_dir: str,
                    target_size: Tuple[int, int] = (224, 224),
                    max_samples: int = 50000,
                    batch_size: int = 2000):
    """
    Prepare dataset with memory-efficient batch processing.
    """
    features = load_features(data_dir, split)
    
    if not features:
        return 0
    
    print(f"Loaded {len(features)} features, limiting to {max_samples} balanced samples...")
    features = balance_classes(features, max_samples)
    print(f"After balancing: {len(features)} samples")
    
    total = len(features)
    num_batches = (total + batch_size - 1) // batch_size
    
    all_images = []
    all_labels = []
    all_valid_features = []
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = features[start:end]
        
        batch_images = []
        
        for feat in tqdm(batch, desc=f"Batch {batch_idx+1}/{num_batches}"):
            image_path = get_image_path(data_dir, split, feat['image_name'])
            crop = crop_vehicle(image_path, feat['box2d'])
            
            if crop is None:
                continue
            
            if crop.size[0] < 32 or crop.size[1] < 32:
                crop.close()
                continue
            
            img_array = resize_and_normalize(crop, target_size)
            crop.close()
            
            batch_images.append(img_array)
            all_labels.append(feat['label'])
            all_valid_features.append(feat)
        
        all_images.extend(batch_images)
        
        # Periodic cleanup
        gc.collect()
    
    if not all_images:
        return 0
    
    # Stack and save
    print("Stacking arrays...")
    X = np.stack(all_images)
    y = np.array(all_labels)
    
    del all_images
    gc.collect()
    
    feature_vectors = create_feature_vectors(all_valid_features)
    
    # Save
    print("Saving to disk...")
    np.save(os.path.join(output_dir, f'{split}_images.npy'), X)
    np.save(os.path.join(output_dir, f'{split}_labels.npy'), y)
    np.save(os.path.join(output_dir, f'{split}_feature_vectors.npy'), feature_vectors)
    
    with open(os.path.join(output_dir, f'{split}_metadata.pkl'), 'wb') as f:
        pickle.dump(all_valid_features, f)
    
    print(f"\nSaved {len(X)} samples")
    print(f"  Images shape: {X.shape}")
    print(f"  Labels: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
    
    # Final estimate: 224*224*3*4 bytes per image = 600KB
    size_gb = len(X) * 224 * 224 * 3 * 4 / (1024**3)
    print(f"  Images file size: ~{size_gb:.2f} GB")
    
    return len(X)


def main():
    """Main function to preprocess data."""
    data_dir = 'data'
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    target_size = (224, 224)
    
    splits_config = {
        'train': MAX_TRAIN_SAMPLES,
        'val': MAX_VAL_SAMPLES,
        'test': MAX_TEST_SAMPLES
    }
    
    for split, max_samples in splits_config.items():
        print(f"\n{'='*50}")
        print(f"Processing {split} split (max {max_samples} samples)...")
        print('='*50)
        
        count = prepare_dataset(
            data_dir, split, output_dir,
            target_size=target_size,
            max_samples=max_samples
        )
        
        if count == 0:
            print(f"No valid samples for {split}")
        
        gc.collect()
    
    print("\n" + "="*50)
    print("Preprocessing complete!")
    print("="*50)


if __name__ == '__main__':
    main()
