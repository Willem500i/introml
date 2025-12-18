"""
Extract vehicle-curb spatial relationships from BDD100K annotations.

This script processes BDD100K dataset annotations to create a binary classification
dataset for detecting vehicles near curbs (potential parking) vs vehicles in traffic lanes.
"""

import json
import os
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import pickle
from tqdm import tqdm


def point_to_line_distance(point: Tuple[float, float], 
                           line_start: Tuple[float, float], 
                           line_end: Tuple[float, float]) -> float:
    """Calculate perpendicular distance from a point to a line segment."""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Line segment length squared
    line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    
    if line_len_sq == 0:
        # Line segment is a point
        return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    
    # Project point onto line, clamped to segment
    t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / line_len_sq))
    
    # Nearest point on segment
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    
    return np.sqrt((x0 - proj_x) ** 2 + (y0 - proj_y) ** 2)


def box_center(box: Dict) -> Tuple[float, float]:
    """Get center point of a bounding box."""
    return ((box['x1'] + box['x2']) / 2, (box['y1'] + box['y2']) / 2)


def box_bottom_center(box: Dict) -> Tuple[float, float]:
    """Get bottom center point of a bounding box (wheel position approximation)."""
    return ((box['x1'] + box['x2']) / 2, box['y2'])


def get_curb_lines(objects: List[Dict]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Extract curb line segments from annotation objects."""
    curb_lines = []
    
    for obj in objects:
        if obj.get('category') == 'lane/road curb' and 'poly2d' in obj:
            points = obj['poly2d']
            # Convert poly2d to line segments
            for i in range(len(points) - 1):
                p1 = (points[i][0], points[i][1])
                p2 = (points[i + 1][0], points[i + 1][1])
                curb_lines.append((p1, p2))
    
    return curb_lines


def get_vehicles(objects: List[Dict]) -> List[Dict]:
    """Extract vehicle bounding boxes from annotation objects."""
    vehicle_categories = {'car', 'truck', 'bus'}
    vehicles = []
    
    for obj in objects:
        if obj.get('category') in vehicle_categories and 'box2d' in obj:
            vehicles.append({
                'category': obj['category'],
                'box2d': obj['box2d'],
                'occluded': obj.get('attributes', {}).get('occluded', False),
                'truncated': obj.get('attributes', {}).get('truncated', False),
                'id': obj.get('id')
            })
    
    return vehicles


def min_distance_to_curbs(vehicle: Dict, curb_lines: List) -> float:
    """Calculate minimum distance from vehicle to any curb line."""
    if not curb_lines:
        return float('inf')
    
    # Use bottom center of vehicle (approximate wheel position)
    vehicle_point = box_bottom_center(vehicle['box2d'])
    
    min_dist = float('inf')
    for line_start, line_end in curb_lines:
        dist = point_to_line_distance(vehicle_point, line_start, line_end)
        min_dist = min(min_dist, dist)
    
    return min_dist


def extract_features_from_annotation(annotation: Dict) -> List[Dict]:
    """
    Extract vehicle-curb features from a single annotation file.
    
    Returns list of feature dictionaries, one per vehicle.
    """
    features = []
    
    for frame in annotation.get('frames', []):
        objects = frame.get('objects', [])
        
        curb_lines = get_curb_lines(objects)
        vehicles = get_vehicles(objects)
        
        # Scene attributes
        scene_attrs = annotation.get('attributes', {})
        
        for vehicle in vehicles:
            # Skip heavily occluded or truncated vehicles
            if vehicle.get('truncated', False):
                continue
            
            # Calculate distance to nearest curb
            curb_distance = min_distance_to_curbs(vehicle, curb_lines)
            
            # Vehicle size (proxy for distance from camera)
            box = vehicle['box2d']
            vehicle_width = box['x2'] - box['x1']
            vehicle_height = box['y2'] - box['y1']
            vehicle_area = vehicle_width * vehicle_height
            
            # Vertical position (lower = closer to camera, more reliable)
            vehicle_y_pos = box['y2']  # Bottom of vehicle
            
            # Normalize curb distance by vehicle size (accounts for perspective)
            normalized_curb_dist = curb_distance / max(vehicle_width, 1)
            
            feature = {
                'image_name': annotation.get('name', ''),
                'vehicle_id': vehicle.get('id'),
                'vehicle_type': vehicle['category'],
                'curb_distance': curb_distance,
                'normalized_curb_distance': normalized_curb_dist,
                'vehicle_width': vehicle_width,
                'vehicle_height': vehicle_height,
                'vehicle_area': vehicle_area,
                'vehicle_y_position': vehicle_y_pos,
                'vehicle_x_center': (box['x1'] + box['x2']) / 2,
                'occluded': vehicle.get('occluded', False),
                'has_curb_visible': len(curb_lines) > 0,
                'scene': scene_attrs.get('scene', 'unknown'),
                'weather': scene_attrs.get('weather', 'unknown'),
                'timeofday': scene_attrs.get('timeofday', 'unknown'),
                'box2d': vehicle['box2d']
            }
            
            features.append(feature)
    
    return features


def create_labels(features: List[Dict], 
                  near_curb_threshold: float = 2.0,
                  min_vehicle_area: float = 1000) -> List[Dict]:
    """
    Create binary labels for vehicles based on curb proximity.
    
    Args:
        features: List of feature dictionaries
        near_curb_threshold: Normalized distance threshold for "near curb"
        min_vehicle_area: Minimum vehicle area to include (filters tiny/distant vehicles)
    
    Returns:
        List of labeled feature dictionaries
    """
    labeled = []
    
    for f in features:
        # Filter out vehicles that are too small or don't have curb info
        if f['vehicle_area'] < min_vehicle_area:
            continue
        if not f['has_curb_visible']:
            continue
        
        # Create label: 1 = near curb, 0 = in lane
        label = 1 if f['normalized_curb_distance'] < near_curb_threshold else 0
        
        f['label'] = label
        f['label_name'] = 'near_curb' if label == 1 else 'in_lane'
        labeled.append(f)
    
    return labeled


def process_dataset(data_dir: str, split: str, max_files: Optional[int] = None) -> List[Dict]:
    """Process all annotation files in a dataset split."""
    anno_dir = os.path.join(data_dir, '100k-2', split)
    
    if not os.path.exists(anno_dir):
        print(f"Directory not found: {anno_dir}")
        return []
    
    files = [f for f in os.listdir(anno_dir) if f.endswith('.json')]
    if max_files:
        files = files[:max_files]
    
    all_features = []
    
    for filename in tqdm(files, desc=f"Processing {split}"):
        filepath = os.path.join(anno_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                annotation = json.load(f)
            
            features = extract_features_from_annotation(annotation)
            all_features.extend(features)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return all_features


def main():
    """Main function to extract features and create dataset."""
    data_dir = 'data'
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\n{'='*50}")
        print(f"Processing {split} split...")
        print('='*50)
        
        # Extract features
        features = process_dataset(data_dir, split)
        print(f"Extracted {len(features)} vehicle instances")
        
        # Create labels
        labeled_features = create_labels(features)
        print(f"Labeled {len(labeled_features)} valid instances")
        
        # Count labels
        label_counts = defaultdict(int)
        for f in labeled_features:
            label_counts[f['label_name']] += 1
        print(f"Label distribution: {dict(label_counts)}")
        
        # Save features
        output_file = os.path.join(output_dir, f'{split}_features.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(labeled_features, f)
        print(f"Saved to {output_file}")
        
        # Also save as JSON for inspection
        json_file = os.path.join(output_dir, f'{split}_features_sample.json')
        with open(json_file, 'w') as f:
            # Save first 100 samples for inspection
            json.dump(labeled_features[:100], f, indent=2)
    
    print("\n" + "="*50)
    print("Feature extraction complete!")
    print("="*50)


if __name__ == '__main__':
    main()

