
# Inspect the dataset (count classes):
#     python3 inspect_dataset.py
#     
# Visualize a specific image with its bounding boxes:
#     python3 inspect_dataset.py --image path/to/your/image.jpg
#    
# Specify a different dataset directory:
#     python3 inspect_dataset.py --dataset /path/to/your/dataset

import yaml
import os
import argparse
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def inspect_yolo_dataset(dataset_root):
    """
    Inspects the YOLO dataset and displays the count of objects for each class.
    Also collects and outputs 10 image paths for each class (7 with bboxes, 3 without).
    """
    dataset_root = Path(dataset_root)
    yaml_path = dataset_root / 'data.yaml'

    if not yaml_path.exists():
        print(f"Error: {yaml_path} not found.")
        return []

    # Load class names from yaml
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading {yaml_path}: {e}")
        return []
    class_names = data.get('names', [])

    if not class_names:
        print("Error: No class names found in data.yaml")
        return []

    # The directories we found in the file system
    possible_splits = {
        'train': 'train',
        'valid': 'valid',
        'test': 'test'
    }
    
    results = {split: Counter() for split in possible_splits}
    
    # New structures for image collection
    # class_to_images: class_id -> list of image paths containing that class
    class_to_images = {i: [] for i in range(len(class_names))}

    # no_bbox_images: list of image paths with no bounding boxes
    no_bbox_images = []

    for split_key, dir_name in possible_splits.items():
        images_dir = dataset_root / dir_name / 'images'
        labels_dir = dataset_root / dir_name / 'labels'
        
        if not images_dir.exists():
            continue
        print(f"Processing {split_key} split ({dir_name})...")
        
        # Find all images
        all_images_set = set()
        all_images_set.update(images_dir.glob('*.jpg'))
        
        all_images = sorted(list(all_images_set))
        
        for img_path in all_images:
            label_path = labels_dir / f"{img_path.stem}.txt" if labels_dir.exists() else None
            has_bboxes = False
            
            if label_path and label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            has_bboxes = True
                            for line in lines:
                                parts = line.split()
                                if len(parts) > 0:
                                    class_id = int(parts[0])
                                    if class_id < len(class_names):
                                        # Update counts
                                        results[split_key][class_id] += 1
                                        # Update class to images mapping
                                        if img_path not in class_to_images[class_id]:
                                            class_to_images[class_id].append(img_path)
                except Exception as e:
                    print(f"Warning: Could not read {label_path}: {e}")
            
            if not has_bboxes:
                no_bbox_images.append(img_path)

    # Display results (Table)
    print("\n" + "="*75)
    print(f"{'Class Name':<25} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    print("-" * 75)
    total_counts = Counter()
    for split in results:
        total_counts.update(results[split])
    for i in range(len(class_names)):
        name = class_names[i]
        train_cnt = results['train'].get(i, 0)
        val_cnt = results['valid'].get(i, 0)
        test_cnt = results['test'].get(i, 0)
        total = total_counts.get(i, 0)
        print(f"{name:<25} | {train_cnt:<8} | {val_cnt:<8} | {test_cnt:<8} | {total:<8}")
    print("-" * 75)
    
    train_total = sum(results['train'].values())
    val_total = sum(results['valid'].values())
    test_total = sum(results['test'].values())
    grand_total = sum(total_counts.values())

    print(f"{'TOTAL':<25} | {train_total:<8} | {val_total:<8} | {test_total:<8} | {grand_total:<8}")
    print("="*75)
    # Output image paths for each class
    print("\n" + "="*75)
    print("Sample Image Paths per Class:")
    print("="*75)
    for i in range(len(class_names)):
        name = class_names[i]
        print(f"\nClass: {name}")
        
        # 7 images with bboxes
        print("  7 images with bboxes:")
        imgs_with_bbox = class_to_images[i][:7]
        if not imgs_with_bbox:
            print("    (None found)")
        for img in imgs_with_bbox:
            print(f"    - {img}")
            
        # 3 images without any bboxes
        print("  3 images without any bboxes:")
        imgs_without_bbox = no_bbox_images[:3]
        if not imgs_without_bbox:
            print("    (None found)")
        for img in imgs_without_bbox:
            print(f"    - {img}")
    print("="*75)
    
    return class_names


def visualize_image(dataset_root, image_path, class_names):
    """
    Loads an image and displays it with its bounding boxes using matplotlib.
    """
    # import ipdb; ipdb.sset_trace()
    dataset_root = Path(dataset_root)
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Error: Image {image_path} not found.")
        return

    # Try to find the corresponding label file
    label_path = None
    
    # Strategy 1: If it's within the dataset_root, use the images/labels structure
    if dataset_root in img_path.parents:
        parts = img_path.parts
        try:
            # Find where 'images' is in the path
            idx = -1
            for i, part in enumerate(parts):
                if part == 'images':
                    idx = i
                    break
            
            if idx != -1:
                # The label should be in the same parent directory but in 'labels'
                new_parts = list(parts)
                new_parts[idx] = 'labels'
                label_path = Path(*new_parts).with_suffix('.txt')
        except Exception:
            pass
            
    if not label_path or not label_path.exists():
        # Strategy 2: Try searching in all labels directories for the same stem
        stem = img_path.stem
        for split in ['train', 'valid', 'test']:
            possible_label = dataset_root / split / 'labels' / f"{stem}.txt"
            if possible_label.exists():
                label_path = possible_label
                break

    # Load image
    try:
        img = Image.open(img_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return
        
    img_width, img_height = img.size
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    if label_path and label_path.exists():
        print(f"Found label file: {label_path}")
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        # Convert normalized YOLO coordinates to pixel coordinates
                        xmin = (x_center - w / 2) * img_width
                        ymin = (y_center - h / 2) * img_height
                        width = w * img_width
                        height = h * img_height
                        # Draw bounding box
                        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        # Draw class label
                        label_text = class_names[class_id] if class_id < len(class_names) else f"ID: {class_id}"
                        ax.text(xmin, ymin - 5, label_text, color='white', fontsize=10, fontweight='bold',
                                bbox=dict(facecolor='red', alpha=0.5, pad=0))
        except Exception as e:
            print(f"Error reading label file {label_path}: {e}")
    else:
        if label_path:
             print(f"Warning: Label file {label_path} not found.")
        else:
             print(f"Warning: Could not find a corresponding label file for {img_path}")
    plt.axis('off')
    plt.title(f"Image: {img_path.name}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect YOLO dataset or visualize an image with bboxes.")
    parser.add_argument("--image", type=str, help="Path to an image file to visualize with bounding boxes.")
    parser.add_argument("--dataset", type=str, default="BoneFractureYolo8", help="Path to the dataset root.")
    # import ipdb; ipdb.sset_trace()
    
    args = parser.parse_args()
    
    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(os.getcwd(), dataset_path)

    if args.image:
        # Load class names from data.yaml first to know what to label the boxes with
        yaml_path = Path(dataset_path) / 'data.yaml'
        if not yaml_path.exists():
            print(f"Error: {yaml_path} not found. Cannot load class names.")
        else:
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                class_names = data.get('names', [])
                visualize_image(dataset_path, args.image, class_names)
            except Exception as e:
                print(f"Error: {e}")
    else:
        inspect_yolo_dataset(dataset_path)
