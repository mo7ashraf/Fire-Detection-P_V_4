import os
import shutil
from pathlib import Path

class DFireValidationExtractor:
    """
    Extract validation images and labels for dFire dataset
    """
    
    def __init__(self, dataset_root, val_txt_file):
        """
        Args:
            dataset_root: Root directory of your dFire dataset
            val_txt_file: Path to validation.txt or val.txt file
        """
        self.dataset_root = Path('../../../dataset/dfire')
        self.val_txt_file = Path('../../../dataset/dfire/dfire_valid1.txt')
        
    def extract_validation_set(self, output_dir='validation_set'):
        """
        Extract validation images and labels based on val.txt
        
        Two scenarios handled:
        1. If val.txt contains image paths
        2. If you need to manually split from train set
        """
        output_path = Path(output_dir)
        val_images_dir = output_path / 'images'
        val_labels_dir = output_path / 'labels'
        
        val_images_dir.mkdir(parents=True, exist_ok=True)
        val_labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Reading validation file: {self.val_txt_file}")
        
        if not self.val_txt_file.exists():
            print(f"ERROR: {self.val_txt_file} not found!")
            return
        
        with open(self.val_txt_file, 'r') as f:
            lines = f.readlines()
        
        # Check if it's a list of paths or a split percentage file
        first_line = lines[0].strip()
        
        if self._is_image_path(first_line):
            # Scenario 1: File contains image paths
            self._extract_from_paths(lines, val_images_dir, val_labels_dir)
        else:
            print("The txt file doesn't contain image paths.")
            print("You may need to create validation split manually.")
            self._suggest_manual_split()
    
    def _is_image_path(self, line):
        """Check if line looks like an image path"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        return any(ext in line.lower() for ext in image_extensions)
    
    def _extract_from_paths(self, lines, val_images_dir, val_labels_dir):
        """Extract images and labels from path list"""
        copied_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Handle both absolute and relative paths
            img_path = Path(line)
            
            # If relative path, make it relative to dataset root
            if not img_path.is_absolute():
                img_path = self.dataset_root / img_path
            
            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue
            
            # Copy image
            shutil.copy2(img_path, val_images_dir / img_path.name)
            
            # Find and copy corresponding label file
            label_path = self._find_label_file(img_path)
            if label_path and label_path.exists():
                shutil.copy2(label_path, val_labels_dir / label_path.name)
                copied_count += 1
            else:
                print(f"Warning: Label not found for {img_path.name}")
        
        print(f"\n✓ Extracted {copied_count} validation image-label pairs")
        print(f"  Images: {val_images_dir}")
        print(f"  Labels: {val_labels_dir}")
    
    def _find_label_file(self, img_path):
        """Find corresponding label file for an image"""
        # Common label locations in YOLO datasets
        possible_label_dirs = [
            img_path.parent.parent / 'labels' / img_path.parent.name,
            img_path.parent.parent / 'labels',
            img_path.parent / 'labels',
        ]
        
        label_name = img_path.stem + '.txt'
        
        for label_dir in possible_label_dirs:
            label_path = label_dir / label_name
            if label_path.exists():
                return label_path
        
        return None
    
    def _suggest_manual_split(self):
        """Suggest manual splitting approach"""
        print("\n" + "="*60)
        print("MANUAL VALIDATION SPLIT SUGGESTION")
        print("="*60)
        print("\nOption 1: Use train_test_split to create validation set")
        print("Option 2: Use the separate validation download link")
        print("\nSee the code below for Option 1 implementation.")


def create_validation_split_from_train(train_images_dir, train_labels_dir, 
                                        val_images_dir, val_labels_dir, 
                                        val_ratio=0.2):
    """
    Create validation split from training data (if no validation exists)
    
    Args:
        train_images_dir: Directory containing training images
        train_labels_dir: Directory containing training labels
        val_images_dir: Directory to write validation images
        val_labels_dir: Directory to write validation labels
        val_ratio: Percentage to use for validation (default: 20%)
    """
    import random
    
    train_images_path = Path(train_images_dir)
    train_labels_path = Path(train_labels_dir)
    val_images_path = Path(val_images_dir)
    val_labels_path = Path(val_labels_dir)
    
    val_images_path.mkdir(parents=True, exist_ok=True)
    val_labels_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(train_images_path.glob('*.jpg')) + \
                  list(train_images_path.glob('*.jpeg')) + \
                  list(train_images_path.glob('*.png'))
    
    # Randomly select validation images
    random.seed(42)  # For reproducibility
    val_count = int(len(image_files) * val_ratio)
    if val_count == 0 and len(image_files) > 0:
        val_count = 1
    val_images = random.sample(image_files, val_count) if val_count > 0 else []
    
    print(f"Creating validation split: {val_count} images ({val_ratio*100}%)")
    
    copied = 0
    for img_path in val_images:
        # Copy image
        shutil.copy2(img_path, val_images_path / img_path.name)
        
        # Copy label
        label_path = train_labels_path / (img_path.stem + '.txt')
        if label_path.exists():
            shutil.copy2(label_path, val_labels_path / label_path.name)
            copied += 1
    
    print(f"✓ Created validation set with {copied} image-label pairs")
    print(f"  Location: images: {val_images_path} labels: {val_labels_path}")


def evaluate_model_on_validation(model_path, val_images_dir, val_labels_dir, 
                                 img_size=640, conf_threshold=0.25):
    """
    Evaluate trained YOLO model on validation data
    (Works with YOLOv5/YOLOv8/etc.)
    
    Args:
        model_path: Path to your trained model weights (.pt file)
        val_images_dir: Directory containing validation images
        val_labels_dir: Directory containing validation labels
        img_size: Image size for inference
        conf_threshold: Confidence threshold
    """
    print("\n" + "="*60)
    print("EVALUATING MODEL ON VALIDATION SET")
    print("="*60)
    
    # Try YOLOv8 (ultralytics) first
    try:
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(
            data=None,  # Will use the validation images directly
            imgsz=img_size,
            conf=conf_threshold,
            batch=16,
            save_json=True,
            plots=True
        )
        
        print(f"\n✓ Validation Results:")
        print(f"  mAP50: {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")
        print(f"  Precision: {results.box.mp:.4f}")
        print(f"  Recall: {results.box.mr:.4f}")
        
        return results
        
    except ImportError:
        print("Ultralytics not found. Trying YOLOv5...")
        
        # Try YOLOv5
        try:
            import torch
            
            # Load model
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            
            # You'll need to run val.py from YOLOv5 repo for full metrics
            print("\nFor YOLOv5, run this command:")
            print(f"python val.py --weights {model_path} --data your_data.yaml --img {img_size}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("\nPlease install required packages:")
            print("  pip install ultralytics  # For YOLOv8")
            print("  or")
            print("  pip install yolov5  # For YOLOv5")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("dFire Validation Data Extractor\n")
    '''
    # Example 1: Extract validation set from val.txt
    print("Example 1: Extract from val.txt file")
    print("-" * 60)
    extractor = DFireValidationExtractor(
        dataset_root='path/to/dfire_dataset',
        val_txt_file='path/to/val.txt'
    )
    extractor.extract_validation_set(output_dir='validation_set')
    
    print("\n" + "="*60 + "\n")
    '''
    # Example 2: Create validation split from training data (if val.txt doesn't work)
    print("Example 2: Create validation split from training set")
    print("-" * 60)
    print("Uncomment and run:")
    create_validation_split_from_train(
        train_images_dir='../../../dataset/dfire/train/images',
        train_labels_dir='../../../dataset/dfire/train/labels',
        val_images_dir='../../../dataset/fire_smoke_combined_new/images/val',
        val_labels_dir='../../../dataset/fire_smoke_combined_new/labels/val',
        val_ratio=0.2   
    )
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Evaluate model
    print("Example 3: Evaluate your trained model")
    print("-" * 60)
    print("Uncomment and run:")
    print("""
evaluate_model_on_validation(
    model_path='runs/train/exp/weights/best.pt',
    val_images_dir='validation_set/images',
    val_labels_dir='validation_set/labels',
    img_size=640,
    conf_threshold=0.25
)
    """)
    
    print("\n" + "="*60)
    print("\nNext Steps:")
    print("1. Update the paths in the examples above")
    print("2. Run the appropriate function for your situation")
    print("3. Evaluate your model - NO RETRAINING NEEDED!")
    print("="*60)