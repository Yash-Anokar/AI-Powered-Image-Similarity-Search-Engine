"""
Triplet Data Generator
Generates (anchor, positive, negative) triplets for training
Handles numeric filenames by grouping into categories
"""

import os
import cv2
import numpy as np
import random


class TripletDataGenerator:
    """Generate triplets (anchor, positive, negative) for training"""
    
    def __init__(self, image_folder, target_size=(224, 224), images_per_category=50):
        self.image_folder = image_folder
        self.target_size = target_size
        self.images_per_category = images_per_category
        self.load_dataset()
    
    def load_dataset(self):
        """Load all images and organize by category"""
        self.images = []
        self.labels = []
        self.label_to_images = {}
        
        print("=" * 60)
        print("Loading Dataset...")
        print("=" * 60)
        
        # Get all image files
        image_files = []
        for filename in os.listdir(self.image_folder):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_files.append(filename)
        
        # Sort files to ensure consistent ordering
        image_files.sort()
        
        print(f"Found {len(image_files)} image files")
        
        # Strategy 1: Try to extract categories from filenames
        has_categories = False
        for filename in image_files:
            # Try different naming patterns
            if '_' in filename:
                label = filename.split('_')[0]
                has_categories = True
            elif '-' in filename:
                label = filename.split('-')[0]
                has_categories = True
            else:
                break
        
        # Strategy 2: If no categories found, create them based on numeric ranges
        if not has_categories or len(image_files) < 100:
            print("No category prefixes found. Creating categories from filename ranges...")
            
            # Extract numeric part from filenames
            numeric_files = []
            for filename in image_files:
                try:
                    # Extract number from filename
                    num_str = ''.join(filter(str.isdigit, os.path.splitext(filename)[0]))
                    if num_str:
                        numeric_files.append((int(num_str), filename))
                except:
                    pass
            
            numeric_files.sort()
            
            if len(numeric_files) == 0:
                print("ERROR: Could not extract numeric IDs from filenames!")
                print("Please ensure files are named with numbers (e.g., 1163.jpg, 1525.jpg)")
                exit(1)
            
            # Group images into categories based on ranges
            for num, filename in numeric_files:
                # Group by hundreds (adjust this logic as needed)
                label = f"cat_{num // self.images_per_category}"
                
                img_path = os.path.join(self.image_folder, filename)
                self.images.append(img_path)
                self.labels.append(label)
                
                if label not in self.label_to_images:
                    self.label_to_images[label] = []
                self.label_to_images[label].append(img_path)
        
        else:
            # Use extracted categories from filenames
            print("Category prefixes detected in filenames")
            for filename in image_files:
                if '_' in filename:
                    label = filename.split('_')[0]
                elif '-' in filename:
                    label = filename.split('-')[0]
                else:
                    label = os.path.splitext(filename)[0]
                
                img_path = os.path.join(self.image_folder, filename)
                self.images.append(img_path)
                self.labels.append(label)
                
                if label not in self.label_to_images:
                    self.label_to_images[label] = []
                self.label_to_images[label].append(img_path)
        
        # Filter out categories with less than 2 images
        self.label_to_images = {
            k: v for k, v in self.label_to_images.items() if len(v) >= 2
        }
        
        if len(self.label_to_images) < 2:
            print("ERROR: Need at least 2 categories with 2+ images each!")
            print(f"Currently have {len(self.label_to_images)} valid categories")
            print("\nSolutions:")
            print("   1. Add more images to your dataset")
            print(f"   2. Adjust 'images_per_category' parameter (currently {self.images_per_category})")
            print("   3. Use category prefixes in filenames (e.g., shirt_001.jpg)")
            exit(1)
        
        print(f"Loaded {len(self.images)} images")
        print(f"Created {len(self.label_to_images)} categories")
        print("\nCategory Distribution:")
        for label, imgs in sorted(self.label_to_images.items()):
            print(f"   {label}: {len(imgs)} images")
        print("=" * 60)
    
    def load_and_preprocess_image(self, img_path):
        """Load and preprocess a single image"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)
        img = img.astype('float32') / 255.0
        return img
    
    def generate_triplet(self):
        """Generate one triplet (anchor, positive, negative)"""
        # Select random category for anchor
        anchor_label = random.choice(list(self.label_to_images.keys()))
        
        # Ensure at least 2 images in category for positive pair
        if len(self.label_to_images[anchor_label]) < 2:
            return self.generate_triplet()
        
        # Select anchor and positive from same category
        anchor_path, positive_path = random.sample(
            self.label_to_images[anchor_label], 2
        )
        
        # Select negative from different category
        negative_labels = [l for l in self.label_to_images.keys() if l != anchor_label]
        if not negative_labels:
            return self.generate_triplet()
        
        negative_label = random.choice(negative_labels)
        negative_path = random.choice(self.label_to_images[negative_label])
        
        # Load images
        try:
            anchor = self.load_and_preprocess_image(anchor_path)
            positive = self.load_and_preprocess_image(positive_path)
            negative = self.load_and_preprocess_image(negative_path)
            return anchor, positive, negative
        except Exception as e:
            print(f"Error loading triplet: {e}")
            return self.generate_triplet()
    
    def generate_batch(self, batch_size=32):
        """Generate a batch of triplets"""
        anchors = []
        positives = []
        negatives = []
        
        for _ in range(batch_size):
            a, p, n = self.generate_triplet()
            anchors.append(a)
            positives.append(p)
            negatives.append(n)
        
        return (
            np.array(anchors),
            np.array(positives),
            np.array(negatives)
        )
    
    def get_statistics(self):
        """Get dataset statistics"""
        return {
            'total_images': len(self.images),
            'num_categories': len(self.label_to_images),
            'categories': list(self.label_to_images.keys()),
            'images_per_category': {k: len(v) for k, v in self.label_to_images.items()}
        }
