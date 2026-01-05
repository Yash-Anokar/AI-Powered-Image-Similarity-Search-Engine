"""
Feature Extraction using Trained Triplet Network
Extracts embeddings for all images in the dataset
"""

import os
import cv2
import numpy as np
from tensorflow import keras
from triplet_model import triplet_loss


class TripletFeatureExtractor:
    """Extract embeddings using trained triplet network"""
    
    def __init__(self, model_path='triplet_base_final.h5'):
        """
        Load the trained base network
        
        Args:
            model_path: Path to the saved trained model
        """
        print("=" * 60)
        print("📦 Loading Trained Triplet Network")
        print("=" * 60)
        
        try:
            self.model = keras.models.load_model(
                model_path,
                custom_objects={'triplet_loss': triplet_loss},
                compile=False
            )
            print(f"✅ Model loaded from: {model_path}")
            print(f"📊 Model parameters: {self.model.count_params():,}")
            print(f"📐 Input shape: {self.model.input_shape}")
            print(f"📐 Output shape: {self.model.output_shape}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\n💡 Make sure you have:")
            print("   1. Trained the model (run train_triplet.py)")
            print("   2. The model file exists: triplet_base_final.h5")
            exit(1)
        
        print("=" * 60)
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to the image file
            target_size: Target size for resizing
        
        Returns:
            Preprocessed image array ready for model input
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def extract_features(self, image_path):
        """
        Extract embedding for a single image
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Embedding vector (flattened numpy array)
        """
        img = self.preprocess_image(image_path)
        embedding = self.model.predict(img, verbose=0)
        return embedding.flatten()
    
    def extract_all_features(self, image_folder, output_prefix='triplet'):
        """
        Extract embeddings for all images in folder
        
        Args:
            image_folder: Folder containing images
            output_prefix: Prefix for output files
        
        Returns:
            features_list: List of feature vectors
            image_names: List of corresponding image filenames
        """
        print("\n" + "=" * 60)
        print("🔍 Extracting Features from Dataset")
        print("=" * 60)
        print(f"📁 Image folder: {image_folder}")
        
        features_list = []
        image_names = []
        failed_images = []
        
        # Get all image files
        image_files = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        
        total_images = len(image_files)
        print(f"📊 Total images found: {total_images}")
        print("-" * 60)
        
        for i, filename in enumerate(image_files, 1):
            img_path = os.path.join(image_folder, filename)
            
            try:
                # Extract features
                features = self.extract_features(img_path)
                features_list.append(features)
                image_names.append(filename)
                
                # Progress indicator
                if i % 10 == 0 or i == total_images:
                    print(f"✓ Processed {i}/{total_images} images ({i/total_images*100:.1f}%)")
                    
            except Exception as e:
                failed_images.append((filename, str(e)))
                print(f"⚠ Failed to process {filename}: {e}")
        
        print("-" * 60)
        print(f"✅ Successfully processed: {len(features_list)}/{total_images} images")
        
        if failed_images:
            print(f"⚠ Failed images: {len(failed_images)}")
            for fname, error in failed_images[:5]:  # Show first 5 failures
                print(f"   - {fname}: {error}")
        
        # Save features to disk
        if len(features_list) > 0:
            features_array = np.array(features_list)
            images_array = np.array(image_names)
            
            features_file = f"{output_prefix}_features.npy"
            images_file = f"{output_prefix}_images.npy"
            
            np.save(features_file, features_array)
            np.save(images_file, images_array)
            
            print("\n" + "=" * 60)
            print("💾 Files Saved")
            print("=" * 60)
            print(f"✅ {features_file}")
            print(f"   Shape: {features_array.shape}")
            print(f"   Size: {features_array.nbytes / 1024:.2f} KB")
            print(f"✅ {images_file}")
            print(f"   Count: {len(images_array)} filenames")
            print("=" * 60)
        else:
            print("❌ No features extracted!")
        
        return features_list, image_names
    
    def visualize_embedding_space(self, features, labels=None, output_file='embedding_space.png'):
        """
        Visualize the embedding space using t-SNE (optional)
        Requires: pip install scikit-learn matplotlib
        """
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            
            print("\n📊 Generating embedding space visualization...")
            
            # Reduce to 2D using t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
            embeddings_2d = tsne.fit_transform(features)
            
            # Plot
            plt.figure(figsize=(12, 8))
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                       alpha=0.6, s=50, c=range(len(embeddings_2d)), cmap='viridis')
            plt.colorbar(label='Image Index')
            plt.title('Triplet Network Embedding Space (t-SNE)', fontsize=14, fontweight='bold')
            plt.xlabel('t-SNE Dimension 1', fontsize=12)
            plt.ylabel('t-SNE Dimension 2', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            plt.close()
            
            print(f"✅ Visualization saved: {output_file}")
            
        except ImportError:
            print("⚠ Skipping visualization (scikit-learn not installed)")
        except Exception as e:
            print(f"⚠ Could not create visualization: {e}")


def main():
    """Main feature extraction function"""
    
    # Configuration
    MODEL_PATH = 'triplet_base_final.h5'
    IMAGE_FOLDER = 'static/dataset'
    OUTPUT_PREFIX = 'triplet'
    
    print("\n" + "=" * 60)
    print("🎯 TRIPLET FEATURE EXTRACTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load trained model
    extractor = TripletFeatureExtractor(MODEL_PATH)
    
    # Step 2: Extract features for all images
    features, image_names = extractor.extract_all_features(
        IMAGE_FOLDER, 
        output_prefix=OUTPUT_PREFIX
    )
    
    # Step 3: Optional visualization
    if len(features) > 0:
        print("\n📊 Creating embedding space visualization...")
        extractor.visualize_embedding_space(
            np.array(features),
            output_file='triplet_embedding_space.png'
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ FEATURE EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"📁 Generated Files:")
    print(f"   ✓ triplet_features.npy ({len(features)} embeddings)")
    print(f"   ✓ triplet_images.npy ({len(image_names)} filenames)")
    print(f"   ✓ triplet_embedding_space.png (visualization)")
    print(f"\n🎯 Next Step: Run Flask app with updated features")
    print(f"   Update app.py to load 'triplet_features.npy'")
    print("=" * 60)


if __name__ == "__main__":
    main()