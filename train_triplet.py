"""
Quick Training Script for Small Datasets
Reduced epochs and steps for faster completion
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow import keras

from triplet_data_generator import TripletDataGenerator
from triplet_model import create_triplet_network, compile_triplet_model


class TripletTrainer:
    """Handles training of the Triplet Network"""
    
    def __init__(self, triplet_model, base_network, data_generator):
        self.triplet_model = triplet_model
        self.base_network = base_network
        self.data_generator = data_generator
        self.history = {'loss': [], 'val_loss': []}
    
    def train(self, epochs=10, batch_size=16, steps_per_epoch=20, 
              validation_split=0.2, save_every=5):
        """Quick training with reduced parameters"""
        
        print("\n" + "=" * 60)
        print("🚀 QUICK TRIPLET NETWORK TRAINING")
        print("=" * 60)
        print(f"📊 Training Configuration:")
        print(f"   Epochs: {epochs} (reduced from 50)")
        print(f"   Batch Size: {batch_size} (reduced from 32)")
        print(f"   Steps per Epoch: {steps_per_epoch} (reduced from 100)")
        print(f"   Estimated Time: ~5-10 minutes")
        print("=" * 60)
        
        val_steps = int(steps_per_epoch * validation_split)
        train_steps = steps_per_epoch - val_steps
        
        os.makedirs('checkpoints', exist_ok=True)
        
        for epoch in range(epochs):
            print(f"\n📈 Epoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            # Training phase
            train_losses = []
            for step in range(train_steps):
                anchors, positives, negatives = self.data_generator.generate_batch(batch_size)
                dummy_labels = np.zeros((batch_size, 1))
                
                loss = self.triplet_model.train_on_batch(
                    [anchors, positives, negatives],
                    dummy_labels
                )
                
                train_losses.append(loss)
                
                if (step + 1) % 5 == 0:
                    avg_loss = np.mean(train_losses[-5:])
                    print(f"   Step {step + 1}/{train_steps} - Loss: {avg_loss:.4f}")
            
            # Validation phase
            val_losses = []
            if val_steps > 0:
                for step in range(val_steps):
                    anchors, positives, negatives = self.data_generator.generate_batch(batch_size)
                    dummy_labels = np.zeros((batch_size, 1))
                    
                    loss = self.triplet_model.test_on_batch(
                        [anchors, positives, negatives],
                        dummy_labels
                    )
                    val_losses.append(loss)
            
            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses) if val_losses else 0.0
            
            self.history['loss'].append(epoch_train_loss)
            self.history['val_loss'].append(epoch_val_loss)
            
            print(f"\n✅ Epoch {epoch + 1} Summary:")
            print(f"   Training Loss: {epoch_train_loss:.4f}")
            if val_losses:
                print(f"   Validation Loss: {epoch_val_loss:.4f}")
            
            if (epoch + 1) % save_every == 0:
                checkpoint_path = f'checkpoints/triplet_base_epoch_{epoch + 1}.h5'
                self.base_network.save(checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 60)
        
        final_model_path = 'triplet_base_final.h5'
        self.base_network.save(final_model_path)
        print(f"✅ Final model saved: {final_model_path}")
        
        self.plot_training_history()
        
        return self.history
    
    def plot_training_history(self):
        """Plot training and validation loss"""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.history['loss'], label='Training Loss', linewidth=2)
            if self.history['val_loss'] and self.history['val_loss'][0] != 0:
                plt.plot(self.history['val_loss'], label='Validation Loss', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Triplet Loss', fontsize=12)
            plt.title('Quick Training History', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = 'training_history.png'
            plt.savefig(plot_path, dpi=300)
            print(f"📊 Training plot saved: {plot_path}")
            plt.close()
        except Exception as e:
            print(f"⚠ Could not save training plot: {e}")


def main():
    """Quick training function"""
    
    # REDUCED CONFIGURATION FOR FASTER TRAINING
    IMAGE_FOLDER = "static/dataset"
    INPUT_SHAPE = (224, 224, 3)
    EMBEDDING_DIM = 128
    MARGIN = 0.2
    LEARNING_RATE = 0.0001
    
    EPOCHS = 10              # Reduced from 50
    BATCH_SIZE = 16          # Reduced from 32
    STEPS_PER_EPOCH = 20     # Reduced from 100
    VALIDATION_SPLIT = 0.2
    SAVE_EVERY = 5
    
    print("\n" + "=" * 60)
    print("⚡ QUICK TRIPLET NETWORK TRAINING")
    print("=" * 60)
    print(f"⚠ WARNING: Only 9 images detected in categories!")
    print(f"   This is a test run with limited data")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Step 1: Load dataset
    print("\n📊 STEP 1: Loading Dataset")
    data_generator = TripletDataGenerator(IMAGE_FOLDER, target_size=INPUT_SHAPE[:2])
    stats = data_generator.get_statistics()
    
    # Step 2: Create model
    print("\n🏗️  STEP 2: Building Model")
    triplet_model, base_network = create_triplet_network(
        input_shape=INPUT_SHAPE,
        embedding_dim=EMBEDDING_DIM
    )
    
    # Step 3: Compile model
    print("\n⚙️  STEP 3: Compiling Model")
    compile_triplet_model(triplet_model, learning_rate=LEARNING_RATE, margin=MARGIN)
    
    # Step 4: Train model (QUICK VERSION)
    print("\n⚡ STEP 4: Quick Training")
    trainer = TripletTrainer(triplet_model, base_network, data_generator)
    history = trainer.train(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_split=VALIDATION_SPLIT,
        save_every=SAVE_EVERY
    )
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ QUICK TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📁 Generated Files:")
    print(f"   ✓ triplet_base_final.h5")
    print(f"   ✓ training_history.png")
    print(f"\n⚠ NOTE: Results may not be optimal with only 9 images")
    print(f"   Consider organizing more images into categories")
    print(f"\n🎯 Next Step: python extract_features_triplet.py")
    print("=" * 60)


if __name__ == "__main__":
    main()