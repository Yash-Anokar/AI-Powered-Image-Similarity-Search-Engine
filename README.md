# AI Image Similarity Search

Find visually similar images using deep learning. Train a Triplet Network on your dataset and search through thousands of images instantly.

## Quick Start

1. **Prepare images**: Place your images in `static/dataset/`
2. **Train model**: `python train_triplet.py`
3. **Extract features**: `python extract_features_triplet.py`
4. **Start server**: `python app_triplet.py`
5. **Open browser**: Visit `http://localhost:5000`

## Project Files

### Core Model

**triplet_model.py** - Deep learning model architecture
- `triplet_loss()`: Loss function for training
- `create_base_network()`: CNN backbone (ResNet-like architecture)
- `create_triplet_network()`: Three-input Siamese network
- `compile_triplet_model()`: Compile with Adam optimizer
- Inputs: 224x224 RGB images
- Outputs: 128-dimensional embeddings

### Data Pipeline

**triplet_data_generator.py** - Dataset management and triplet generation
- `TripletDataGenerator` class loads images from folder
- Auto-detects categories from filenames (e.g., `shirt_001.jpg`, `pants_001.jpg`)
- `load_dataset()`: Organizes images by category
- `generate_triplet()`: Creates (anchor, positive, negative) triplets
- `generate_batch()`: Returns batches of triplets for training
- `get_statistics()`: Shows dataset breakdown

### Training

**train_triplet.py** - Training script
- `TripletTrainer` class handles model training
- Loads data using `TripletDataGenerator`
- Trains for configurable epochs with validation
- Saves checkpoints every 5 epochs
- Outputs: `triplet_base_final.h5` (trained model)
- Quick version: 10 epochs, 20 steps per epoch
- Generates `training_history.png` visualization

**Configuration options:**
```python
EPOCHS = 10              # Training epochs
BATCH_SIZE = 16        # Triplets per batch
STEPS_PER_EPOCH = 20   # Batches per epoch
LEARNING_RATE = 0.0001 # Adam optimizer rate
MARGIN = 0.2           # Triplet loss margin
```

### Feature Extraction

**extract_features_triplet.py** - Generate embeddings and index
- `TripletFeatureExtractor` class loads trained model
- `extract_features()`: Get embedding for single image
- `extract_all_features()`: Process entire dataset
- Outputs:
  - `triplet_features.npy`: All image embeddings
  - `triplet_images.npy`: Image filenames
  - `triplet_embedding_space.png`: t-SNE visualization (optional)

**feature_extraction.py** - Alternative using MobileNetV2 (reference)
- Pre-trained CNN-based feature extraction
- Not used if triplet network is available
- Kept for comparison

### Web Application

**app_triplet.py** - Flask server
- `@app.route('/')`: Main page
- `@app.route('/search')`: Upload image and find similar results
  - Input: Image file, number of results, similarity threshold
  - Output: JSON with similar images and similarity scores
- `@app.route('/stats')`: System statistics
- Loads `triplet_base_final.h5` model
- Loads `triplet_features.npy` embeddings
- Returns results with Euclidean distance scoring

**index.html** (root) & **templates/index.html** - Web interface
- Drag-drop image upload zone
- Preview uploaded image
- Adjustable number of results (5-50)
- Similarity threshold slider (0-100%)
- Results grid with similarity percentages
- Real-time search with loading indicator

### Utility

**rename_images.py** - Organize images into categories
- Interactive script to rename files
- Groups images into categories
- Creates pattern: `category_001.jpg`, `category_002.jpg`, etc.
- Example:
  ```bash
  python rename_images.py
  # Enter 3 categories: shirt, pants, dress
  # Distributes images evenly across categories
  ```

**requirements.txt** - Python dependencies
```
tensorflow==2.13.0
opencv-python==4.12.0.88
scikit-learn==1.3.2
numpy==1.24.3
flask==3.0.0
werkzeug==3.0.6
matplotlib==3.7.1
hnswlib==0.7.1
gunicorn==20.1.0
```

## Data Format

### Image Naming
Images must follow this pattern to auto-detect categories:
```
shirt_001.jpg       Category: "shirt"
shirt_002.jpg       Category: "shirt"
pants_001.jpg       Category: "pants"
pants_002.jpg       Category: "pants"
dress_001.jpg       Category: "dress"
```

Or use prefixes with delimiter:
```
shirt-001.jpg
pants-001.jpg
dress-001.jpg
```

Minimum requirement: 2 categories with 2+ images each

### Directory Structure
```
static/
  dataset/          # Your images go here
    shirt_001.jpg
    shirt_002.jpg
    pants_001.jpg
    ...
  uploads/          # Temporary upload storage (auto-created)

templates/
  index.html        # Web UI template

triplet_model.py         # Model architecture
triplet_data_generator.py # Data loading
train_triplet.py         # Training
extract_features_triplet.py # Feature extraction
app_triplet.py           # Flask server
index.html               # Static HTML
```

## Workflow

### 1. Organize Images
```bash
python rename_images.py
# Distributes images into categories automatically
# Creates: shirt_001.jpg, shirt_002.jpg, pants_001.jpg, etc.
```

### 2. Train Model
```bash
python train_triplet.py
# Loads images from static/dataset/
# Generates triplet batches (anchor, positive, negative from different categories)
# Trains with triplet loss
# Saves: triplet_base_final.h5
```

### 3. Extract Features
```bash
python extract_features_triplet.py
# Loads trained model
# Extracts 128-d embeddings for all images
# Saves: triplet_features.npy, triplet_images.npy
# Creates visualization: triplet_embedding_space.png
```

### 4. Run Web Server
```bash
python app_triplet.py
# Starts Flask on http://localhost:5000
# Loads model and embeddings
# Ready for searches
```

## How It Works

### Triplet Loss Training
The model learns by comparing three images simultaneously:
- **Anchor**: Reference image
- **Positive**: Similar image (same category)
- **Negative**: Different image (different category)

Loss function pushes positive closer and negative further away:
```
Loss = max(distance(anchor, positive) - distance(anchor, negative) + margin, 0)
```

### Similarity Search
After training, searching works like this:
1. Extract embedding of uploaded image (128-d vector)
2. Calculate Euclidean distance to all dataset embeddings
3. Sort by distance (smaller = more similar)
4. Convert to similarity percentage
5. Return top N results above threshold

## Customization

### Change Embedding Dimension
Edit `triplet_model.py`:
```python
embedding_dim = 256  # Instead of 128
```

### Adjust Training
Edit `train_triplet.py`:
```python
EPOCHS = 50              # More training
BATCH_SIZE = 32         # Larger batches
LEARNING_RATE = 0.00005 # Slower learning
MARGIN = 0.5            # Harder triplets
```

### Change Image Size
Edit `triplet_model.py`:
```python
input_shape = (128, 128, 3)  # Instead of (224, 224, 3)
```

## Requirements

- Python 3.9+
- TensorFlow 2.13+
- OpenCV 4.12+
- NumPy 1.24+
- Flask 3.0+
- 4GB+ RAM

Install dependencies:
```bash
pip install -r requirements.txt
```

## Troubleshooting

**"No categories detected"**
- Rename images: `python rename_images.py`
- Or manually use pattern: `category_001.jpg`

**"Model not loaded"**
- Run training first: `python train_triplet.py`

**"No embeddings loaded"**
- Extract features first: `python extract_features_triplet.py`

**Web server won't start**
- Check port 5000 is free
- Check all files exist: `triplet_base_final.h5`, `triplet_features.npy`, `triplet_images.npy`

**Out of memory**
- Reduce `BATCH_SIZE` in `train_triplet.py`
- Reduce `STEPS_PER_EPOCH`

## API Response

### /search (POST)
```json
{
  "results": [
    {
      "filename": "shirt_002.jpg",
      "similarity": 94.5,
      "embedding_distance": 0.18
    },
    {
      "filename": "shirt_003.jpg",
      "similarity": 89.2,
      "embedding_distance": 0.32
    }
  ],
  "total_results": 2
}
```

### /stats (GET)
```json
{
  "model_loaded": true,
  "embeddings_loaded": true,
  "total_images": 1008,
  "embedding_dimension": 128,
  "model_type": "Triplet Network",
  "similarity_metric": "Euclidean Distance"
}
```

## File Purposes at a Glance

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| triplet_model.py | Model architecture | - | Keras model class |
| triplet_data_generator.py | Load & batch images | Image folder | Triplet batches |
| train_triplet.py | Train model | triplet_data_generator | triplet_base_final.h5 |
| extract_features_triplet.py | Get embeddings | triplet_base_final.h5 | .npy files |
| app_triplet.py | Web server | Models + embeddings | JSON responses |
| index.html | Web UI | - | User interface |
| rename_images.py | Organize images | Image folder | Renamed files |

## Performance

- Training: 5-10 minutes (10 epochs, 1000 images)
- Feature extraction: 2-3 minutes (1000 images)
- Single search: <100ms
- Batch size: 16 triplets per step
- Embedding size: 128 dimensions

---

**Ready to use. Start with:** `python rename_images.py` → `python train_triplet.py` → `python extract_features_triplet.py` → `python app_triplet.py`
   - Use data augmentation during training

## Future Improvements

Planned enhancements:
- Hard negative mining for better training
- Automatic threshold tuning
- Batch API endpoint
- Containerized deployment (Docker)
- Mobile inference support
- Incremental index updates

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features
- Improve documentation
- Submit pull requests

## Citation

If you use this project in your research or work, please cite:
```
@software{image_similarity_search,
  title={AI Image Similarity Search},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/AI-Image-Similarity-Search}
}
```

## Support

For questions, issues, or suggestions:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Include error messages and steps to reproduce
4. Attach sample images if relevant

## Acknowledgments

Built with:
- TensorFlow/Keras for deep learning
- OpenCV for image processing
- hnswlib for efficient indexing
- Flask for web interface

---

Last updated: January 2026

For more information, visit the GitHub repository or check the project wiki.
