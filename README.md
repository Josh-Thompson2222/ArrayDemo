# Pharmaceutical Drug Classification CNN

This repository contains a Convolutional Neural Network (CNN) for classifying pharmaceutical drugs and vitamins from synthetic images.

## Model Training & Checkpointing

The CNN model (`Pill_Image_CNN.ipynb`) now includes **automatic epoch training checkpointing** to ensure your training progress is saved:

### Features:

1. **Model Checkpointing During Training**
   - The best model is automatically saved during training based on validation accuracy
   - Checkpoint files are saved in the `model_checkpoints/` directory
   - Each checkpoint filename includes the epoch number and validation accuracy for easy tracking
   - Format: `best_model_epoch_XX_val_acc_X.XXXX.h5`

2. **Final Model Saving**
   - After training completes, the final model is saved in two formats:
     - `final_trained_model.h5` (HDF5 format - compatible with older TensorFlow/Keras)
     - `final_trained_model.keras` (Keras format - recommended for TensorFlow 2.x)

### Saved Files:

- **During Training**: `model_checkpoints/best_model_epoch_*.h5` - Best model based on validation accuracy
- **After Training**: 
  - `final_trained_model.h5` - Final model in HDF5 format
  - `final_trained_model.keras` - Final model in Keras format

### Loading Saved Models:

To load a saved model for inference or further training:

```python
from tensorflow import keras

# Load the final model
model = keras.models.load_model('final_trained_model.keras')

# Or load a specific checkpoint
model = keras.models.load_model('model_checkpoints/best_model_epoch_08_val_acc_0.9234.h5')
```

### Note:

Model files (`.h5`, `.keras`) and the `model_checkpoints/` directory are excluded from git via `.gitignore` to avoid committing large binary files.

## Dataset

The model is trained on the [Pharmaceutical Drugs and Vitamins Synthetic Images](https://www.kaggle.com/datasets/vencerlanz09/pharmaceutical-drugs-and-vitamins-synthetic-images) dataset from Kaggle.

## Requirements

- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- scikit-learn
- kagglehub (for dataset download)
