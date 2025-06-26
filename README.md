# Pet Breed Classification with MobileNetV2 (Fine-Tuning)

This project presents a complete solution for classifying pet breeds using a fine-tuned MobileNetV2 model trained on the Oxford-IIIT Pet Dataset. The notebook implements a full machine learning workflow, including data preprocessing, augmentation, a two-phase training strategy, and detailed model evaluation.

---

## Notebook Overview

### Dataset

The project uses the **Oxford-IIIT Pet Dataset**, which includes 37 pet breed categories with around 200 images per class. Images and labels are loaded and processed using `tensorflow_datasets`.

### Model Architecture

A pre-trained **MobileNetV2** is used as the backbone for feature extraction. A custom classification head is added on top, consisting of Dense layers, BatchNormalization, and Dropout.

### Data Augmentation

On-the-fly data augmentation is integrated directly into the model pipeline using Keras preprocessing layers (`RandomFlip`, `RandomRotation`, `RandomZoom`), helping to improve generalization.

### Two-Phase Training Strategy

- **Phase 1: Feature Extraction**
  - The MobileNetV2 base is frozen (`base_model.trainable = False`).
  - Only the custom classification head is trained.
  - This phase allows the new layers to learn how to interpret the extracted features.

- **Phase 2: Fine-Tuning**
  - The MobileNetV2 base is unfrozen (`base_model.trainable = True`), except for BatchNormalization layers.
  - A significantly lower learning rate is used.
  - Fine-tunes the entire model to better adapt to the pet breed classification task.

### Callbacks Used

- `EarlyStopping`: Stops training to prevent overfitting.
- `ReduceLROnPlateau`: Dynamically adjusts learning rate based on validation loss.
- `TerminateOnNaN`: Halts training in case of numerical instability.

### Evaluation and Visualization

The notebook includes:

- Final loss and accuracy on the validation set.
- Training/validation accuracy and loss plots over epochs.
- Visual predictions:
  - Displaying the true and predicted label.
  - Showing the confidence score.
  - Listing the top 5 predicted classes with probabilities.
