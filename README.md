# Pet Breed Classification with MobileNetV2 (Fine-Tuning)

**A single Jupyter notebook that fine-tunes a pretrained MobileNetV2 to classify the 37 cat and dog breeds of the Oxford-IIIT Pet Dataset, for anyone learning transfer learning with Keras.**

> **Context.** Built in June 2025 as a personal project, when I was a Computer Engineering student at UPV and had just started publishing on GitHub with machine learning and deep learning exercises. It is one of my first public repos: a compact transfer learning exercise, kept as it was written and not maintained since.

## What it does

- Loads the Oxford-IIIT Pet Dataset (37 breeds) through `tensorflow_datasets`, using its `train` split (3,680 images) for training and its `test` split (3,669 images) as validation.
- Resizes images to 224x224 and applies the MobileNetV2 input normalization to the range [-1, 1].
- Builds a classifier on an ImageNet-pretrained MobileNetV2 with in-model augmentation and a small dense head.
- Trains in two phases: feature extraction with the base frozen, then fine-tuning with the base unfrozen.
- Plots accuracy and loss across both phases and prints sample predictions with their top 5 probabilities.

## How it works

1. Data: `tfds.load('oxford_iiit_pet')` with `as_supervised=True`, batch size 32, shuffle buffer of 1000, prefetch.
2. Model: augmentation layers (`RandomFlip` horizontal, `RandomRotation(0.05)`, `RandomZoom(0.1)`), MobileNetV2 without its top, `GlobalAveragePooling2D`, `Dense(128, relu)`, `BatchNormalization`, `Dropout(0.3)` and a softmax layer over 37 classes.
3. Phase 1: base frozen, Adam at 1e-3, up to 20 epochs.
4. Phase 2: base unfrozen except its `BatchNormalization` layers, Adam at 1e-4, continuing from the epoch where phase 1 stopped.
5. Both phases use `EarlyStopping` on `val_loss` (patience 3, best weights restored), `ReduceLROnPlateau` (factor 0.5) and `TerminateOnNaN`.

## Results

Printed in the notebook outputs (run on an NVIDIA RTX 3070 Laptop GPU, 26 June 2025):

- Phase 1 stopped at epoch 8, restoring the weights of epoch 5.
- Phase 2 stopped at epoch 23, restoring the weights of epoch 20.
- Final evaluation on the `test` split used as validation: loss 0.3606, accuracy 0.8983.
- Training accuracy reaches 1.0000 in the last epochs while validation accuracy stays near 0.90, so the model overfits the training set.
- The notebook also prints 15 sample predictions; one is wrong (a Leonberger predicted as wheaten terrier with probability 0.25).

There is no separate held-out test set: the same `test` split drives early stopping, learning rate reduction and the final number.

## Run it

Not re-run for this documentation. The notebook has no requirements file, so install the libraries it imports:

```bash
pip install tensorflow keras tensorflow-datasets matplotlib numpy
jupyter notebook MobileNetV2_Classifier.ipynb
```

Run the cells in order. The first run downloads the dataset and the ImageNet weights, so it needs internet access. A GPU is recommended; the recorded run used one. Some printed warning messages inside the notebook are written in Spanish.

## Repository layout

```
MobileNetV2_Classifier.ipynb   full workflow: data, model, two-phase training, evaluation, sample predictions
README.md                      this file
```

## Stack

Python, TensorFlow, Keras, TensorFlow Datasets, Matplotlib, NumPy, Jupyter.
