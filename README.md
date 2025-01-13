# Clothes_Classification

The goal is to classify e-commerce products into 27 categories. The data includes categorical features, a noisy text description and a noisy image for each product.

A transformer was trained on the text data, and three CNNs (ResNet, EfficientNet, and ResNeXt) were trained on the image data. The results were aggregated by stacking and voting.

### Installation

This project requires Pytorch with GPU.

Install the libraries

```
pip install -r requirements.txt
```

### Dataset

E-commerce products in 27 categories. The data includes categorical features, a noisy text description and a noisy image for each product.

Dataset directory structure
```
dataset
├── noisy-images
|   ├── 3257.jpg
|   └── ...
├── train.csv
└── test.csv
```
The image filenames match the ```id``` columns in the training and test sets. 

The dataset here is truncated. Please contact me for access to the whole dataset.

### Run

The code is in the notebook ```code.ipynb```.

Summary of steps:
1. Preprocess the datasets.
2. Split the original training set into a training set and a validation set.
3. Train base models, including image models and text models using the training set.
4. Select the best base models based on the validation set.
5. Freeze the base models, and train an ensemble head for each pair of image and text models.
6. Make predictions on the test set by voting.

Please refer to [workflow.md](./workflow.md) for more details.
