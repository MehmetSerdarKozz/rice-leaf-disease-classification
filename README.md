# 🌾 Rice Leaf Disease Classification

Comparing three approaches to classifying **10 rice leaf conditions** on the **Paddy Doctor** dataset: transfer learning, deep features + classical ML, and handcrafted features + classical ML.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

## 📌 Highlights

- **Best result: DenseNet121 features + SVM → 89.7% accuracy** (5-fold CV)
- Transfer learning with a frozen backbone reached **72.5%** (best accuracy: EfficientNetB0; best F1: ResNet50V2) on the hold-out test set
- Handcrafted features + SVM reached **67.9%**, and **HOG alone carried ~95%** of the Random Forest feature importance

## 🗂️ Dataset

- Paddy Doctor, 10 classes: `bacterial_leaf_blight`, `bacterial_leaf_streak`, `bacterial_panicle_blight`, `blast`, `brown_spot`, `dead_heart`, `downy_mildew`, `hispa`, `normal`, `tungro`
- Balanced: **320 images per class, 3,200 in total**
- Method 1 split: 2,240 train / 320 validation / 640 test (70/10/20), stratified (64 test images per class)

The dataset is not included in this repository because of its size.

## 🧪 Methods

| # | Notebook | Approach | Evaluation |
|---|---|---|---|
| 1 | `01_transfer_learning.ipynb` | 8 ImageNet-pretrained CNNs, frozen backbone + custom classification head | Hold-out test (640 images) |
| 2 | `02_dl_features_classical_ml.ipynb` | Deep features from the same 8 CNNs → 5 classical ML models (40 combinations) | 5-fold stratified CV (3,200 images) |
| 3 | `03_handcrafted_features_classical_ml.ipynb` | Handcrafted features (HOG, GLCM, LBP, wavelet, Canny, Hu moments, pixel statistics) → 5 classical ML models | 5-fold stratified CV (3,200 images) |

### Method 1: Transfer learning
- Backbones: Xception, DenseNet121, InceptionV3, ResNet101V2, MobileNetV2, EfficientNetB0, VGG16, ResNet50V2
- Input 256×256, global average pooling, then Dense 128 → 64 → 32 with dropout (0.4 / 0.3 / 0.2), softmax output
- Adam (lr 1e-3), up to 50 epochs, `EarlyStopping` + `ReduceLROnPlateau`, light augmentation (rotation, horizontal flip)

### Method 2: Deep features + classical ML
- Features from the global average pooling layer of each CNN
- Classifiers: SVM (RBF, C=10), Random Forest, KNN, Logistic Regression, XGBoost
- PCA (95% variance) applied to the 2048-dimensional feature sets

### Method 3: Handcrafted features + classical ML
- 128×128 images; features: HOG, GLCM texture, LBP, wavelet (Haar), Canny edge statistics, Hu moments, pixel statistics
- Standardization + PCA (95% variance, 1,288 components), same 5 classifiers

## 📊 Results

### Overall comparison (best model per method)

| Method | Best model | Accuracy | F1-score | Evaluation |
|---|---|---|---|---|
| **2. Deep features + ML** | **DenseNet121 + SVM** | **89.75%** | **0.897** | 5-fold CV |
| 1. Transfer learning | ResNet50V2 | 72.19% | 0.722 | Hold-out |
| 3. Handcrafted + ML | SVM | 67.94% | 0.681 | 5-fold CV |

> ⚠️ Method 1 is scored on a single 640-image test split, Methods 2 and 3 on 5-fold CV over all 3,200 images. The numbers are indicative, not a strictly identical protocol.

### Method 1: all 8 CNNs (hold-out test)

| Model | Accuracy | F1-score |
|---|---|---|
| ResNet50V2 | 72.19% | 0.7221 |
| EfficientNetB0 | 72.50% | 0.7196 |
| ResNet101V2 | 71.72% | 0.7176 |
| DenseNet121 | 70.78% | 0.7068 |
| MobileNetV2 | 70.16% | 0.6984 |
| InceptionV3 | 68.13% | 0.6800 |
| Xception | 65.47% | 0.6546 |
| VGG16 | 62.97% | 0.6295 |

### Method 2: top 5 combinations (5-fold CV)

| DL model | ML model | Accuracy | F1-score |
|---|---|---|---|
| DenseNet121 | SVM | 89.75% | 0.8973 |
| EfficientNetB0 | SVM | 87.38% | 0.8736 |
| MobileNetV2 | SVM | 85.66% | 0.8558 |
| Xception | SVM | 83.69% | 0.8364 |
| DenseNet121 | KNN | 83.56% | 0.8338 |

### Method 3: classical models on handcrafted features (5-fold CV)

| Model | Accuracy | F1-score |
|---|---|---|
| SVM | 67.94% | 0.6808 |
| KNN | 64.12% | 0.6405 |
| XGBoost | 56.06% | 0.5575 |
| Logistic Regression | 51.66% | 0.5147 |
| Random Forest | 50.87% | 0.5039 |

Feature-group importance (Random Forest): HOG 95.1%, GLCM 2.1%, LBP 1.1%, the rest under 0.5% each.

## 💡 Takeaways

- Deep features with an SVM clearly beat the end-to-end frozen-backbone heads (≈ +17 points).
- SVM was the strongest classical model on both deep and handcrafted features.
- Hardest classes: `hispa`, `tungro`, `brown_spot`. Easiest: `bacterial_panicle_blight`, `bacterial_leaf_streak`, `dead_heart`.
- Next step: fine-tune the backbones (Method 1 used frozen weights only) and evaluate all methods on the same split.

## 🚀 Run it

1. Open a notebook in Google Colab (GPU runtime).
2. Download the Paddy Doctor dataset and put it in your Drive, then update the path variables at the top of each notebook (`SPLIT_DIR`, `OUTPUT_DIR`).
3. Run Method 1 first. Method 2 builds on the weights saved by Method 1.

## 👤 Author

**Mehmet Serdar** · [LinkedIn](https://www.linkedin.com/in/mehmet-serdar-koz/) · [GitHub](https://github.com/MehmetSerdarKozz)
