# TP1 KNN Classification

## 1. Overview

This project applies `K-Nearest Neighbors (KNN)` to the `CIFAR-10` dataset and compares performance under different distance metrics and `k` values.  
The notebook `TP1_KNN_Classification.ipynb` includes three experimental settings:

1. Simple Train/Test evaluation with a fixed `k`
2. Hyperparameter search using Train/Validation/Test split
3. Generalization analysis using 5-Fold Cross Validation

## 2. Dataset

The experiments use `torchvision.datasets.CIFAR10`.

- Original training set: `50,000` images
- Original test set: `10,000` images
- Image size: `32 x 32`
- Channels: `3 (RGB)`
- Number of classes: `10`

To reduce computation time, the notebook uses only a subset of the data:

- Final training samples: `8,000`
- Final test samples: `2,000`

## 3. Preprocessing

Since KNN requires vector inputs, each image is flattened into a one-dimensional feature vector and then standardized.

- Convert images using `ToTensor()`
- Flatten each image into a `3072 (= 32 x 32 x 3)`-dimensional vector
- Apply `StandardScaler` fitted on the training data
- Transform the test data using the same scaler

This preprocessing helps stabilize distance-based classification by reducing scale differences across pixel features.

## 4. Libraries

The notebook uses the following libraries:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `scikit-learn`

Main modules used in the implementation:

- `KNeighborsClassifier`
- `train_test_split`
- `KFold`
- `StandardScaler`
- `accuracy_score`, `precision_score`, `recall_score`, `f1_score`

## 5. Evaluation Metrics

All experiments report the following four metrics:

- Accuracy
- Precision (`macro`)
- Recall (`macro`)
- F1-score (`macro`)

Macro averaging is used to reflect performance across all classes more evenly.

## 6. Experimental Setup

### 6.1 Simple Train/Test Split

The most basic experiment trains on all `8,000` training samples and evaluates on `2,000` test samples.

- `k = 5`
- Distance metric: `euclidean`

### 6.2 Train/Validation/Test Split

The `8,000` training samples are split again into train and validation sets to find the best combination of `k` and distance metric.

- Validation ratio: `20%`
- `stratify=y_train`
- `random_state=42`
- Candidate `k` values: `[1, 3, 5, 7, 9]`
- Candidate metrics: `['euclidean', 'manhattan']`

The best hyperparameters are selected by validation accuracy. Then the model is retrained on the combined train and validation data and evaluated on the test set.

### 6.3 5-Fold Cross Validation

After selecting the best distance metric, 5-Fold Cross Validation is performed for multiple `k` values.

- `n_splits = 5`
- `shuffle = True`
- `random_state = 42`
- Metric: `manhattan`
- Candidate `k` values: `[1, 3, 5, 7, 9]`

For each `k`, the mean and standard deviation of Accuracy, Precision, Recall, and F1-score are reported.

## 7. Results

### 7.1 Simple Train/Test Result

Setting: `k=5`, `metric='euclidean'`

| Metric | Score |
|---|---:|
| Accuracy | 0.2765 |
| Precision | 0.3634 |
| Recall | 0.2765 |
| F1 | 0.2525 |

### 7.2 Hyperparameter Search with Validation Split

Validation accuracy results are shown below.

#### Euclidean Distance

| k | Validation Accuracy |
|---|---:|
| 1 | 0.2863 |
| 3 | 0.2888 |
| 5 | 0.2981 |
| 7 | 0.2812 |
| 9 | 0.2888 |

#### Manhattan Distance

| k | Validation Accuracy |
|---|---:|
| 1 | 0.3219 |
| 3 | 0.2975 |
| 5 | 0.3125 |
| 7 | 0.3025 |
| 9 | 0.3044 |

Best configuration:

- Best metric: `manhattan`
- Best k: `1`
- Best validation accuracy: `0.3219`

Test performance after retraining with the best configuration:

| Metric | Score |
|---|---:|
| Accuracy | 0.3025 |
| Precision | 0.3365 |
| Recall | 0.3025 |
| F1 | 0.2963 |

The test accuracy improved from `0.2765` to `0.3025` compared with the simple baseline setting.

### 7.3 5-Fold Cross Validation Result

With `manhattan` fixed as the distance metric, the 5-fold cross-validation results are:

| k | Accuracy (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---|---|---|---|
| 1 | 0.3100 ± 0.0111 | 0.3514 ± 0.0146 | 0.3104 ± 0.0107 | 0.3038 ± 0.0106 |
| 3 | 0.3041 ± 0.0069 | 0.3799 ± 0.0106 | 0.3046 ± 0.0075 | 0.2909 ± 0.0063 |
| 5 | 0.3180 ± 0.0056 | 0.3808 ± 0.0140 | 0.3184 ± 0.0055 | 0.3032 ± 0.0073 |
| 7 | 0.3139 ± 0.0051 | 0.3824 ± 0.0095 | 0.3144 ± 0.0058 | 0.2988 ± 0.0060 |
| 9 | 0.3179 ± 0.0070 | 0.3914 ± 0.0068 | 0.3183 ± 0.0078 | 0.3018 ± 0.0071 |

Based on mean accuracy, `k=5` and `k=9` produced the strongest average performance, while the best single validation split selected `k=1`.  
This shows that the optimal `k` can vary depending on the data split strategy.

## 8. Discussion

The results suggest the following:

- Hyperparameter tuning with a validation split performs better than using a fixed baseline setting.
- In this sampled CIFAR-10 experiment, `manhattan` distance outperformed `euclidean` distance on validation accuracy.
- Cross-validation results indicate that no single `k` overwhelmingly dominates in every setting.
- KNN is simple and interpretable, but its performance is limited on high-dimensional image data such as CIFAR-10.

## 9. How to Run

This assignment is implemented in Jupyter Notebook format.

### 9.1 Install Dependencies

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

### 9.2 Run the Notebook

```bash
jupyter notebook
```

Then open `TP1_KNN_Classification.ipynb` and run the cells in order.  
When executed for the first time, the `CIFAR-10` dataset will be downloaded automatically into the `./data` directory.

## 10. File Structure

- `TP1_KNN_Classification.ipynb`
  - Main notebook containing all experiments and outputs
- `README.md`
  - Summary of the project, setup, and results

## 11. Conclusion

This project explored KNN-based image classification on CIFAR-10 and compared the effects of different distance metrics and `k` values.  
The best single validation-based configuration was `manhattan distance` with `k=1`, achieving a test accuracy of `0.3025`.  
Additional 5-fold cross-validation showed that the preferred `k` can change depending on how the data is split.
