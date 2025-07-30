# Classification of Biased Datasets with Random Forest

**Author:** Marco Fasoli
**Supervisors:** Prof. Vito Paolo Pastore, Dr. Massimiliano Ciranni

---

## 📖 1. Project Overview & Introduction

This project investigates the impact of dataset bias on the performance and learning behavior of Random Forest models. **Data bias** refers to a spurious correlation between a feature and a class label, leading to systematic distortions in a dataset. For example, if the digit "0" is almost always colored red, a model might incorrectly learn that "red" means "0".

Using a custom-colored MNIST dataset (CMNIST), we demonstrate how Random Forest, a traditionally interpretable model, can learn these spurious correlations (like color) instead of the essential features of the data (the shape of the digits). The analysis shows that a model trained on a biased dataset fails to generalize, performing poorly on data that doesn't conform to the learned bias. An item is considered **bias-aligned** if its spurious feature matches the correlation (a red "0") and **bias-conflicting** otherwise (a green "0").

The project further explores this phenomenon by:

1. Training models on features extracted from pre-trained neural networks (a biased MLP and a more general ResNet50).
2. Comparing these results against models trained on balanced and unbiased (greyscale) datasets.
3. Conducting in-depth analyses of feature importance and Gini impurity reduction to visually and quantitatively prove that the biased model takes "shortcuts" based on color.

---

## 🎯 2. Objective

The primary goal is to **evaluate the dependency on bias in the predictions of a Random Forest algorithm**. We aim to understand how an interpretable model reacts to data with strong, spurious correlations. While the study of bias is a critical topic in Deep Learning, where complex models are known to learn spurious correlations, this project focuses on demonstrating the vulnerability of a foundational machine learning algorithm.

---

## 💾 3. Dataset: Colored MNIST (CMNIST)

This project uses a modified version of the MNIST dataset where each digit image is assigned a specific color. The dataset is intentionally biased to create a strong, but spurious, correlation between the color and the digit's class label.

* **Bias Aligned (95% of Training Data)**: The vast majority of images for each digit are colored with a predefined color associated with its class label (e.g., '0' is red, '1' is orange, '2' is yellow).
* **Bias Conflicting (5% of Training Data)**: A small fraction of images are colored "incorrectly," using a color associated with a different digit, creating a challenge for the model's generalization capability.

The data is pre-split into `training`, `validation`, and `testing` sets. The input for the Random Forest model is each $28 \times 28 \times 3$ image, flattened into a vector of 2352 features.

---

## 🌳 4. Model: Random Forest Classifier

The **Random Forest** is an ensemble learning method that constructs a "forest" of decision trees during training.

* **Forest Construction**: Each tree is trained on a random subset of the training data, selected using the bootstrap method (sampling with replacement).
* **Prediction**: The final classification is determined by a majority vote among all the individual trees in the forest.

---

## 🔬 5. Methodology & Experiments

To evaluate the model's dependency on bias, we trained several Random Forest models on different data representations.

### Model Training Scenarios

1.  **Model 1: Raw CMNIST Images**: A standard `RandomForestClassifier` trained directly on the flattened pixel vectors of the biased CMNIST images.
2.  **Model 2: Features from a Biased NN**: A Random Forest trained on features extracted from a 3-layer MLP that was itself pre-trained on the biased CMNIST dataset. These features predominantly represent color information.
3.  **Model 3: Features from ResNet50**: A model trained on features extracted from a ResNet50 network (pre-trained on ImageNet). These features are expected to be more robust and capture more general structural information.

### Control Experiments (Sanity Checks)

4.  **Random Forest on a Balanced Dataset**: To counteract the bias, a model was trained on a dataset where the number of "bias-aligned" and "bias-conflicting" images was equal.
5.  **Random Forest on Greyscale MNIST (Unbiased Control)**: A baseline model was trained on the original greyscale MNIST dataset. This model acts as a control, showing the performance of a Random Forest when no color bias is present.

### In-Depth Analysis

* **Feature Importance Visualization**: We trained 10 separate binary classifiers (e.g., "is a 0" vs. "is not a 0") to see what pixels the model deemed important. The resulting maps show the biased model focuses on color, while the unbiased model correctly learns the digit's shape.
* **Gini Impurity Analysis**: By plotting the average Gini impurity at each depth of a decision tree, we visualized the learning strategy. The biased tree shows a rapid drop in impurity, confirming it uses the color "shortcut." The unbiased tree shows a much more gradual decrease, as it must learn the more complex structural features.

---

## 📊 6. Results

The results clearly demonstrate that the Random Forest algorithm is highly susceptible to the spurious correlations present in the dataset.

### Accuracy Comparison

| Model / Input Data | General Sample Accuracy | Biased Aligned Accuracy | Biased Conflicting Accuracy |
| :--- | :--- | :--- | :--- |
| **Images Model (CMNIST)** | 41% | **100%** | 34% |
| **Features (MLP on CMNIST)** | 30% | 93% | 23% |
| **Features (ResNet50)** | 33% | **100%** | 26% |
| **Balanced Model** | **84%** | 65% | **86%** |
| **Greyscale Model (Unbiased)** | **97%** | - | - |

### Confusion Matrices (Images Model)

The confusion matrices for the model trained on raw CMNIST images starkly illustrate the problem. The model is perfect on bias-aligned data but performs poorly on bias-conflicting data, where its predictions are scattered.

---

## 🏁 7. Conclusion

* **The Random Forest model is extremely vulnerable to the bias present in the training data.** It learns to use color as a shortcut, ignoring the actual shape of the digits.
* **Using features from a pre-trained neural network can worsen the problem** if that network is also biased, as it reinforces the spurious correlation.
* **Eliminating the bias from the data (by balancing it or removing the spurious feature) restores high performance**, confirming that the issue lies with the data, not the model's fundamental capability.
* These results highlight the critical need to implement bias prevention and mitigation methods to build robust, reliable, and fair machine learning models.

---

## 🚀 8. How to Run

### Prerequisites

* Python 3.x
* A virtual environment (recommended)

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### `requirements.txt`

```
numpy
opencv-python-headless
matplotlib
seaborn
scikit-learn
prettytable
```

### Execution

The project is structured as a series of Python scripts or a Jupyter Notebook.

1.  Ensure the `cmnist/`, `features/`, and `features_RN/` datasets are placed in the correct directory structure as referenced in the code.
2.  Run the main script or notebook cells sequentially to perform the experiments.

```bash
# Example for running a python script
python main_analysis.py
```

---

## 📚 9. Bibliography

* Breiman, L. (2001). Random Forests. *Machine Learning, 45*, 5-32.
* He, Z., Zhang, S., Wang, B., et al. (2023). Spurious Correlations in Machine Learning: A Survey.
* Altmann, A., Toloşi, L., Sander, O., Lengauer, T. (2010). Permutation importance: a corrected feature importance measure. *Bioinformatics, 26*(10), 1340-1347.
