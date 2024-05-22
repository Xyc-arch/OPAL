import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from meta_data import *

def generate_causal_dataset_rounded(n_samples):
    # Set seed for reproducibility
    np.random.seed(42)

    # Generate independent features, rounded to one decimal place
    X1 = np.round(np.random.randn(n_samples), 1)
    X2 = np.round(np.random.randn(n_samples), 1)
    
    # X3 is causally dependent on X1 and X2
    X3 = np.round(0.5 * X1 + 0.3 * X2 + np.random.normal(0, 0.5, n_samples), 1)

    # Interaction and other features
    X4 = np.round(np.random.randn(n_samples) * X1, 1)  # Interaction between X1 and X4
    X5 = np.round(np.random.randn(n_samples) + 0.5 * X3, 1)  # X5 is influenced by X3
    X6 = np.round(np.random.randn(n_samples), 1)
    X7 = np.round(np.random.randn(n_samples), 1)
    X8 = np.round(X2 * X3, 1)  # Direct interaction between X2 and X3
    X9 = np.round(X1 * X2, 1)  # Additional interaction between X1 and X2

    # Generate binary target with less noise
    noise = np.random.normal(0, 0.5, n_samples)  # Reduced noise variance for more evident causal effect
    Y = (1.5 + 0.7 * X1 - 0.6 * X2 + 0.8 * X3 + 0.4 * X9 + noise) > np.median(1.5 + 0.7 * X1 - 0.6 * X2 + 0.8 * X3 + 0.4 * X9 + noise)
    Y = Y.astype(int)

    # Create DataFrame
    data = pd.DataFrame({
        'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'X6': X6, 'X7': X7, 'X8': X8, 'X9': X9, 'Target': Y
    })

    return data


if __name__ == "__main__":
    data = generate_causal_dataset_rounded(1000)
    dataset_name = "craft"
    raw_size = 100
    current_post = post_imbalanced_focus
    
    parent_path = "../data/{}/{}_see_small_raw{}{}".format(dataset_name, dataset_name, raw_size, current_post)
    
    print(data.head())
    
    data.to_csv(parent_path + "/craft.csv")