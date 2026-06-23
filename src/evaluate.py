import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple

import __main__
from train import MultinomialNaiveBayes
__main__.MultinomialNaiveBayes = MultinomialNaiveBayes

def calculate_metrics(y_true: List[str], y_pred: List[str], positive_label: str = 'spam') -> Tuple[float, float, float, float]:
    """
    Computes classification metrics from scratch.
    
    Returns:
        Tuple containing Accuracy, Precision, Recall, and F1-Score.
    """
    tp = fp = tn = fn = 0
    
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == positive_label and pred_label == positive_label:
            tp += 1
        elif true_label != positive_label and pred_label == positive_label:
            fp += 1
        elif true_label != positive_label and pred_label != positive_label:
            tn += 1
        elif true_label == positive_label and pred_label != positive_label:
            fn += 1

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return accuracy, precision, recall, f1_score

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mnb_model.pkl')

    print("[INFO] Loading Model Artifact...")
    with open(MODEL_PATH, 'rb') as f:
        saved_objects = pickle.load(f)
        
    model = saved_objects['model']
    test_df = saved_objects['test_data']
    
    X_test, y_test = test_df['text'], test_df['label']
    
    print(f"[INFO] Initiating Inference Phase on {len(X_test)} hold-out samples...")
    y_pred = model.predict(X_test)
    
    # Determine the positive label dynamically (assuming binary classification: spam/ham)
    # We set 'spam' as the positive class if it exists, otherwise use the first class.
    classes = np.unique(y_test)
    pos_label = 1 if 1 in classes else '1'
    
    print("[INFO] Computing Evaluation Metrics...")
    acc, prec, rec, f1 = calculate_metrics(y_test.tolist(), y_pred, positive_label=pos_label)
    
    print("\n=========================================")
    print("      MODEL EVALUATION REPORT            ")
    print("=========================================")
    print(f" Positive Class : '{pos_label}'")
    print(f" Accuracy       : {acc:.4f}")
    print(f" Precision      : {prec:.4f}")
    print(f" Recall         : {rec:.4f}")
    print(f" F1-Score       : {f1:.4f}")
    print("=========================================")