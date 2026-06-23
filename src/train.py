import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any

from preprocessing import TextPreprocessor
from data_loader import load_and_split_data
from collections import defaultdict
from typing import List, Dict, Any

# --- CRITICAL: SIBLING MODULE PATH RESOLUTION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from preprocessing import TextPreprocessor
from data_loader import load_and_split_data


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes Classifier for text classification.
    Implements MAP (Maximum A Posteriori) estimation using Log-Probabilities 
    and Laplace (Additive) Smoothing to prevent zero-frequency problems.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha (float): Laplace smoothing parameter. 
                           alpha=1.0 represents standard Laplace smoothing.
        """
        self.alpha = alpha
        self.classes: np.ndarray = None
        self.vocab: set = set()
        
        # P(c): Prior probability of each class
        self.log_priors: Dict[Any, float] = {}
        
        # P(w|c): Likelihood of a word given a class
        self.log_likelihoods: Dict[Any, Dict[str, float]] = {}

    def fit(self, X: pd.Series, y: pd.Series) -> None:
        """
        Calculates vocabulary, log-priors, and log-likelihoods from training data.
        """
        self.classes = np.unique(y)
        total_documents = len(y)
        
        class_doc_counts = defaultdict(int)
        word_counts_per_class = defaultdict(lambda: defaultdict(int))
        total_words_per_class = defaultdict(int)

        # Iterate through corpus to build frequency distributions
        for text, label in zip(X, y):
            class_doc_counts[label] += 1
            tokens = TextPreprocessor.tokenize(text)
            
            for token in tokens:
                self.vocab.add(token)
                word_counts_per_class[label][token] += 1
                total_words_per_class[label] += 1

        vocab_size = len(self.vocab)

        # Compute log probabilities (Training phase)
        for c in self.classes:
            # log(P(c))
            self.log_priors[c] = np.log(class_doc_counts[c] / total_documents)
            
            # Denominator for likelihood: N_c + (alpha * |V|)
            denominator = total_words_per_class[c] + (self.alpha * vocab_size)

            self.log_likelihoods[c] = {}
            
            # Compute log(P(w|c)) for every word in vocabulary
            for word in self.vocab:
                numerator = word_counts_per_class[c][word] + self.alpha
                self.log_likelihoods[c][word] = np.log(numerator / denominator)

    def predict(self, X: pd.Series) -> List[Any]:
        """
        Assigns labels to new documents based on trained log-probabilities.
        """
        predictions = []
        
        for text in X:
            tokens = TextPreprocessor.tokenize(text)
            
            # Initialize scores with log-priors
            class_scores = {c: self.log_priors[c] for c in self.classes}
            
            for c in self.classes:
                for token in tokens:
                    # Ignore Out-Of-Vocabulary (OOV) tokens during inference
                    if token in self.vocab:
                        class_scores[c] += self.log_likelihoods[c][token]
            
            # Select class with maximum posterior probability
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
            
        return predictions

if __name__ == "__main__":
    # Define file paths based on project structure
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'email_data.csv')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'mnb_model.pkl')

    print("[INFO] Initiating Data Ingestion Phase...")
    train_df, test_df = load_and_split_data(DATA_PATH)
    
    X_train, y_train = train_df['text'], train_df['label']
    
    print(f"[INFO] Training corpus size: {len(X_train)} documents.")
    print("[INFO] Fitting Multinomial Naive Bayes Model...")
    
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(X_train, y_train)
    
    print(f"[INFO] Model fitted successfully. Extracted vocabulary size: {len(model.vocab)}")
    
    # Serialize model to disk
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'test_data': test_df}, f)
        
    print(f"[INFO] Model artifact saved to: {MODEL_PATH}")