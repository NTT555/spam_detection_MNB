import os
import sys
import pickle
import pandas as pd

# --- CRITICAL: ENVIRONMENT PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# --- CRITICAL: PICKLE REFERENCE MAPPING ---
import train
import __main__
__main__.MultinomialNaiveBayes = train.MultinomialNaiveBayes
# ------------------------------------------------

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mnb_model.pkl')

def load_spam_model():
    """Loads the pre-trained model artifact from a .pkl file."""
    print("[INFO] Loading system model...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            saved_objects = pickle.load(f)
            return saved_objects['model']
    except FileNotFoundError:
        print("[-] ERROR: Model artifact not found. Please execute 'python src/train.py' first.")
        return None

def predict_unseen_data(model, email_text: str):
    """
    Performs inference on unseen text data and outputs the prediction.
    """
    X_unseen = pd.Series([email_text])
    prediction = model.predict(X_unseen)[0]
    
    pred_str = str(prediction)
    result_label = "SPAM 🚫" if pred_str == '1' else "HAM ✅"
    
    print("-" * 60)
    display_text = email_text[:100] + "..." if len(email_text) > 100 else email_text
    print(f"CONTENT  : {display_text}")
    print(f"RESULT   : {result_label}")
    print("-" * 60)

if __name__ == "__main__":
    mnb_model = load_spam_model()
    
    if mnb_model:
        print("\n=== UNSEEN DATA INFERENCE SYSTEM ===")
        
        unseen_spam = (
            "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! "
            "Txt the word: CLAIM to No: 81010 T&C www.dbuk.net. Get your cheap viagra pills now."
        )
        
        unseen_ham = (
            "Hi team, attached is the revised project schedule for the Multinomial Naive Bayes report. "
            "Let me know if you have any questions before our meeting tomorrow at 9 AM. Best regards."
        )
        
        unseen_short_text = (
            "Hello, we have a group meeting scheduled for tomorrow at 8 AM."
        )

        predict_unseen_data(mnb_model, unseen_spam)
        predict_unseen_data(mnb_model, unseen_ham)
        predict_unseen_data(mnb_model, unseen_short_text)