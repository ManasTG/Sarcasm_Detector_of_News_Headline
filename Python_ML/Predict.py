#%% Pathing - so that it can run on any device
import os

# This gets the folder where the current file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to that
MODEL_PATH = os.path.join(BASE_DIR, "nb_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

#%%
import pickle

# Load saved model and vectorizer
with open(MODEL_PATH, "rb") as f:
    loaded_model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    loaded_vectorizer = pickle.load(f)

def predict_sarcasm(headline):
    cleaned = headline.lower()  # Converts input into lowecase
    vec = loaded_vectorizer.transform([cleaned])    # Converts input to numbers
    result = loaded_model.predict(vec)[0]   # asks model for yes or no
    confidence = loaded_model.predict_proba(vec).max()  # which ever percentage is higher, the models shows that.
    
    if result == 1:
        print(f"SARCASTIC! ({confidence*100:.1f}% confidence)")
    else:
        print(f"NOT SARCASTIC ({confidence*100:.1f}% confidence)")

# Interactive loop
while True:
    headline = input("\nEnter a news headline (or type 'quit' to stop): ")
    if headline.lower() == 'quit':
        print("EXIT")
        break
    predict_sarcasm(headline)