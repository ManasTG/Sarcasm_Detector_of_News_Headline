#%% Pathing - so that it can run on any device
import os

# This gets the folder where the current file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to that
MODEL_PATH = os.path.join(BASE_DIR, "nb_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "..", "Dataset", "Sarcasm_Headlines_Dataset_v2.json")

#%% Load Data

import pandas as pd

# Inputing the dataset
df = pd.read_json(DATASET_PATH, lines = True)


print("Shape:", df.shape)   # Tells the size
print("\nFirst 5 rows:")    
print(df.head())    # preview of the data {5 rows in default}
print("\nColumn names:", df.columns.tolist())   #extracts columns header
print("\nSarcastic vs Not:")    
print(df['is_sarcastic'].value_counts())    # tells numbers of time unique values appears(here 0s and 1s from 'is_sarcastic')
print("\nAny missing values?")
print(df.isnull().sum())    # is null converts null data to true and contains data to false, and sum() converts true/false to 1/0

#%% Clean Text 

import re


df = df [['headline', 'is_sarcastic']]

def clean_text(text):
    text = text.lower() # makes everything lowercase
    text = re.sub(r'[^a-z\s]', '', text)   # remove punctuation & numbers
    text = text.strip() # remove extra space
    return text

df['clean_headline'] = df['headline'].apply(clean_text)

# Preview
print('Before cleaning:')
print(df['headline'][0])
print('\nAfter cleaning')
print(df['clean_headline'][0])
print("\nCleaned data preview:")
print(df.head())
    

#%% TF-IDF Vectorization

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Seperate input and output
x = df['clean_headline']    # input 'cleaned headlines'
y = df['is_sarcastic']  # output '0s or 1s'


# Split the data into 80% train and 20% test
x_train, x_test, y_train, y_test = train_test_split(
    x, y , test_size = 0.2, random_state = 42   
)   # random_state - shuffles the rows so that model wont get same saracastic or non sarcastic every time leading to bias, 42 is the seed just which makes the shuffles the exact same way; test_size = 0.2 divides the data(80-20) 
 

# Convert words to numbers using TF-IDF

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))  # keeps top 5000 most frequent words.. saves from confusion; ngram captures context not only words
x_train_vec = vectorizer.fit_transform(x_train) # fit- takes the 5000 words from the dictionary; transform changes those words to numbers
x_test_vec = vectorizer.transform(x_test)   # numerical version of the headlines, ready to handed over to model

# For confirmation

print("Training samples:", x_train_vec.shape[0])
print("Testing samples:", x_test_vec.shape[0])
print("Features (unique words):", x_train_vec.shape[1])


#%% Training of Models

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Define models
nb_model = MultinomialNB()
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')   # (maximum number of times the models is given to improve itself{default = 100}, we have a bit more non sarcastic comments so it ensures balance and not bias to one another)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')   # (built 100 decision trees and more majority vote. More trees= more accurate but slow, randomness of the input, to insure balance)

# Train all 3
nb_model.fit(x_train_vec, y_train)
print("Naive Bayes trained!")

lr_model.fit(x_train_vec, y_train)
print("Logistic Regression trained!")

rf_model.fit(x_train_vec, y_train)
print("Random Forest trained!")


#%% Evaluate Models

from sklearn.metrics import accuracy_score, classification_report

models = {
    "Naive Bayes": nb_model,
    "Logistic Regression": lr_model,
    "Random Forest": rf_model
    }

for name, model in models.items():
    y_pred = model.predict(x_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*45}")
    print(f"{name}")
    print(f"{'='*45}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(classification_report(y_test, y_pred, 
          target_names=['Not Sarcastic', 'Sarcastic']))
    
    

#%% Save the Model
import pickle
import os

# Save in the project folder

with open(MODEL_PATH, "wb") as f:
    pickle.dump(nb_model, f)

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved!")
print("Vectorizer saved!")
