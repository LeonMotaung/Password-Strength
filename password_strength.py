import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

data = pd.read_csv("data.csv", on_bad_lines='skip')
print("Loaded Data Preview:\n", data.head())

data = data.dropna()
data["strength"] = data["strength"].map({0: "Weak", 1: "Medium", 2: "Strong"})

def word(password):
    character=[]
    for i in password:
        character.append(i)
    return character
  
x = np.array(data["password"])
y = np.array(data["strength"])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.05, random_state=42)

# Create a Machine Learning Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=word, token_pattern=None)),
    ('clf', RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=42))
])

print("Training the machine learning pipeline... Please wait, this may take a minute!")
pipeline.fit(xtrain, ytrain)
print("Training complete!")
print("Model Accuracy:", pipeline.score(xtest, ytest))

# Save the trained pipeline!
print("Saving pipeline to disk as 'password_model.pkl'...")
joblib.dump(pipeline, 'password_model.pkl')
print("Saved! You can now use it in your API.")
