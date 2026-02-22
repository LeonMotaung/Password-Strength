import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import getpass

data = pd.read_csv("data.csv", on_bad_lines='skip')
print(data.head())

data = data.dropna()
data["strength"] = data["strength"].map({0: "Weak", 
                                         1: "Medium",
                                         2: "Strong"})
print(data.sample(5))

def word(password):
    character=[]
    for i in password:
        character.append(i)
    return character
  
x = np.array(data["password"])
y = np.array(data["strength"])

tdif = TfidfVectorizer(tokenizer=word, token_pattern=None)
x = tdif.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.05, 
                                                random_state=42)

print("Training the machine learning model... Please wait, this may take a minute!")
model = RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=42)
model.fit(xtrain, ytrain)
print("Training complete!")
print("Model Accuracy:", model.score(xtest, ytest))

user = getpass.getpass("Enter Password to test: ")
user_data = tdif.transform([user]).toarray()
output = model.predict(user_data)
print("Predicted Strength:", output[0])
