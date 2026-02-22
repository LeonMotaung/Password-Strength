from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn
import warnings

warnings.filterwarnings('ignore')

app = FastAPI(title="Password Strength API", 
              description="An API that checks password strength based on a Machine Learning model.")

# We must define the custom word tokenizer here so joblib knows how to load it!
def word(password):
    character=[]
    for i in password:
        character.append(i)
    return character

# Load the trained pipeline that we saved from our ML code
try:
    model = joblib.load('password_model.pkl')
except Exception as e:
    model = None
    print("Warning: Model not found. Did you run password_strength.py first?")

# Define the structure for JSON requests
class PasswordRequest(BaseModel):
    password: str

@app.post("/predict")
def predict_strength(request: PasswordRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is currently unavailable. Ensure the model has been trained and saved.")
    
    password_text = request.password
    if not password_text:
        raise HTTPException(status_code=400, detail="Password cannot be empty")
        
    # The pipeline automatically vectorizes the text and then predicts!
    prediction = model.predict([password_text])
    
    return {
        "password": password_text, 
        "predicted_strength": prediction[0]
    }

if __name__ == "__main__":
    print("Starting API Server. You can access the automatic docs at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
