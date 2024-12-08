from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import os
import joblib

# Load the best classification model and objects:
best_model = joblib.load("random_forest_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Initialize FastAPI app
app = FastAPI()

# Load the fine-tuned BLIP model and processor
model_path = "./blip_finetuned"
model = BlipForConditionalGeneration.from_pretrained(model_path)
processor = BlipProcessor.from_pretrained(model_path)

# Ensure the model is on the correct device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load Flan-t5 model and tokenizer
model2 = T5ForConditionalGeneration.from_pretrained('./checkpoint-2000-forServer')
tokenizer = T5Tokenizer.from_pretrained('./checkpoint-2000-forServer')

# Define a Pydantic model for request validation
class PromptRequest(BaseModel):
    prompt: str

# Define a request model for classification model api
class Query(BaseModel):
    question: str    

@app.post("/generate")
async def generate_text(request: PromptRequest):
    # Access the 'prompt' from the request body
    prompt = request.prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    output_sequences = model2.generate(input_ids=inputs["input_ids"], max_length=100)
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

@app.post("/generate-caption")
async def generate_caption(file: UploadFile = File(...)):
    """
    Endpoint to generate captions for an uploaded image.
    """
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Open and preprocess the image
        image = Image.open(temp_file_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Generate caption
        model.eval()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=50, num_beams=5)
            caption = processor.decode(generated_ids[0], skip_special_tokens=True)

        # Delete the temporary file
        os.remove(temp_file_path)

        return JSONResponse({"caption": caption})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)    


# Define a prediction endpoint for classification api
@app.post("/predict")
def predict_category(query: Query):
    # Preprocess the input
    user_input_tfidf = tfidf_vectorizer.transform([query.question])
    # Predict the category
    predicted_category_encoded = best_model.predict(user_input_tfidf)
    # Decode the predicted category
    predicted_category = label_encoder.inverse_transform(predicted_category_encoded)
    return {"predicted_category": predicted_category[0]}        

#pip install fastapi uvicorn transformers torch
#run with : uvicorn app:app --host 0.0.0.0 --port 8000

#Then build and run the Docker container:
#docker build -t flan-t5-api .
#docker run -p 8000:8000 flan-t5-api