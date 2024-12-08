from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

# Initialize FastAPI app
app = FastAPI()

# Load the fine-tuned BLIP model and processor
model_path = "./blip_finetuned"
model = BlipForConditionalGeneration.from_pretrained(model_path)
processor = BlipProcessor.from_pretrained(model_path)

# Ensure the model is on the correct device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


@app.post("/generate-caption/")
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
