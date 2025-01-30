from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import sentencepiece as spm
import uvicorn

app = FastAPI()

# Load models (should be in models/ directory)
sp = spm.SentencePieceProcessor()
sp.load("./models/spiece.model")

# ========== API 1: Encoding ==========
class EncodeRequest(BaseModel):
    question: str

class EncodeResponse(BaseModel):
    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    decoder_input_ids: List[List[int]]

@app.post("/encode", response_model=EncodeResponse)
async def encode_input(request: EncodeRequest):
    """Convert natural language question to model inputs"""
    prompt = f"Please answer this biology question: {request.question}"
    
    # Tokenization
    max_length = 64
    input_ids = sp.encode(prompt, add_eos=True, out_type=int)
    
    # Process input_ids
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
    else:
        input_ids += [sp.pad_id()] * (max_length - len(input_ids))
    
    # Create attention mask
    attention_mask = [1 if token != sp.pad_id() else 0 for token in input_ids]
    
    # Initial decoder input
    decoder_input_ids = [[sp.pad_id()]]  # Initial bos/pad token
    
    return {
        "input_ids": [input_ids],
        "attention_mask": [attention_mask],
        "decoder_input_ids": decoder_input_ids
    }

# ========== API 3: Decoding ==========
class DecodeRequest(BaseModel):
    decoder_input_ids: List[List[int]]

class DecodeResponse(BaseModel):
    answer: str

@app.post("/decode", response_model=DecodeResponse)
async def decode_output(request: DecodeRequest):
    """Convert model outputs to natural language"""
    decoder_input_ids = np.array(request.decoder_input_ids, dtype=np.int64)
    output_tokens = decoder_input_ids[0].tolist()
    
    # Remove special tokens
    filtered_tokens = [t for t in output_tokens if t not in {sp.pad_id(), sp.eos_id()}]
    
    return {"answer": sp.decode(filtered_tokens)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)