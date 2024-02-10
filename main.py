import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import pickle

import yaml
with open("config.YAML", 'r') as stream:
    config = yaml.safe_load(stream)

model = load_model(config['model_path'])

with open(config['tokenizer_path'], 'rb') as file:
    tokenizer = pickle.load(file)

app = FastAPI()

class TextPair(BaseModel):
    sentence1: str
    sentence2: str

class SimilarityResponse(BaseModel):
    similarity_score: float

def compute_similarity(text_pair):
    sentence1 = preprocess_text(text_pair.sentence1)
    sentence2 = preprocess_text(text_pair.sentence2)

    sequence1 = tokenizer.texts_to_sequences([sentence1])
    sequence2 = tokenizer.texts_to_sequences([sentence2])
    padded_sequence1 = pad_sequences(sequence1, maxlen=config['max_seq_length'])
    padded_sequence2 = pad_sequences(sequence2, maxlen=config['max_seq_length'])

    similarity_score = model.predict([padded_sequence1, padded_sequence2])[0][0]
    return similarity_score

@app.post("/similarity/")
async def get_similarity(text_pair: TextPair):
    similarity_score = compute_similarity(text_pair)
    return SimilarityResponse(similarity_score=similarity_score)

if __name__=='__main__':
    app.run(host="0.0.0.0")