import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Concatenate, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load the saved model
model = load_model('siamese_attention_model.h5')

# Define FastAPI app
app = FastAPI()

# Define request and response models
class TextPair(BaseModel):
    sentence1: str
    sentence2: str

class SimilarityResponse(BaseModel):
    similarity_score: float

# Tokenization and padding for model input
tokenizer = Tokenizer()
tokenizer.fit_on_texts([sentence1, sentence2])
vocab_size = len(tokenizer.word_index) + 1

# Max sequence length for padding
max_seq_length = max(max(tokenizer.texts_to_sequences([sentence1])), max(tokenizer.texts_to_sequences([sentence2])))

# Function to preprocess text
def preprocess_text(text):
    # Implement your preprocessing logic here
    return text

# Function to compute similarity score using the loaded model
def compute_similarity(text_pair):
    sequence1 = tokenizer.texts_to_sequences([text_pair.sentence1])
    sequence2 = tokenizer.texts_to_sequences([text_pair.sentence2])

    padded_sequence1 = pad_sequences(sequence1, maxlen=max_seq_length)
    padded_sequence2 = pad_sequences(sequence2, maxlen=max_seq_length)

    similarity_score = model.predict([padded_sequence1, padded_sequence2])[0][0]
    return similarity_score

# API endpoint to compute similarity score
@app.post("/similarity/")
async def get_similarity(text_pair: TextPair):
    similarity_score = compute_similarity(text_pair)
    return SimilarityResponse(similarity_score=similarity_score)
