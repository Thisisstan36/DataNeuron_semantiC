import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

def load_data(file_path):
    data = pd.read_csv(file_path)  
    return data[['text1', 'text2']]  


def split_data(data, test_size=0.2):
    return data, None, None, None

def build_model(input_shape):
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)

    lstm = LSTM(64, return_sequences=True)
    encoded1 = lstm(input1)
    encoded2 = lstm(input2)

    attention = Attention()([encoded1, encoded2])
    concatenated = Concatenate(axis=-1)([encoded1, attention, encoded2])
    flattened = Flatten()(concatenated)
    output = Dense(1, activation='sigmoid')(flattened)

    model = Model(inputs=[input1, input2], outputs=output)
    return model


def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=10):
    pass

def evaluate_model(model, X_test, y_test):
    similarities = []
    for i in range(len(X_test)):
        similarity = cosine_similarity([X_test.iloc[i]['text1']], [X_test.iloc[i]['text2']])[0][0]
        similarities.append(similarity)
    return similarities

def save_model(model, filepath):
    model.save(filepath)
    print("Model saved successfully.")

def main():
    data = load_data('D:\DataNeuron_DataScience_Task1\DataNeuron_DataScience_Task1\DataNeuron_Text_Similarity.csv')

    X_test, _, _, _ = split_data(preprocessed_data)

    input_shape = (None,) 
    model = build_model(input_shape)

    similarities = evaluate_model(model, X_test, None)
    print("Similarities:", similarities)

    save_model(model, 'siamese_attention_model.h5')  # Save the model to a file

if __name__ == "__main__":
    main()






