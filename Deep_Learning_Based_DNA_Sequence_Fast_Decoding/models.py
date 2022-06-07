import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose, Lambda, \
    BatchNormalization, Bidirectional, LSTM, Dropout, Dense, InputLayer, Conv2D, MaxPooling2D, Flatten,\
    AveragePooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D, AveragePooling1D, MultiHeadAttention,\
    LayerNormalization, Embedding, LeakyReLU, Conv1DTranspose


def cnn_model(max_len, vocab_size):
    model = Sequential([
        InputLayer(input_shape=(max_len, vocab_size)),
        Conv1D(32, 17, padding='same', activation='relu'),
        Conv1D(64, 11, padding='same', activation='relu'),
        Conv1D(128, 5, padding='same', activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    return model

## Step 1: Implement your own model below

## Step 2: Add your model name and model initialisation in the model dictionary below

def return_model(model_name, max_len, vocab_size):
    model_dic={'cnn': cnn_model(max_len, vocab_size)}
    return model_dic[model_name]



