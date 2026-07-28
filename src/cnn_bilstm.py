import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


MAX_WORDS = 50000
MAX_SEQUENCE_LENGTH = 128
EMBEDDING_DIM = 128


def create_tokenizer():
    return Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>",
    )


def prepare_sequences(tokenizer, texts):
    sequences = tokenizer.texts_to_sequences(texts)

    padded_sequences = pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    return padded_sequences


def build_cnn_bilstm_model():

    model = Sequential()

    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=EMBEDDING_DIM,
        )
    )

    model.add(
        Conv1D(
            filters=128,
            kernel_size=5,
            activation="relu",
        )
    )

    model.add(
        MaxPooling1D(
            pool_size=2
        )
    )

    model.add(
        Bidirectional(
            LSTM(
                128,
                dropout=0.2,
                recurrent_dropout=0.2,
            )
        )
    )

    model.add(
        Dense(
            64,
            activation="relu",
        )
    )

    model.add(
        Dropout(0.5)
    )

    model.add(
        Dense(
            3,
            activation="softmax",
        )
    )

    model.build(input_shape=(None, MAX_SEQUENCE_LENGTH))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model