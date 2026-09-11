import tensorflow as tf
from tensorflow.keras.layers import Embedding, Flatten, Input, Multiply, Dense, Concatenate
from textblob import TextBlob

def build_ncf(num_users, num_movies, embedding_dim=32):
    # Inputs
    user_input = Input(shape=(1,), name='user_input')
    movie_input = Input(shape=(1,), name='movie_input')

    # Embeddings
    user_emb_gmf = Embedding(num_users, embedding_dim)(user_input)
    movie_emb_gmf = Embedding(num_movies, embedding_dim)(movie_input)

    # GMF branch
    gmf = Multiply()([Flatten()(user_emb_gmf), Flatten()(movie_emb_gmf)])

    # MLP branch
    user_emb_mlp = Embedding(num_users, embedding_dim)(user_input)
    movie_emb_mlp = Embedding(num_movies, embedding_dim)(movie_input)
    mlp = Concatenate()([Flatten()(user_emb_mlp), Flatten()(movie_emb_mlp)])
    mlp = Dense(64, activation='relu')(mlp)
    mlp = Dense(32, activation='relu')(mlp)

    # Fusion
    concat = Concatenate()([gmf, mlp])
    output = Dense(1, activation='sigmoid')(concat)

    model = tf.keras.Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def get_sentiment_score(text):
    # Returns polarity between -1 and 1, normalized to 0-1
    analysis = TextBlob(text)
    return (analysis.sentiment.polarity + 1) / 2

def calculate_hybrid_score(ncf_score, sentiment_score, alpha=0.8):
    return (alpha * ncf_score) + ((1 - alpha) * sentiment_score)
