# Sentiment-Analysis-in-Text-using-LSTM-Neural-Network

Introduction:
In this project, we aim to develop a model using Long Short-Term Memory (LSTM) neural network to classify the sentiment (positive or negative) associated with each review in the IMDB dataset. We will start by preparing the data, designing the model, training, and evaluating its performance.

Dataset:
The IMDB dataset, commonly known as the IMDB review dataset, comprises 25,000 positive and negative reviews for training, along with a similar number for evaluation. The task is to design a model that can accurately determine the positive or negative sentiment expressed in each review.

Data Preparation and Analysis:
Before diving into the model design, it's important to preprocess and analyze the data. This involves converting all text to lowercase, removing punctuation and special characters, and tokenizing the text into separate units. We will use whitespace and punctuation marks as delimiters between words.

Word Embeddings:
To convert text data into a format suitable for neural networks, we will use pre-trained word embeddings. Specifically, we will utilize BERT (Bidirectional Encoder Representations from Transformers) embeddings. You can obtain the BERT embeddings from the following link:
https://github.com/google-research/bert

Model Design and Training:
We will build an LSTM neural network for sentiment analysis. The model will take a review as input and produce an output indicating agreement or disagreement with the sentiment expressed. The LSTM algorithm will be implemented in a flexible manner, allowing for adjustments to the number of hidden layers, the number of neurons in each layer, activation functions, batch size, and more.

Single-directional LSTM:
We will begin by creating a single-directional LSTM with the following specifications:

Batch size: 50
One hidden layer with 64 memory units
Sigmoid activation function
Learning rate: 0.01
We will train the model on the training data and plot a graph showing the change in Mean Squared Error (MSE) on the training dataset. The accuracy of the trained model on the test data will also be reported.

Bi-directional LSTM:
Next, we will modify the LSTM model to be bi-directional. We will compare the results of the bi-directional model with the single-directional one.

Model Improvement with Layer Variation:
To enhance the model's performance, we will experiment with different configurations. We will increase the number of hidden layers to two and vary the number of memory units in each layer through multiple trials. This experimentation will help us identify a better architecture for sentiment analysis.
