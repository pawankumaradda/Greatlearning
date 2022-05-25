#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 18:03:48 2022

@author: padda
"""


import unicodedata
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import unicodedata
from nltk.corpus import stopwords
import string

model = load_model('model_lstm.h5')


# clean the input and convert them all to lower case.
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def text_cleaning(text):
    # Initialize the object for Lemmatizer class
    lemmatizer = WordNetLemmatizer()
    # Set the stopwords to English
    stopwords = nltk.corpus.stopwords.words('english')
    # Normalize the text in order deal with accented words and unicodes
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore').lower())
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    #word tokenization
    words = nltk.word_tokenize(text.translate(remove_punct_dict))
    # Consider the words which are not in stopwords of english and lemmatize them
    lems = [lemmatizer.lemmatize(i) for i in words if i not in stopwords]
    return lems

def chatbot_response(sentence):
    data = " ".join(text_cleaning(sentence))
    #vectorizing the tweet by the pre-fitted tokenizer instance
    tokenizer = Tokenizer(num_words=5000)
    data = tokenizer.texts_to_sequences(data)
    #padding the tweet to have exactly the same shape as `embedding_2` input
    data = pad_sequences(data, maxlen=100, dtype='int32', value=0)
    ypred = model.predict(data,batch_size=1,verbose = 2)[0]
    class_label = np.argmax(ypred)
    output=''
    if class_label == 0:
        output = 'AccidentLevel I'
    elif class_label == 1:
        output = 'AccidentLevel II'
    else:
        output = 'AccidentLevel III'
    return output


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    result = chatbot_response(userText)
    return result
#chatbot_response(userText)

if __name__ == "__main__":
    app.run()
