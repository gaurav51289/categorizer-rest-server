import os
import csv
import re
import string
import nltk
import json
import numpy as np
import pickle
import operator

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

cwd = os.getcwd()

bow_path = os.path.join(cwd, 'categorizer', 'bagOfWords.json')
Gmodel_path = os.path.join(cwd, 'categorizer', 'categorize_model.h5')
categories_csv_path = os.path.join(cwd, 'categorizer', 'categories.csv')

f = open(bow_path, 'r')
bag = json.loads(f.read())

with open(categories_csv_path, 'r') as f:
    reader = csv.reader(f)
    categories_list = list(reader)

def getCategoriesFromCSV(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        return list(reader)[0]


def getCategories(question):
    test_question = [question]
    sequences = []

    with open('tokenizer.pkl', 'rb') as input:
        tokenizer = pickle.load(input)
        sequences = tokenizer.texts_to_sequences(test_question)

    MAX_SEQUENCE_LENGTH = 200
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.3


    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    model  = load_model("categorize_model_with_embedding.h5")


    pred = model.predict(data)
    categories_list = getCategoriesFromCSV(categories_csv_path)

    cat_dict = dict(zip(categories_list, pred[0]))
    que_cats = []
    sorted_cat=sorted(cat_dict.items(), key=operator.itemgetter(1), reverse=True)

    que_cats=[k[0] for k in sorted_cat[:5]]
    print("Categories found: ")
    print(que_cats)
    return que_cats

