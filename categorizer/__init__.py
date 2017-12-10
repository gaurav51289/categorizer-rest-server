import os
import csv
import re
import string
import nltk
import json
import numpy as np
import pickle
import operator


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from html.entities import name2codepoint
from bs4 import BeautifulSoup
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


cwd = os.getcwd()

bow_path = os.path.join(cwd, 'categorizer', 'bagOfWords.json')
model_path = os.path.join(cwd, 'categorizer', 'categorize_model.h5')
categories_csv_path = os.path.join(cwd, 'categorizer', 'categories.csv')

regex = re.compile('[%s]' % re.escape(string.punctuation))
wordnet = WordNetLemmatizer()

f = open(bow_path, 'r')
bag = json.loads(f.read())

with open(categories_csv_path, 'r') as f:
    reader = csv.reader(f)
    categories_list = list(reader)

def getCategoriesFromCSV(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        return list(reader)[0]

def unescape(text):
    def fixup(m):
        text = m.group(0)
        if text[:2] == "&#":
            # character reference
            try:
                if text[:3] == "&#x":
                    return chr(int(text[3:-1], 16))
                else:
                    return chr(int(text[2:-1]))
            except ValueError:
                pass
        else:
            # named entity
            try:
                text = chr(name2codepoint[text[1:-1]])
            except KeyError:
                pass
        return text  # leave as is

    return re.sub("&#?\w+;", fixup, text)


def cleanText(text):
    text = text.lower()
    soup = BeautifulSoup(unescape(text), "html.parser")
    text = soup.get_text()  # nltk.clean_html(unescape(text))

    tokens = word_tokenize(text)
    new_tokens = []
    for t in tokens:
        nt = regex.sub(u'', t)
        if not nt == u'' and nt not in stopwords.words('english'):
            new_tokens.append(wordnet.lemmatize(nt))

    return " ".join(new_tokens)

""""
def getCategories(question):
    

    userInput = cleanText(question)
    inputTok = userInput.split()[:160]
    inputTok = [bag[t] if t in bag else 0 for t in inputTok]
    inputTok = inputTok + ([0] * (160 - len(inputTok)))

    inputTok = np.asarray(inputTok, dtype=np.float32)
    final = np.array([inputTok])

    loaded_model = load_model(model_path)

    preds = loaded_model.predict(final)
    categories_list = getCategoriesFromCSV(categories_csv_path)

    cat_dict = dict(zip(categories_list, preds[0]))
    print('', end='')
    print('================================================================================')
    print('DICT ==> {CATEGORY : SCORE}')
    print('================================================================================')
    print(cat_dict)

    que_cats = []
    for cat in cat_dict:
        if cat_dict[cat] == 1.0:
            que_cats.append(cat)
    return que_cats


"""
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
    print(cat_dict)
    que_cats = []
    sorted_cat=sorted(cat_dict.items(), key=operator.itemgetter(1), reverse=True)

    que_cats=[k[0] for k in sorted_cat[:5]]
    print (que_cats)
    #for cat in cat_dict:
     #   if cat_dict[cat] >= 0.02 :
     #       que_cats.append(cat)
    return que_cats

