import pandas as pd
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import re


filepath = "NRC-emotion-lexicon.txt"
emolex_df = pd.read_csv(filepath,  names=["word", "emotion", "association"], skiprows=45, sep='\t', keep_default_na=False)
emolex_words = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()

emotions = ["Sadness", "Joy"]


df = pd.read_csv("Data.csv")
#df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df = df.applymap(lambda x: re.sub(r'\s+', ' ', x.lower()) if isinstance(x, str) else x)
tokensDf = pd.DataFrame(index=df.index, columns=df.columns)
arr = df.to_numpy()
labels = df.columns.to_list()

stemmer = PorterStemmer()


sadnessLexicon = []
joyLexicon = []




def createLexicons(sadness, joy):
    for i in emolex_words[emolex_words.sadness == 1].word:
        sadnessLexicon.append(i)
    for i in emolex_words[emolex_words.joy == 1].word:
        joyLexicon.append(i)

createLexicons(sadnessLexicon, joyLexicon)
sadnessLexiconStemmed = {stemmer.stem(word) for word in sadnessLexicon}
joyLexiconStemmed = {stemmer.stem(word) for word in joyLexicon}

sadnessLexiconStemmed = set(sadnessLexiconStemmed)
joyLexiconStemmed = set(joyLexiconStemmed)

X = []
Y = []


def test(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]

    x1 = sum(1 for word in stemmed_tokens if word in joyLexiconStemmed)
    x2 = sum(1 for word in stemmed_tokens if word in sadnessLexiconStemmed)
    x3 = len(tokens)

    return [x1, x2, x3]

def create_data_xi(text,shouldPrint):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]

    x1 = sum(1 for word in stemmed_tokens if word in joyLexiconStemmed)
    x2 = sum(1 for word in stemmed_tokens if word in sadnessLexiconStemmed)
    x3 = len(tokens)

    if(shouldPrint):
        print(stemmed_tokens)
        print(text)

    return [1, x1, x2, x3] # 1 is bias term


def create_data(X, Y, testing=-1):
    priors = {}

    for i in emotions:
        priors[i] = 0
    a, b = arr.shape

    a = 30

    emotionSentenceCounts = np.zeros(len(emotions))
    counter = 0

    for i in range(len(emotions)):
        for col in range(1,b,2):
            if(emotions[i].lower() in labels[col].lower()):
                for k in range(a):
                    text = arr[k, col]
                    if(isinstance(text, float)):
                        continue
                    shouldPrint = False
                    if(testing == counter):
                        shouldPrint = True 
                    x_i = create_data_xi(text, shouldPrint)    
                    X.append(x_i)
                    Y.append(i)
                    counter +=1

testValue = -1
create_data(X, Y, testValue)


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def calculate_item_loss(x, y_real, w):
    y_pred = sigmoid(w @ x)
    
    loss = - (y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))

    return loss, y_pred


def learnlogreg(X,Y):    
    eta = 10e-5
    m, n = X.shape
    w = np.zeros(n,)
    totalLoss = 0
    
    while(True):
        g = 0
        newLoss = 0
        for i in range(m):
            itemLoss, y_pred = calculate_item_loss(X[i], Y[i], w)
            print(itemLoss)
            
            j = np.dot(Y[i] * w.T, X[i])
            p = sigmoid(j)
            
            print(np.log(p))

            g_i = (1 - p) * Y[i] * X[i]
            g = (y_pred - Y[i]) * X[i]

            print(g_i, g_i)

            return


            w -= eta * g
        
        if(abs(newLoss - totalLoss) < 0.1):
            return w
        totalLoss = newLoss
        
    return w

X = np.array(X)
Y = np.array(Y)


learnlogreg(X,Y)
