import pandas as pd
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import re
import sys


filepath = "NRC-emotion-lexicon.txt"
emolex_df = pd.read_csv(filepath,  names=["word", "emotion", "association"], skiprows=45, sep='\t', keep_default_na=False)
emolex_words = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()

emotions = ["Sadness", "Joy"]
emotionToIndex = {
    "Sadness": 0,
    "Joy": 1
}


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

def create_data_xi(text,shouldPrint=False):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]

    x1 = sum(1 for word in stemmed_tokens if word in joyLexiconStemmed)
    x2 = sum(1 for word in stemmed_tokens if word in sadnessLexiconStemmed)
    x3 = len(tokens)

    if(shouldPrint):
        print(text)
        print(stemmed_tokens)
        print(x1, x2, x3)


    return [1, x1, x2, x3] # 1 is bias term

def create_data(X, Y, colstart, colend,testing=-1):
    priors = {}

    for i in emotions:
        priors[i] = 0
    a, b = arr.shape

    emotionSentenceCounts = np.zeros(len(emotions))
    counter = 0

    for i in range(len(emotions)):
        for col in range(1,b,2):
            if(emotions[i].lower() in labels[col].lower()):
                for k in range(colstart, colend):
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

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def calculate_item_loss(x, y_real, w):
    y_pred = sigmoid(w @ x)
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)

    try:
        loss = - 1 * (y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))
    except Exception as e:
        print(y_pred)

    return loss, y_pred


def learnlogreg(X,Y, eta=0.1):    
    m, n = X.shape
    w = np.zeros(n,)
    totalLoss = 0
    
    while(True):
        g = 0
        newLoss = 0
        for i in range(m):
            itemLoss, y_pred = calculate_item_loss(X[i], Y[i], w)
            newLoss += itemLoss
            g = (y_pred - Y[i]) * X[i]
            w -= eta * g
        if(abs(newLoss - totalLoss) < 0.01):
            return w, newLoss
        totalLoss = newLoss
        
    return w

def calculateValidationLoss(v_X, v_Y, w):
    m, n = v_X.shape
    totalLoss = 0
    for i in range(m):
        totalLoss += calculate_item_loss(v_X[i], v_Y[i], w)[0]
    return totalLoss

def findBestLearningRate(X, Y, v_X=[], v_Y=[], shouldPrint=False):
    #learningRates = [0.1]
    learningRates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    bestLoss = sys.maxsize
    bestLearningRate = 0
    bestWeights = []
    for i in learningRates:
        weights, trainingLoss = learnlogreg(X,Y, i)
        validationLoss = calculateValidationLoss(v_X, v_Y, weights)
        if(shouldPrint): 
            print(i, validationLoss)
        if(validationLoss < bestLoss):
            bestLoss = validationLoss
            bestLearningRate = i
            bestWeights = weights

    return bestLearningRate, bestLoss, bestWeights

def createConfusionMatrix(w):
    confusionMatrix = np.zeros((2,2))

    a, b = arr.shape
    counter = 0
    correct = 0
    for j in range(1,b,2):
        labelEmotions = set(word_tokenize(labels[j])[:-1])
        if("+" in labelEmotions):
            labelEmotions.remove("+")
        for i in range(40, a):
            text = arr[i, j]
            if(isinstance(text, float)):
                continue
            sentences = [text]
            for sentence in sentences:
                x = create_data_xi(sentence)
                y_pred = sigmoid(w @ x)
                y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
                if(y_pred < 0.5):
                    y_pred = "Sadness"
                else:
                    y_pred = "Joy"
                

                for k in labelEmotions:
                    if(k in emotions):
                        x = emotionToIndex[y_pred]
                        y = emotionToIndex[k] 
                        confusionMatrix[x][y] += 1

    return confusionMatrix

def calculatePrecision(matrix):
    precisions = {}

    for i in range(len(matrix)): 
        if(matrix[i].sum() != 0):
            precisions[emotions[i]] = float(matrix[i][i] / matrix[i].sum())
        else:
            precisions[emotions[i]] = 0
    return precisions
        
def calculateRecall(matrix):
    recalls = {}

    matrix = matrix.T
    for i in range(len(matrix)): 
        if(matrix[i].sum() != 0):
            recalls[emotions[i]] = float(matrix[i][i] / matrix[i].sum())
        else:
            recalls[emotions[i]] = 0
    
    return recalls

def calculateAccuracy(matrix):
    accuracies = {}
    
    for i in range(len(matrix)):
        row_sum = np.sum(matrix[i, :])
        col_sum = np.sum(matrix[:, i])
        true_pos = matrix[i, i]
        true_neg = matrix.sum() - row_sum - col_sum + true_pos

        accuracies[emotions[i]] = (true_pos + true_neg) / matrix.sum()
    return accuracies

def calculateTotalAccuracy(matrix):
    correct = 0
    for i in range(len(matrix)): 
        correct += matrix[i][i]
    return correct / matrix.sum()

def calculateF1Score(matrix, P, R):
    F1Scores = {}

    for i in emotions:
        if(R[i] == 0 and P[i] == 0):
            F1Scores[i] = 0
        else:
            F1Scores[i] = (2 * P[i] * R[i]) / (P[i] + R[i])
        

    return F1Scores


X = []
Y = []

testValue = -1


create_data(X, Y, 0, 30, testValue)

X = np.array(X)
Y = np.array(Y)

validation_X = []
validation_Y = []

create_data(validation_X, validation_Y, 30, 40)
validation_X = np.array(validation_X)
validation_Y = np.array(validation_Y)

print("Loss for first example:", calculate_item_loss(X[0], Y[0], [0,0,0,0])[0])

bestLearningRate, bestLearningRateLoss, weights = findBestLearningRate(X, Y, validation_X, validation_Y, True)
print("Best learning rate:", bestLearningRate, "Corresponding Loss:", bestLearningRateLoss)



matrix = createConfusionMatrix(weights)


R = calculateRecall(matrix)
P = calculatePrecision(matrix)
A = calculateAccuracy(matrix)
F1 = calculateF1Score(matrix, P, R)
print("======== Accuracy ========")
print(A["Joy"])
print("======== Recall ========")
print(R["Joy"])
print("======== Precision ========")
print(P["Joy"])
print("======== F1 ========")
print(F1["Joy"])


