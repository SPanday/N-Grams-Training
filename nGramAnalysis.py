# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 23:27:51 2018

@author: Sakshi

Bigram Analysis on data
https://research.google.com/pubs/pub45610.html
"""
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

trainingText = "S Neural Machine Translation NMT is an end-to-end learning approach for automated translation with the potential to overcome many of the weaknesses of conventional phrase-based translation systems. S Unfortunately NMT systems are known to be computationally expensive both in training and in translation inference. S Also most NMT systems have difficulty with rare words. S These issues have hindered NMTs use in practical deployments and services where both accuracy and speed are essential. S In this work we present GNMT Googles Neural Machine Translation system which attempts to address many of these issues. S Our model consists of a deep LSTM network with 8 encoder and 8 decoder layers using attention and residual connections. S To improve parallelism and therefore decrease training time our attention mechanism connects the bottom layer of the decoder to the top layer of the encoder."
#trainingText = "Surprisingly, we have found that signal movement in the brain is closer to SO than to UE. This holds for healthy as well as depressed subjects. Nevertheless, the difference between the two is smaller in depressed subjects than in healthy ones. In other words, our analysis has two important implications: it indicates some form of global control of communication traffic that is unexpected, and raises the question of how and by whom this control is exercised, and it shows a distinction between healthy and depressed subjects that may be used as a diagnostic tool. Analysis of more subjects and finer granularities is desirable to confirm these results."

token = nltk.word_tokenize(trainingText)
trainBigrams = ngrams(token,2)
trainUnigrams = ngrams(token,1)

biTrainDict = (Counter(trainBigrams))
uniTrainDict = (Counter(token))

testText = "To accelerate the final translation speed we employ low-precision arithmetic during inference computations. To improve handling of rare words we divide words into a limited set of common sub-word units wordpieces for both input and output. This method provides a good balance between the flexibility of character-delimited models and the efficiency of word-delimited models naturally handles translation of rare words and ultimately improves the overall accuracy of the system. Our beam search technique employs a length-normalization procedure and uses a coverage penalty which encourages generation of an output sentence that is most likely to cover all the words in the source sentence. On the WMT14 English-to-French and English-to-German benchmarks GNMT achieves competitive results to state-of-the-art. Using a human side-by-side evaluation on a set of isolated simple sentences it reduces translation errors by an average of 60% compared to Googles phrase-based production system"
#testText="Furthermore, communication in the brain is closer to an SO state rather than a UE state. This is surprising since the only means of communication in the brain that neuroscience knows are the signals. Thus, in this model it is hard to see how individual signals can be controlled to achieve global optimality, and if so by whom. Is it possible that the global optimization is related to consciousness?"

def sentenceProb(ngrams, trainDict):
    sentProb = 1
    biTestSentence = Counter(ngrams)
    for i in biTestSentence:
        bigramProb = 0
        if(i in trainDict):
            bigramProb = trainDict.get(i)/uniTrainDict.get(i[0])
        sentProb *= bigramProb
        print(i, bigramProb)
    return sentProb

sentences = testText.split('.')

for sentence in sentences:
    sentence = " S " + sentence + "."
    token = nltk.word_tokenize(sentence)
    testBigrams = (ngrams(token,2))
    #testSentDict = Counter(testBigrams)
    prob = sentenceProb(testBigrams, biTrainDict)
    print("\n\n",sentence, prob)




