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

trainingText = "Neural Machine Translation NMT is an end-to-end learning approach for automated translation with the potential to overcome many of the weaknesses of conventional phrase-based translation systems. Unfortunately NMT systems are known to be computationally expensive both in training and in translation inference. Also most NMT systems have difficulty with rare words. These issues have hindered NMTs use in practical deployments and services where both accuracy and speed are essential. In this work we present GNMT Googles Neural Machine Translation system which attempts to address many of these issues. Our model consists of a deep LSTM network with 8 encoder and 8 decoder layers using attention and residual connections. To improve parallelism and therefore decrease training time our attention mechanism connects the bottom layer of the decoder to the top layer of the encoder."
token = nltk.word_tokenize(trainingText)
trainBigrams = ngrams(token,2)

biTrainDict = (Counter(trainBigrams))

testText = "To accelerate the final translation speed we employ low-precision arithmetic during inference computations. To improve handling of rare words we divide words into a limited set of common sub-word units wordpieces for both input and output. This method provides a good balance between the flexibility of character-delimited models and the efficiency of word-delimited models naturally handles translation of rare words and ultimately improves the overall accuracy of the system. Our beam search technique employs a length-normalization procedure and uses a coverage penalty which encourages generation of an output sentence that is most likely to cover all the words in the source sentence. On the WMT14 English-to-French and English-to-German benchmarks GNMT achieves competitive results to state-of-the-art. Using a human side-by-side evaluation on a set of isolated simple sentences it reduces translation errors by an average of 60% compared to Googles phrase-based production system."


sentences = testText.split('.')
testBigrams = []
for sentence in sentences:
    sentence = sentence + "."
    testBigrams.append(ngrams(token,2))


for sent in testBigrams:
    print("\n\n\n\n",Counter(sent))
