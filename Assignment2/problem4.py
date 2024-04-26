#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""
import codecs
import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random


vocab = codecs.open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    line = line.rstrip()
    if line not in word_index_dict:
        word_index_dict[line] = i

    #TODO: import part 1 code to build dictionary

f = codecs.open("brown_100.txt")

counts = np.zeros((len(word_index_dict),len(word_index_dict)))
text = f.read()
words = text.split()
previous_word = words[0].lower()
for i in range(1,len(words)):
    current_word = words[i].lower()
    counts[word_index_dict[previous_word],word_index_dict[current_word]] += 1
    previous_word = current_word

counts += 0.1
#import sys
# np.set_printoptions(threshold=sys.maxsize)
# print(counts)
probs = normalize(counts,norm='l1',axis=1)
example_words = [('all', 'the'),
         ('the', 'jury'),
         ('the', 'campaign'),
         ('anonymous','calls')]
new_file = open('smooth_probs.txt',"w")
for i,tuple in enumerate(example_words):
    new_file.write(str(probs[word_index_dict[example_words[i][0]],word_index_dict[example_words[i][1]]]) + "\n")
new_file.close()
 #TODO: normalize counts


# #TODO: writeout bigram probabilities



f.close()