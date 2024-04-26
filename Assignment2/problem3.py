#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np

word_to_index = {}

with open("brown_vocab_100.txt", "r") as file:
    lines = file.readlines()
    for index, line in enumerate(lines):
        word_to_index[line.strip()] = index

with open("brown_100.txt", "r") as file:
    lines = file.readlines()


# Replace create_ngram_model() with your own implementation of n-gram
def create_ngram_model(n, smoothing):
    shape = (len(word_to_index),) * n
    counts = np.zeros(shape)
    last_n_words = np.zeros(n, dtype=int)

    with open("brown_100.txt", "r") as file:
        for line in file.readlines():
            words = [word.lower() for word in line.rstrip().split()]
            last_n_words[:] = [word_to_index[word] for word in words[:n]]

            for word in words[n:]:
                counts[tuple(last_n_words)] += 1
                index = word_to_index[word]
                last_n_words = np.roll(last_n_words, -1)
                last_n_words[-1] = index

            counts[tuple(last_n_words)] += 1

    if smoothing:
        counts += 0.1

    # Avoid division by zero
    eps = 1e-23
    probs = counts / (counts.sum(axis=n - 1, keepdims=True) + eps)
    return probs


# Code below for evaluation
def get_last_word_prob(sentence, probs, n):
    words = [word.lower() for word in sentence.rstrip().split()]
    word_ids = [word_to_index[word] for word in words[:n]]

    return probs[tuple(word_ids)]


N = 2
smoothing = False
probs = create_ngram_model(N, smoothing)
#writing prob file

with open("toy_corpus.txt", "r") as file:
    for line in file.readlines():
        target_words = [word.lower() for word in line.rstrip().split()]
        sentprob = 1.0

        for index in range(N, len(target_words) + 1):
            subsentence = " ".join(target_words[index - N : index])
            subprob = get_last_word_prob(subsentence, probs, N)
            sentprob *= subprob

        perplexity = 1.0 / np.power(sentprob, 1.0 / (len(target_words) - N + 1))
        print(perplexity)

example_words = [('all', 'the'),
         ('the', 'jury'),
         ('the', 'campaign'),
         ('anonymous','calls')]
new_file = open('bigram_probs.txt',"w")
for i,tuple in enumerate(example_words):
    new_file.write(str(probs[word_to_index[example_words[i][0]],word_to_index[example_words[i][1]]]) + "\n")
new_file.close()
