#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""
import numpy as np

from generate import GENERATE

word_to_index = {}

with open("txt/brown_vocab_100.txt", "r") as file:
    lines = file.readlines()
    for index, line in enumerate(lines):
        word_to_index[line.strip()] = index

with open("txt/brown_100.txt", "r") as file:
    lines = file.readlines()


def create_ngram_model(n, smoothing):
    shape = (len(word_to_index),) * n
    counts = np.zeros(shape)
    last_n_words = np.zeros(n, dtype=int)

    with open("txt/brown_100.txt", "r") as file:
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


# Change this parameter to generate with different n-gram models
N = 2
num_sentences = 10
smoothing = True
probs = create_ngram_model(N, smoothing)

if N == 1:
    model = "unigram"
elif N == 2:
    model = "bigram"

for i in range(num_sentences):
    sentence = GENERATE(word_to_index, probs, model, 150, "<s>")
    print(f"========={i}=========")
    print(sentence)
