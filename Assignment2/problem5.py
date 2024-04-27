#!/usr/bin/env python3

import numpy as np


word_to_index = {}

with open("txt/brown_vocab_100.txt", "r") as file:
    lines = file.readlines()
    for index, line in enumerate(lines):
        word_to_index[line.strip()] = index

with open("txt/brown_100.txt", "r") as file:
    lines = file.readlines()


def get_last_word_prob(sentence, smoothing=False):
    target_words = [word.lower() for word in sentence.rstrip().split()]
    target_ids = np.array([word_to_index[word] for word in target_words])
    target_count = 0

    prefix_ids = target_ids[:-1]
    prefix_count = 0

    n = len(target_ids) - 1
    last_n_words = np.zeros(n, dtype=int)

    for line in lines:
        words = [word.lower() for word in line.rstrip().split()]
        last_n_words[:] = [word_to_index[word] for word in words[:n]]

        for word in words[n:]:
            prefix_correct = last_n_words == prefix_ids
            prefix_count += int(np.all(prefix_correct))

            index = word_to_index[word]

            target_correct = np.concatenate([prefix_correct, [index == target_ids[-1]]])
            target_count += int(np.all(target_correct))

            last_n_words = np.roll(last_n_words, -1)
            last_n_words[-1] = index

    if smoothing:
        words_in_dict = len(word_to_index)
        prefix_count += words_in_dict * 0.1
        target_count += 0.1

    return target_count / prefix_count


sentences = [
    "in the past",
    "in the time",
    "the jury said",
    "the jury recommended",
    "jury said that",
    "agriculture teacher ,",
]

for sentence in sentences:
    print(sentence)
    prob_no_smooth = get_last_word_prob(sentence, smoothing=False)
    prob_smooth = get_last_word_prob(sentence, smoothing=True)
    print(f"Prob w/out smooth: {prob_no_smooth}\nProb w smooth: {prob_smooth}")
    print()
