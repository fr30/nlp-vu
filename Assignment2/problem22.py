#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np

from generate import GENERATE
from problem1 import return_word_index_dict


def main() -> None:
    # load the indices dictionary
    word_index_dict = return_word_index_dict()

    with open("txt/brown_100.txt") as f:
        words = [w.lower() for sent in f.readlines() for w in sent.split()]

    counts = np.zeros(len(word_index_dict))

    for word in words:
        counts[word_index_dict[word]] += 1

    probs = counts / np.sum(counts)

    np.savetxt("output/unigram_probs.txt", probs)


if __name__ == "__main__":
    main()
