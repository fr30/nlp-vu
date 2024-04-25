import numpy as np


word_to_index = {}

with open("brown_vocab_100.txt", "r") as file:
    lines = file.readlines()
    for index, line in enumerate(lines):
        word_to_index[line.strip()] = index

with open("brown_100.txt", "r") as file:
    lines = file.readlines()


def create_ngram_model(n):
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

    return counts


n = len(word_to_index)
count1 = create_ngram_model(1)
count2 = create_ngram_model(2)
pmi = np.zeros((n, n))

with open("brown_100.txt", "r") as file:
    for line in file.readlines():
        words = [word.lower() for word in line.rstrip().split()]
        index = [word_to_index[word] for word in words]
        for i in range(1, len(index)):
            index1, index2 = index[i - 1], index[i]

            if count1[index1] < 10 or count1[index2] < 10:
                continue

            pmi[index1, index2] = np.log(
                count2[index1, index2] * n / (count1[index1] * count1[index2])
            )

k = 20
index_to_word = {index: word for word, index in word_to_index.items()}
sorted_pmi = np.sort(pmi, axis=None)
top_k = np.argwhere(np.isin(pmi, sorted_pmi[-k:]))
bottom_k = np.argwhere(np.isin(pmi, sorted_pmi[:k]))
top_k_words = [(index_to_word[i1], index_to_word[i2]) for i1, i2 in top_k]
bottom_k_words = [(index_to_word[i1], index_to_word[i2]) for i1, i2 in bottom_k]

for (w1, w2), p in zip(top_k_words, pmi[top_k[:, 0], top_k[:, 1]]):
    print(f"PMI({w1}, {w2}) = {p:.2f}")

for (w1, w2), p in zip(bottom_k_words, pmi[bottom_k[:, 0], bottom_k[:, 1]]):
    print(f"{w1}, {w2} = {p:.2f}")
