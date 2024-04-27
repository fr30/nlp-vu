#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""


def return_word_index_dict(
    from_path: str = "txt/brown_vocab_100.txt",
) -> dict[str, int]:
    word_index_dict = {}

    with open(from_path, "r") as f:
        for i, w in enumerate(f.readlines()):
            word_index_dict[w.rstrip()] = i
    return word_index_dict


def main() -> None:
    word_index_dict = return_word_index_dict()
    with open("output/word_to_index_100.txt", "w") as f:
        f.write(str(word_index_dict))

    print(word_index_dict["all"])
    print(word_index_dict["resolution"])
    print(len(word_index_dict))


if __name__ == "__main__":
    main()
