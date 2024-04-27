#!/usr/bin/env python3

import argparse
import os
import re
from collections import defaultdict
from typing import DefaultDict, Optional

import matplotlib.pyplot as plt
import nltk
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.tagged import CategorizedTaggedCorpusReader
from nltk.corpus.util import LazyCorpusLoader

NLTK_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nltk_data")

nltk.data.path.append(NLTK_DATA)


class CorpusNameNotFound(Exception):
    def __init__(self, string: str, *args: object) -> None:
        msg = f"Unable to find corpus name in: {string}"
        super().__init__(msg, *args)


class DownloadCorpusError(Exception):
    def __init__(self, name: str, *args: object) -> None:
        msg = f"Unable to download corpus: {name}"
        super().__init__(msg, *args)


def download_corpus(name: str) -> bool:
    if not os.path.exists(NLTK_DATA):
        os.mkdir(NLTK_DATA)
    return nltk.download(name, download_dir=NLTK_DATA)


def get_corpus_name(corpus: CorpusReader) -> str:
    name = re.search(r"corpora/([\d\w]*)'", str(corpus))
    if name:
        return name[1]
    raise CorpusNameNotFound(str(corpus))


def ensure_loaded(corpus: LazyCorpusLoader) -> CategorizedTaggedCorpusReader:
    try:
        corpus.ensure_loaded()
    except LookupError:
        name = get_corpus_name(corpus)
        if not download_corpus(name):
            raise DownloadCorpusError(name)
        corpus.ensure_loaded()
    return corpus


def return_word_to_freq(words: list[str]) -> DefaultDict[str, int]:
    word_to_freq = defaultdict(int)
    for w in words:
        word_to_freq[w] += 1
    return word_to_freq


def extract_words(tokens: list[str]) -> list[str]:
    words = [w for w in tokens if any(c.isalpha() for c in w)]
    return words


def return_pos_to_freq(pos_tags: list[tuple[str, str]]) -> DefaultDict[str, int]:
    pos_to_freq = defaultdict(int)
    for _, tag in pos_tags:
        pos_to_freq[tag] += 1
    return pos_to_freq


def plot_freq_curves(word_to_freq: dict[str, int], fname: str) -> None:
    x = range(1, len(word_to_freq) + 1)
    y = sorted(word_to_freq.values(), reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    for ax, scale in zip(axes, ["linear", "log"]):
        ax.plot(x, y, marker="o", label=f"Frec. curve with {scale}-{scale} axes")
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Word rank")
        ax.legend()
    fig.savefig(fname)


def main(_corpus: LazyCorpusLoader, category: Optional[str] = None):
    corpus = ensure_loaded(_corpus)

    print(f"Running stats on '{corpus.root}' ...\n")

    if category:
        assert category in corpus.categories()
        print(f"Running stats for category {category}\n")

    tokens = corpus.words(categories=category)
    sents = corpus.sents(categories=category)
    words = extract_words(tokens)

    av_words_per_sent = len(words) / len(sents)
    av_word_len = sum(len(w) for w in words) / len(words)
    pos_tags_freq = return_pos_to_freq(corpus.tagged_words(categories=category))

    print(f"Number of tokens: {len(tokens)}")
    print(f"Number of types: {len(set(tokens))}")
    print(f"Number of words: {len(words)}")
    print(f"Average words per sentence: {round(av_words_per_sent, 2)}")
    print(f"Average word length: {round(av_word_len, 2)}")
    print(
        f"10 most frequent POS tags: {dict(sorted(pos_tags_freq.items(), key=lambda x: x[1], reverse=True)[:10]).keys()}"
    )

    fname=f"imgs/freq_curves_{get_corpus_name(corpus)}{"_" + category if category else ""}.png"
    plot_freq_curves( return_word_to_freq(words), fname=fname)

    print(f"\nImage saved to {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, help="Corpus name", required=True)
    parser.add_argument("-c", "--category", type=str, help="Corpus category. Omit for all", required=False)

    args = parser.parse_args()

    corpus_name = args.name
    category = args.category

    # Load corups
    corpus = getattr(nltk.corpus, corpus_name)

    # Run the things
    main(corpus, category=category)
