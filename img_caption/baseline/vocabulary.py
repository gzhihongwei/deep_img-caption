"""Title: Automatic-Image-Captioning
Author: Kshirsagar, Krunal
Date: 2020
Availability: https://github.com/Noob-can-Compile/Automatic-Image-Captioning/
For generating vocabulary given the captions
"""

import nltk
import pickle
import os.path
from pycocotools.coco import COCO
from collections import Counter


class Vocabulary(object):
    def __init__(self,
                 vocab_threshold,
                 vocab_file="./vocab.pkl",
                 start_word="<SOS>",
                 end_word="<EOS>",
                 unk_word="<UNK>",
                 annotations_file="../../datasets/coco/annotations/captions_train2014.json",
                 vocab_from_file=False):
        """
        Initialize the vocabulary.
        :param vocab_threshold: Minimum word count threshold.
        :param vocab_file: File containing the vocabulary.
        :param start_word: Special word denoting sentence start.
        :param end_word: Special word denoting sentence end.
        :param unk_word: Special word denoting unknown words.
        :param annotations_file: Path for train annotation file.
        :param vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                                If True, load vocab from existing vocab_file if it exists
        """

        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""

        if os.path.exists(self.vocab_file) and self.vocab_from_file:
            with open(self.vocab_file, "rb") as f:
                vocab = pickle.load(f)
                self.stoi = vocab.stoi
                self.itos = vocab.itos

            print("Vocabulary successfully loaded from vocab.pkl file!")

        else:
            self.build_vocab()
            with open(self.vocab_file, "wb") as f:
                pickle.dump(self, f)

    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""

        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)"""

        self.stoi = {}
        self.itos = {}
        self.idx = 0

    def add_word(self, word):
        """Add a token to the vocabulary."""

        if word not in self.stoi:
            self.stoi[word] = self.idx
            self.itos[self.idx] = word
            self.idx += 1

    def add_captions(self):
        """Loop over training captions and add all tokens to the vocabulary that meet of exceed the threshold."""

        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = coco.anns.keys()

        for i, id in enumerate(ids):
            caption = str(coco.anns[id]["caption"])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if i % 100000 == 0:
                print(f"[{i + 1}/{len(ids)}] Tokenizing captions...")

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for word in words:
            self.add_word(word)

    def __call__(self, word):
        # Either get the index of the word or the index of the unknown token
        return self.stoi.get(word, self.stoi[self.unk_word])

    def __len__(self):
        return len(self.stoi)
