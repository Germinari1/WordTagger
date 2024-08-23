############################################################################
# Author: Lucas Germinari
# Description: Implements a Hidden Markov Model (HMM) for POS tagging.
# Notes: 
############################################################################

import numpy as np
import pandas as pd
from collections import defaultdict
import math

class POSTagger:
    def __init__(self, alpha=0.001):
        self.alpha = alpha
        self.states = []
        self.vocab = {}
        self.tag_counts = defaultdict(int)
        self.emission_counts = defaultdict(int)
        self.transition_counts = defaultdict(int)
        self.A = None
        self.B = None
        self.noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
        self.verb_suffix = ["ate", "ify", "ise", "ize"]
        self.adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
        self.adv_suffix = ["ward", "wards", "wise"]
        self.punct = set(".,;:!?()[]{}\"'")  

    def load_data(self, training_file, vocab_file):
        with open(training_file, 'r') as f:
            self.training_corpus = f.readlines()

        with open(vocab_file, 'r') as f:
            voc_l = f.read().split('\n')

        for i, word in enumerate(sorted(voc_l)):
            self.vocab[word] = i

    def create_dictionaries(self):
        prev_tag = '--s--'
        for word_tag in self.training_corpus:
            word, tag = self.get_word_tag(word_tag, self.vocab)  # Added self.vocab argument
            self.transition_counts[(prev_tag, tag)] += 1
            self.emission_counts[(tag, word)] += 1
            self.tag_counts[tag] += 1
            prev_tag = tag
        self.states = sorted(self.tag_counts.keys())

    def create_transition_matrix(self):
        num_tags = len(self.states)
        self.A = np.zeros((num_tags, num_tags))
        for i, prev_tag in enumerate(self.states):
            for j, tag in enumerate(self.states):
                count = self.transition_counts.get((prev_tag, tag), 0)
                self.A[i, j] = (count + self.alpha) / (self.tag_counts[prev_tag] + self.alpha * num_tags)

    def create_emission_matrix(self):
        num_tags = len(self.states)
        num_words = len(self.vocab)
        self.B = np.zeros((num_tags, num_words))
        for i, tag in enumerate(self.states):
            for word, j in self.vocab.items():
                count = self.emission_counts.get((tag, word), 0)
                self.B[i, j] = (count + self.alpha) / (self.tag_counts[tag] + self.alpha * num_words)

    def viterbi_algorithm(self, corpus):
        num_tags = len(self.states)
        best_probs = np.zeros((num_tags, len(corpus)))
        best_paths = np.zeros((num_tags, len(corpus)), dtype=int)

        # Initialization
        s_idx = self.states.index("--s--")
        for i in range(num_tags):
            if self.A[s_idx, i] == 0:
                best_probs[i, 0] = float('-inf')
            else:
                best_probs[i, 0] = math.log(self.A[s_idx, i]) + math.log(self.B[i, self.vocab[corpus[0]]])

        # Forward pass
        for t in range(1, len(corpus)):
            for j in range(num_tags):
                best_prob = float('-inf')
                best_path = 0
                for i in range(num_tags):
                    prob = best_probs[i, t-1] + math.log(self.A[i, j]) + math.log(self.B[j, self.vocab[corpus[t]]])
                    if prob > best_prob:
                        best_prob = prob
                        best_path = i
                best_probs[j, t] = best_prob
                best_paths[j, t] = best_path

        # Backward pass
        z = [0] * len(corpus)
        pred = [''] * len(corpus)
        z[-1] = np.argmax(best_probs[:, -1])
        pred[-1] = self.states[z[-1]]

        for t in range(len(corpus) - 2, -1, -1):
            z[t] = best_paths[z[t+1], t+1]
            pred[t] = self.states[z[t]]

        return pred

    def assign_unk(self, tok):
        """
        Assign unknown word tokens
        """
        # Digits
        if any(char.isdigit() for char in tok):
            return "--unk_digit--"

        # Punctuation
        elif any(char in self.punct for char in tok):
            return "--unk_punct--"

        # Upper-case
        elif any(char.isupper() for char in tok):
            return "--unk_upper--"

        # Nouns
        elif any(tok.endswith(suffix) for suffix in self.noun_suffix):
            return "--unk_noun--"

        # Verbs
        elif any(tok.endswith(suffix) for suffix in self.verb_suffix):
            return "--unk_verb--"

        # Adjectives
        elif any(tok.endswith(suffix) for suffix in self.adj_suffix):
            return "--unk_adj--"

        # Adverbs
        elif any(tok.endswith(suffix) for suffix in self.adv_suffix):
            return "--unk_adv--"

        return "--unk--"

    def preprocess(self, data_fp):
        """
        Preprocess data
        """
        orig = []
        prep = []

        # Read data
        with open(data_fp, "r") as data_file:
            for cnt, word in enumerate(data_file):
                # End of sentence
                if not word.split():
                    orig.append(word.strip())
                    word = "--n--"
                    prep.append(word)
                    continue

                # Handle unknown words
                elif word.strip() not in self.vocab:
                    orig.append(word.strip())
                    word = self.assign_unk(word)
                    prep.append(word)
                    continue

                else:
                    orig.append(word.strip())
                    prep.append(word.strip())

        assert(len(orig) == len(open(data_fp, "r").readlines()))
        assert(len(prep) == len(open(data_fp, "r").readlines()))

        return orig, prep

    def get_word_tag(self, line, vocab):
        if not line.split():
            word = "--n--"
            tag = "--s--"
            return word, tag
        else:
            word, tag = line.split()
            if word not in vocab:
                # Handle unknown words
                word = self.assign_unk(word)
            return word, tag

    def train(self, training_file, vocab_file):
        self.load_data(training_file, vocab_file)
        self.create_dictionaries()
        self.create_transition_matrix()
        self.create_emission_matrix()

    def predict(self, corpus):
        return self.viterbi_algorithm(corpus)

    def compute_accuracy(self, pred, y):
        num_correct = 0
        total = 0
        for prediction, label in zip(pred, y):
            word_tag_tuple = label.split()
            if len(word_tag_tuple) != 2:
                continue
            _, tag = word_tag_tuple
            if prediction == tag:
                num_correct += 1
            total += 1
        return num_correct / total if total > 0 else 0

