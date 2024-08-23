############################################################################
# Author: Lucas Germinari
# Description: helper functions for training and loading the word tagging model
# Notes: 
############################################################################

import pickle
from pos_tagger import POSTagger

MODEL_FILE = 'pos_tagger_model.pkl'
TRAINING_FILE = "WSJ_02-21.pos"
VOCAB_FILE = "hmm_vocab.txt"

def save_model(tagger, filename):
    with open(filename, 'wb') as f:
        pickle.dump(tagger, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def train_model():
    print("Training new model...")
    tagger = POSTagger()
    tagger.train(TRAINING_FILE, VOCAB_FILE)
    save_model(tagger, MODEL_FILE)
    return tagger

def save_results(words, tags, filename):
    with open(filename, 'w') as f:
        for word, tag in zip(words, tags):
            f.write(f"{word}\t{tag}\n")
    print(f"Results saved to {filename}")
