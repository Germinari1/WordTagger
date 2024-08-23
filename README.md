# Hidden Markov Model Part-of-Speech Tagger

This project implements a Part-of-Speech (POS) tagger using a Hidden Markov Model (HMM). It's designed to analyze text and assign grammatical tags to each word, providing insights into the structure and meaning of sentences and demonstrate one use of Markov models.

Here are some things that you will find in this repository:
- Train a Hidden Markov Model for POS tagging
- Tag words in user-provided text files
- Save and load trained models for quick reuse
- Option to retrain the model on demand
- Output tagging results to console or file

## Requirements

- Python 3.6+
- NumPy
- Pandas

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/hmm-pos-tagger.git
   cd hmm-pos-tagger
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

The main script `main.py` provides a command-line interface for using the POS tagger. Here are some common usage patterns:

1. Tag a file using an existing model (or train a new one if no model exists):
   ```
   python main.py input.txt
   ```

2. Force retraining of the model before tagging:
   ```
   python main.py --retrain input.txt
   ```

3. Save tagging results to a file:
   ```
   python main.py input.txt --output results.txt
   ```

4. Combine options:
   ```
   python main.py --retrain input.txt --output results.txt
   ```

For more information on available options, use the help flag:
```
python main.py --help
```

## Descriptions

- `main.py`: The main script for running the POS tagger
- `pos_tagger.py`: Contains the implementation of the HMM-based POS tagger
- `WSJ_02-21.pos`: Training data file 
- `hmm_vocab.txt`: Vocabulary file for the model 
- `pos_tagger_model.pkl`: Saved model file (generated after training)

When saving results to a file, each line contains a word and its predicted tag, separated by a tab:

```
word1   TAG1
word2   TAG2
...
```
