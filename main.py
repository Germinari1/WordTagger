############################################################################
# Author: Lucas Germinari
# Description: Part-of-Speech Tagger using Hidden Markov Model. Accepts a text file as input and tags the words in it.z
# Notes: 
############################################################################

import sys
import os
import argparse
from pos_tagger import POSTagger
from modelUtils import *

def main():
    parser = argparse.ArgumentParser(
        description="Part-of-Speech Tagger using Hidden Markov Model",
        epilog="""
Examples:
  python main.py input.txt
    Tag the text in input.txt using the existing model (or train a new one if no model exists).

  python main.py --retrain input.txt
    Force retraining of the model before tagging input.txt.

  python main.py input.txt --output results.txt
    Tag the text and save the results to results.txt.

Note:
  - The program uses 'WSJ_02-21.pos' for training and 'hmm_vocab.txt' for vocabulary by default.
  - The trained model is saved as 'pos_tagger_model.pkl' for future use.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input_file', help="Path to the input file containing text to be tagged")
    parser.add_argument('--retrain', action='store_true', help="Force retraining of the model before tagging")
    parser.add_argument('--output', help="Path to save the tagging results (if not specified, results are printed to console)")
    args = parser.parse_args()

    # Check if model exists or if retraining is requested
    if not os.path.exists(MODEL_FILE) or args.retrain:
        tagger = train_model()
    else:
        print("Loading existing model...")
        tagger = load_model(MODEL_FILE)

    # Read user's input file
    try:
        with open(args.input_file, 'r') as f:
            user_text = f.read().split()
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    # Preprocess the user's text
    processed_text = [tagger.assign_unk(word) if word not in tagger.vocab else word for word in user_text]

    # Predict tags
    predictions = tagger.predict(processed_text)

    # Output results
    if args.output:
        save_results(user_text, predictions, args.output)
    print("\nWord\tPredicted Tag")
    print("-----------------")
    for word, tag in zip(user_text, predictions):
        print(f"{word}\t{tag}")

if __name__ == "__main__":
    main()