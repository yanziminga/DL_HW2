#!/bin/sh

TEST_FEATURE_DIRECTORY=("$1")
TEST_OUTPUT_FILENAME=("$2")

python model_seq2seq.py $TEST_FEATURE_DIRECTORY
python bleu_eval.py $TEST_OUTPUT_FILENAME