# Deep Learning Hw2

## Introduction
In this assignment, Implementing a sequence-to-sequence (S2VT) to generate caption for input videos

## Requirement
* Tensorflow 1.15
* numpy
* pickle
* json

## Details about each Part and corresponding code
* dataprocess.py : Prepare the data for training and testing

* model_seq2seq.py : Model,training and testing

* bleu_eval.py : calcualte the bleu score based on test result, which is saved in test_result.txt

* test_result.txt : The output form the testing

* model: trained model

* data_tokenizer: training data token


## test the model performance
Add dataset -MLDS_hw2_data in current directory and run following command
```c
sh hw2_seq2seq.sh  ./MLDS_hw2_1_data/testing_data/feat/  test_result.txt
```
## training
Comment the model testing part and run following command:
```c

python model_seq2seq.py 

```
## testing
Comment the model training part and run following command:
```c
python model_seq2seq.py
```
