import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from data_gen import *
#from random import shuffle
import random
#from sklearn.cross_validation import train_test_split
from model import *
#question_pairs_file = sys.argv[1]
#embeddings_file = sys.argv[2]

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--lr-decay', type=float, default=0.95, help='learning rate decay factor')
parser.add_argument('--lr-step', type=int, default=1000, help='learning rate decay step')
parser.add_argument('--lstm-size', type=int, default=100, help='LSTM size')
#parser.add_argument('--variant', default=None, help='mix | alt | 1by1')
#parser.add_argument('--scaling', type=float, default=1.0, help='scaling factor for PPDB')
#parser.add_argument('--n-train', type=int, default=100000, help='no. of Quora examples to use for training')
args = parser.parse_args()

#word_vectors = read_word_embeddings('../data/paragram-phrase-XXL.txt')
word_vectors = read_word_embeddings('../data/glove.6B.300d-relativized.txt')
exs = read_and_index('../data/q100k.txt', word_vectors.word_indexer)
ppdb = read_and_index_ppdb('../data/all_pairs_100k.txt', word_vectors.word_indexer)
#exs = exs[:1000]
#exs = read_and_index('../data/quora_duplicate_questions.tsv', word_vectors.word_indexer)
#raw_input()

def train_valid_test_split(exs):
    random.Random(17).shuffle(exs)
    n = len(exs)
    s1 = int(n * 0.6)
    s2 = int(n * 0.8)
    return (exs[:s1], exs[s1:s2], exs[s2:])

train_exs, valid_exs, test_exs = train_valid_test_split(exs)
#train_exs = train_exs[:args.n_train]
print len(train_exs), 'examples from Quora dataset.'
print len(ppdb), 'examples from PPDB.'

#avg_model = train_avg_model(train_exs, word_vectors)
#test_exs_predicted = avg_model.predict(test_exs)

#print args.lr
#print args.lr_decay
#print args.lr_step
#print args.lstm_size

#lstm_model = train_lstm_model(train_exs, word_vectors, ppdb, test_exs)
lstm_model = train_lstm_model_two_step(train_exs, word_vectors, ppdb, valid_exs, test_exs, args.lstm_size, args.lr, args.lr_step, args.lr_decay)

