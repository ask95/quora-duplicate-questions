import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from data_gen import *
from random import shuffle
#from sklearn.cross_validation import train_test_split
from model import *
#question_pairs_file = sys.argv[1]
#embeddings_file = sys.argv[2]

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--lr-decay', type=float, default=0.95, help='learning rate decay factor')
parser.add_argument('--lr-step', type=int, default=1000, help='learning rate decay step')
parser.add_argument('--lstm-size', type=int, default=100, help='LSTM size')
parser.add_argument('--variant', default=None, help='mix | alt | 1by1')
parser.add_argument('--scaling', type=float, default=1.0, help='scaling factor for PPDB')
args = parser.parse_args()

word_vectors = read_word_embeddings('../data/paragram-phrase-XXL.txt')
exs = read_and_index('../data/q100k.txt', word_vectors.word_indexer)
ppdb = read_and_index_ppdb('../data/all_pairs_100k.txt', word_vectors.word_indexer)
#exs = exs[:1000]
#exs = read_and_index('../data/quora_duplicate_questions.tsv', word_vectors.word_indexer)
#raw_input()

def train_valid_test_split(exs):
    shuffle(exs)
    n = len(exs)
    s1 = n * 0.6
    s2 = n * 0.8
    return (exs[int(test_size * n):], exs[:int(test_size * n)])

train_exs, test_exs = train_test_split(exs, test_size=0.3)
print len(train_exs), 'examples from Quora dataset.'
print len(ppdb), 'examples from PPDB.'

#avg_model = train_avg_model(train_exs, word_vectors)
#test_exs_predicted = avg_model.predict(test_exs)

#print args.lr
#print args.lr_decay
#print args.lr_step
#print args.lstm_size

#lstm_model = train_lstm_model(train_exs, word_vectors, ppdb, test_exs)
lstm_model = train_lstm_model(train_exs, word_vectors, ppdb, test_exs, args.lstm_size, args.lr, args.lr_step, args.lr_decay, args.variant, args.scaling)

