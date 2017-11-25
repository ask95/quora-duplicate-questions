import sys
import numpy as np
import matplotlib.pyplot as plt
from data_gen import *
from random import shuffle
#from sklearn.cross_validation import train_test_split
from model import *
#question_pairs_file = sys.argv[1]
#embeddings_file = sys.argv[2]

word_vectors = read_word_embeddings('../data/paragram-phrase-XXL.txt')
exs = read_and_index('../data/q10k.txt', word_vectors.word_indexer)
ppdb = read_and_index_ppdb('../data/all_pairs_1k.txt', word_vectors.word_indexer)
print len(ppdb)
exs = exs[:1000]
#exs = read_and_index('../data/quora_duplicate_questions.tsv', word_vectors.word_indexer)
#print len(exs)
#raw_input()

def train_test_split(exs, test_size):
    shuffle(exs)
    n = len(exs)
    return (exs[int(test_size * n):], exs[:int(test_size * n)])

train_exs, test_exs = train_test_split(exs, test_size=0.3)

#avg_model = train_avg_model(train_exs, word_vectors)
#test_exs_predicted = avg_model.predict(test_exs)

lstm_model = train_lstm_model(train_exs, word_vectors, ppdb)

# load embeddings
word_emb = {}
for line in open(embeddings_file):
  line = line.strip()
  parts = line.split(' ')
  word_emb[parts[0]] = map(eval, parts[1:])

label0_pred = []
label1_pred = []



for line in open(question_pairs_file):
  line = line.strip()
  parts = line.split('\t')
  try:
    label = eval(parts[5])
  except:
    continue
  # generate average embedding
  ques1 = parts[3]
  ques2 = parts[4]

  ques1_emb = np.zeros(len(word_emb['the']))
  count = 0
  for word in ques1.split():
    word = word.lower()
    word = filter(lambda ch: ch not in '?', word)
    try:
      emb = word_emb[word]
      ques1_emb += emb
      count += 1
      #print 'Added:', word
    except KeyError:
      pass
  if count == 0:
    continue
  ques1_emb /= count

  ques2_emb = np.zeros(len(word_emb['the']))
  count = 0
  for word in ques2.split():
    word = word.lower()
    word = filter(lambda ch: ch not in '?', word)
    try:
      emb = word_emb[word]
      ques2_emb += emb
      count += 1
      #print 'Added:', word
    except KeyError:
      pass
      #print 'Not found:', word
  if count == 0:
    continue
  ques2_emb /= count

  #pred = np.sum(ques1_emb * ques2_emb)
  pred = np.sum(ques1_emb * ques2_emb) / (np.linalg.norm(ques1_emb) * np.linalg.norm(ques2_emb))
  if label == 0:
    label0_pred.append(pred)
  else:
    label1_pred.append(pred)

#bins = np.linspace(-5, 15, 100)
bins = np.linspace(-1, 2, 100)
plt.hist(label0_pred, bins, alpha=0.5, label='0')
plt.hist(label1_pred, bins, alpha=0.5, label='1')
plt.legend(loc='upper right')
plt.show()

