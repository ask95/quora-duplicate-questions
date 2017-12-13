# parser.py

import sys
from model import *
from data_gen import *
import random
import numpy as np
from sklearn.cross_validation import train_test_split



if __name__ == '__main__':
    data_path = "/u/akamath/Documents/quora-duplicate-questions/data/"
    # Use either 50-dim or 300-dim vectors
    #word_vectors = read_word_embeddings("../data/glove.6B.50d-relativized.txt")
    word_vectors = read_word_embeddings("/u/akamath/Documents/quora-duplicate-questions/data/glove.6B.300d-relativized.txt")

    # Load train, dev, and test exs
    exs = read_and_index(data_path + "mini_quora_duplicate_questions.tsv", word_vectors.word_indexer)
    train_exs, test_exs = train_test_split(exs, test_size=0.3, random_state = 42)
    #df = pd.read_csv('C:/Dataset.csv')
    #df['split'] = np.random.randn(df.shape[0], 1)

    #dev_exs = read_and_index("data/dev.txt", word_vectors.word_indexer)
    #test_exs = read_and_index(data_path + "test.tsv", word_vectors.word_indexer)

    print repr(len(train_exs)) + " / " + repr(len(test_exs)) + " train/dev/test examples"


    if len(sys.argv) >= 2:
        system_to_run = sys.argv[1]
    else:
        system_to_run = "BENCH1"
    if system_to_run == "BENCH1":
        test_exs_predicted = train_bench1(train_exs, test_exs, word_vectors, 0.1, 0.95)
        #write_sentiment_examples(test_exs_predicted, "test-blind.output.txt", word_vectors.word_indexer)
    elif system_to_run == "BENCH2":
        test_exs_predicted = train_bench2(train_exs, test_exs, word_vectors, 0.1, 0.95)
    elif system_to_run == "BENCH3":
        test_exs_predicted = train_bench3(train_exs, test_exs, word_vectors, 0.1, 0.995)
    elif system_to_run == "BENCH4":
        test_exs_predicted = train_bench4(train_exs, test_exs, word_vectors, 0.1, 0.995)
    elif system_to_run == "BENCH5":
        test_exs_predicted = train_bench5(train_exs, test_exs, word_vectors, 0.1, 0.95)
    elif system_to_run == "BENCH6":
        test_exs_predicted = train_bench6(train_exs, test_exs, word_vectors, 0.1, 0.95)
    elif system_to_run == "BENCH7":
        test_exs_predicted = train_bench7(train_exs, test_exs, word_vectors, 0.01, 0.995)
    elif system_to_run == "BENCH8":
        test_exs_predicted = train_bench8(train_exs, test_exs, word_vectors, 0.001, 0.995)
    elif system_to_run == "BENCH9":
        test_exs_predicted = train_bench9(train_exs, test_exs, word_vectors, 0.001, 0.995)
    # elif system_to_run == "BENCH2":
    #     test_exs_predicted = train_bench2(train_exs, test_exs, word_vectors)
    # elif system_to_run == "FANCY3":
    #     test_exs_predicted = train_fancy3(train_exs, dev_exs, test_exs, word_vectors)
    # elif system_to_run == "FANCY4":
    #     test_exs_predicted = train_fancy4(train_exs, dev_exs, test_exs, word_vectors)
    # elif system_to_run == "FANCY5":
    #     test_exs_predicted = train_fancy5(train_exs, dev_exs, test_exs, word_vectors)
    # elif system_to_run == "FANCY6":
    #     test_exs_predicted = train_fancy6(train_exs, dev_exs, test_exs, word_vectors)
    # elif system_to_run == "FANCY7":
    #     test_exs_predicted = train_fancy7(train_exs, dev_exs, test_exs, word_vectors)
    else:
        raise Exception("Pass in either BENCH1 or BENCH2 to run the appropriate system")
    # Write the test set output
    #write_question_pairs(test_exs_predicted, system_to_run+"_1_test-blind.output.txt", word_vectors.word_indexer)
