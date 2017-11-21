# parser.py

import sys
from model import *
from data_gen import *


if __name__ == '__main__':
    data_path = "/Volumes/anikesh_hdd/"
    # Use either 50-dim or 300-dim vectors
    #word_vectors = read_word_embeddings("data/glove.6B.50d-relativized.txt")
    word_vectors = read_word_embeddings("data/glove.6B.300d-relativized.txt")

    # Load train, dev, and test exs
    train_exs = read_and_index(data_path + "train.tsv", word_vectors.word_indexer)
    #dev_exs = read_and_index("data/dev.txt", word_vectors.word_indexer)
    test_exs = read_and_index(data_path + "test.tsv", word_vectors.word_indexer)
    
    print repr(len(train_exs)) + " / " + repr(len(dev_exs)) + " / " + repr(len(test_exs)) + " train/dev/test examples"

    if len(sys.argv) >= 2:
        system_to_run = sys.argv[1]
    else:
        system_to_run = "FANCY"
    if system_to_run == "FF":
        test_exs_predicted = train_ffnn(train_exs, dev_exs, test_exs, word_vectors)
        write_sentiment_examples(test_exs_predicted, "test-blind.output.txt", word_vectors.word_indexer)
    elif system_to_run == "FANCY":
        test_exs_predicted = train_fancy(train_exs, dev_exs, test_exs, word_vectors)
    elif system_to_run == "FANCY2":
        test_exs_predicted = train_fancy2(train_exs, dev_exs, test_exs, word_vectors)
    elif system_to_run == "FANCY3":
        test_exs_predicted = train_fancy3(train_exs, dev_exs, test_exs, word_vectors)
    elif system_to_run == "FANCY4":
        test_exs_predicted = train_fancy4(train_exs, dev_exs, test_exs, word_vectors)
    elif system_to_run == "FANCY5":
        test_exs_predicted = train_fancy5(train_exs, dev_exs, test_exs, word_vectors)
    elif system_to_run == "FANCY6":
        test_exs_predicted = train_fancy6(train_exs, dev_exs, test_exs, word_vectors)
    elif system_to_run == "FANCY7":
        test_exs_predicted = train_fancy7(train_exs, dev_exs, test_exs, word_vectors)
    else:
        raise Exception("Pass in either FF or FANCY to run the appropriate system")
    # Write the test set output
    write_question_pairs(test_exs_predicted, system_to_run+"_1_test-blind.output.txt", word_vectors.word_indexer)