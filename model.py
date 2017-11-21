# models.py

import tensorflow as tf
import numpy as np
import random
from sentiment_data import *
#from tf.nn.rnn_cell import LSTMCell


# Returns a new numpy array with the data from np_arr padded to be of length length. If length is less than the
# length of the base array, truncates instead.
def pad_to_length(np_arr, length):
    result = np.ones(length)*(-1)
    result[0:np_arr.shape[0]] = np_arr
    return result


def train(train_exs, dev_exs, test_exs, word_embeddings)


# Train a feedforward neural network on the given training examples, using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value). Returned predictions
# should be SentimentExample objects with predicted labels and the same sentences as input (but these won't be
# read for evaluation anyway)
def train_ffnn(train_exs, dev_exs, test_exs, word_embeddings):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    serial_input = np.zeros([len(train_exs), seq_max_len, dim])



    print "Extraction begins!"

    for i in range(len(train_mat)):
        for wi in range(len(train_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(train_mat[i][wi])
            #for j in range(len(word_vec)):
            serial_input[i][wi] = word_vec

    serial_input = serial_input.mean(axis=1)

                #print word_vec[j]
                


    print "Extraction ends!"

    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    # Labels
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    dev_serial_input = np.zeros([len(dev_exs), seq_max_len, dim])

    print "Extraction begins!"

    for i in range(len(dev_mat)):
        for wi in range(len(dev_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(dev_mat[i][wi])
            dev_serial_input[i][wi] = word_vec

    dev_serial_input = dev_serial_input.mean(axis=1)

    print "Extraction ends!"
    
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    # Labels
    test_labels_arr = np.array([ex.label for ex in test_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    test_serial_input = np.zeros([len(test_exs), seq_max_len, dim])

    print "Extraction begins!"

    for i in range(len(test_mat)):
        for wi in range(len(test_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(test_mat[i][wi])
            test_serial_input[i][wi] = word_vec

    test_serial_input = test_serial_input.mean(axis=1)

    print "Extraction ends!"

    #train_xs = np.array([[0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1]])
    #train_ys = np.array([0, 1, 1, 1, 1, 0])
    # Define some constants
    # Inputs are of size 2
    #print type(train_mat[0][0]), train_mat.shape

    
    #serial input (seq of 300 corresponding to each word)
    feat_vec_size = dim
    # Let's use 10 hidden units
    embedding_size = 2.0*feat_vec_size
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    fx = tf.placeholder(tf.float32, feat_vec_size)
    # Other initializers like tf.random_normal_initializer are possible too
    V = tf.get_variable("V", [embedding_size, feat_vec_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    # Can use other nonlinearities: tf.nn.relu, tf.tanh, etc.
    z = tf.tanh(tf.tensordot(V, fx, 1))
    #z = tf.nn.dropout(z, 0.75)
    W = tf.get_variable("W", [num_classes, embedding_size])
    probs = tf.nn.softmax(tf.tensordot(W, z, 1))
    # This is the actual prediction -- not used for training but used for inference
    one_best = tf.argmax(probs)

    # Input for the gold label so we can compute the loss
    label = tf.placeholder(tf.int32, 1)
    # Convert a value-based representation (e.g., [2]) into the one-hot representation ([0, 0, 1])
    # Because label is a tensor of dimension one, the one-hot is actually [[0, 0, 1]], so
    # we need to flatten it.
    label_onehot = tf.reshape(tf.one_hot(label, num_classes), shape=[num_classes])
    loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))


    # TRAINING ALGORITHM CUSTOMIZATION
    # Decay the learning rate by a factor of 0.99 every 10 gradient steps (for larger datasets you'll want a slower
    # weight decay schedule
    decay_steps = 10
    learning_rate_decay_factor = 0.9995
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.005
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    opt = tf.train.AdamOptimizer(lr)
    #opt = tf.train.RMSPropOptimizer(0.005)
    #opt = tf.train.AdagradOptimizer(lr)
    #opt = tf.train.AdamOptimizer()
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    num_epochs = 10
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            print "Epoch:", i
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(serial_input)):
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                # [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {fx: train_xs[ex_idx],
                #                                                                   label: np.array([train_ys[ex_idx]])})
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {fx: serial_input[ex_idx],
                                                                                   label: np.array([train_labels_arr[ex_idx]])})
                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        
        # Evaluate on the train set
            dev_correct = 0
            for ex_idx in xrange(0, len(dev_serial_input)):
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and z
                [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                                  feed_dict={fx: dev_serial_input[ex_idx]})
                if (dev_labels_arr[ex_idx] == pred_this_instance):
                    dev_correct += 1
                # print "Example " + repr(test_serial_input[ex_idx]) + "; gold = " + repr(test_labels_arr[ex_idx]) + "; pred = " +\
                #       repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
                # print "  Hidden layer activations for this example: " + repr(z_this_instance)
            print repr(dev_correct) + "/" + repr(len(dev_labels_arr)) + " correct after training"
            print 1.0*dev_correct/len(dev_labels_arr)
            
        output_lst = []
        for ex_idx in xrange(0, len(test_serial_input)):
            
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                                  feed_dict={fx: test_serial_input[ex_idx]})
            
            test_exs[ex_idx].label = pred_this_instance
        
        return test_exs


# Analogous to train_ffnn, but trains your fancier model.
def train_fancy(train_exs, dev_exs, test_exs, word_embeddings):
    print "HEY"
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])
    print train_labels_arr[5]

    dim = len(word_embeddings.get_embedding_byidx(0))
    serial_input = np.zeros([len(train_exs), seq_max_len, dim])
    
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    # Labels
    test_labels_arr = np.array([ex.label for ex in test_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    test_serial_input = np.zeros([len(test_exs), seq_max_len, dim])

    print "Extraction begins!"

    for i in range(len(test_mat)):
        for wi in range(len(test_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(test_mat[i][wi])
            test_serial_input[i][wi] = word_vec



    print "TRAIN Extraction begins!"

    for i in range(len(train_mat)):
        for wi in range(len(train_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(train_mat[i][wi])
            #for j in range(len(word_vec)):
            serial_input[i][wi] = word_vec

    #serial_input = serial_input.mean(axis=1)

                #print word_vec[j]
                


    print "TRAIN Extraction ends!"

    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    # Labels
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    dev_serial_input = np.zeros([len(dev_exs), seq_max_len, dim])

    print "DEV Extraction begins!"

    for i in range(len(dev_mat)):
        for wi in range(len(dev_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(dev_mat[i][wi])
            dev_serial_input[i][wi] = word_vec

    #dev_serial_input = dev_serial_input.mean(axis=1)

    print "DEV Extraction ends!"
    
    
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    # Labels
    test_labels_arr = np.array([ex.label for ex in test_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    test_serial_input = np.zeros([len(test_exs), seq_max_len, dim])

    print "TEST Extraction begins!"

    for i in range(len(test_mat)):
        for wi in range(len(test_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(test_mat[i][wi])
            test_serial_input[i][wi] = word_vec

    #dev_serial_input = dev_serial_input.mean(axis=1)

    print "TEST Extraction ends!"


    
    #serial input (seq of 300 corresponding to each word)
    batch_size = 1
    feat_vec_size = dim
    # Let's use 10 hidden units
    embedding_size = feat_vec_size
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    #fx = tf.placeholder(tf.float32, feat_vec_size)

    sent = tf.placeholder(tf.float32, [batch_size, seq_max_len, feat_vec_size])
    label = tf.placeholder(tf.int32, 1)
    sent_len = tf.placeholder(tf.int32, 1)

    num_cells = 100

    def myLSTMcell():
        lstm = tf.nn.rnn_cell.LSTMCell(num_cells, reuse=tf.get_variable_scope().reuse)
        return lstm
        #return tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=tf.constant(0.7))

    myLSTMcell = myLSTMcell()

    initialS = myLSTMcell.zero_state(1, tf.float32)
    #sent_input = tf.unstack(sent)
    sent_input = sent
    output, _ = tf.nn.dynamic_rnn(myLSTMcell, sent_input, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)
    print "AKAMATH", output.shape, type(output)
    #z = output[0][-1]
    z = tf.reduce_mean(output[0], axis=0)
    print "Z's shape is ", z.shape
    #print "Hey!", output.shape
    W = tf.get_variable("W", [num_classes, num_cells], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))
    print "HI SEXY", W.shape, z.shape
    #b = tf.get_variable("b", [num_classes])
    #probs = tf.contrib.layers.fully_connected([output[0][-1]], num_classes, activation_fn=tf.nn.softmax)
    #print probs.shape
    #probs = tf.reshape(probs, shape=[1,num_classes])
    probs = tf.nn.softmax(tf.tensordot(W, z, 1))
    one_best = tf.argmax(probs)

    label_onehot = tf.reshape(tf.one_hot(label, num_classes), shape=[num_classes])
    print probs.shape, label_onehot.shape
    

    loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))


    decay_steps = 10
    learning_rate_decay_factor = 0.99995
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.001
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    #opt = tf.train.AdamOptimizer(lr)
    opt = tf.train.AdamOptimizer()
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    num_epochs = 10
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            print "Epoch:", i
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(serial_input)):
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                # [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {fx: train_xs[ex_idx],
                #                                                                   label: np.array([train_ys[ex_idx]])})
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {sent: [serial_input[ex_idx]],
                                                                                   label: np.array([train_labels_arr[ex_idx]]), 
                                                                                   sent_len: np.array([train_seq_lens[ex_idx]])})
                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        
        # Evaluate on the train set
            dev_correct = 0
            for ex_idx in xrange(0, len(dev_serial_input)):
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and z
                [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                                                  feed_dict={sent: [dev_serial_input[ex_idx]],
                                                                                  sent_len: np.array([dev_seq_lens[ex_idx]])})
                if (dev_labels_arr[ex_idx] == pred_this_instance):
                    dev_correct += 1
            # print "Example " + repr(test_serial_input[ex_idx]) + "; gold = " + repr(test_labels_arr[ex_idx]) + "; pred = " +\
            #       repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
            # print "  Hidden layer activations for this example: " + repr(z_this_instance)
            print repr(dev_correct) + "/" + repr(len(dev_labels_arr)) + " correct after training"
            print 1.0*dev_correct/len(dev_labels_arr)
            
        for ex_idx in xrange(0, len(test_serial_input)):
            
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                                  feed_dict={sent: [test_serial_input[ex_idx]]})
            
            test_exs[ex_idx].label = pred_this_instance
        
        return test_exs

def train_fancy2(train_exs, dev_exs, test_exs, word_embeddings):
    print "HEY"
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    print train_mat.shape
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])
    print train_labels_arr[5]

    dim = len(word_embeddings.get_embedding_byidx(0))
    serial_input = np.zeros([len(train_exs), seq_max_len, dim])



    print "TRAIN Extraction begins!"

    for i in range(len(train_mat)):
        for wi in range(len(train_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(train_mat[i][wi])
            #for j in range(len(word_vec)):
            serial_input[i][wi] = word_vec

    #serial_input = serial_input.mean(axis=1)

                #print word_vec[j]
                


    print "TRAIN Extraction ends!"

    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    # Labels
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    dev_serial_input = np.zeros([len(dev_exs), seq_max_len, dim])

    print "DEV Extraction begins!"

    for i in range(len(dev_mat)):
        for wi in range(len(dev_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(dev_mat[i][wi])
            dev_serial_input[i][wi] = word_vec

    #dev_serial_input = dev_serial_input.mean(axis=1)

    print "DEV Extraction ends!"
    
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    # Labels
    test_labels_arr = np.array([ex.label for ex in test_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    test_serial_input = np.zeros([len(test_exs), seq_max_len, dim])

    print "TEST Extraction begins!"

    for i in range(len(test_mat)):
        for wi in range(len(test_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(test_mat[i][wi])
            test_serial_input[i][wi] = word_vec

    #dev_serial_input = dev_serial_input.mean(axis=1)

    print "TEST Extraction ends!"


    
    #serial input (seq of 300 corresponding to each word)
    batch_size = 1
    feat_vec_size = dim
    # Let's use 10 hidden units
    embedding_size = feat_vec_size
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    #fx = tf.placeholder(tf.float32, feat_vec_size)

    sent = tf.placeholder(tf.float32, [batch_size, seq_max_len, feat_vec_size])
    label = tf.placeholder(tf.int32, batch_size)
    sent_len = tf.placeholder(tf.int32, batch_size)

    num_cells = 100
    number_of_layers = 3

    def myLSTMcell():
        lstm = tf.nn.rnn_cell.LSTMCell(num_cells, reuse=tf.get_variable_scope().reuse)
        return tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=tf.constant(0.7))

    #myLSTMcell = myLSTMcell()
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [myLSTMcell() for _ in range(number_of_layers)])
    
    initS = stacked_lstm.zero_state(batch_size, tf.float32)
    
    
    #initialS = myLSTMcell.zero_state(1, tf.float32)
    #sent_input = tf.unstack(sent)
    sent_input = sent
    #print "HEY", sent_input.shape 
    #output, _ = tf.nn.dynamic_rnn(myLSTMcell, sent_input, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)
    #output, _ = stacked_lstm(sent_input, initS)
    
    output, _ = tf.nn.dynamic_rnn(stacked_lstm, sent_input, dtype=tf.float32)#sequence_length=sent_len, dtype=tf.float32)
    print output.shape
    output = tf.transpose(output, perm=[1, 0, 2])
    z = output[-1]
    
    print "Z's shape is ", z.shape
    z = tf.transpose(z)
    print "After transpose", z.shape
    #print "Hey!", output.shape
    W = tf.get_variable("W", [num_classes, num_cells], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))
    #b = tf.get_variable("b", [num_classes])
    #probs = tf.contrib.layers.fully_connected([output[0][-1]], num_classes, activation_fn=tf.nn.softmax)
    #print probs.shape
    #probs = tf.reshape(probs, shape=[1,num_classes])
    probs = tf.nn.softmax(tf.tensordot(W, z, 1))
    probs = tf.transpose(probs)
    one_best = tf.argmax(probs)

    label_onehot = tf.reshape(tf.one_hot(label, num_classes), shape=[num_classes, batch_size])
    print label_onehot.shape

    

    loss = tf.negative(tf.log(tf.reduce_sum(tf.tensordot(probs, label_onehot, 1))))


    decay_steps = 10
    learning_rate_decay_factor = 0.99995
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.001
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    #opt = tf.train.AdamOptimizer(lr)
    opt = tf.train.AdamOptimizer()
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    num_epochs = 10
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            print "Epoch:", i
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(serial_input)):
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                # [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {fx: train_xs[ex_idx],
                #                                                                   label: np.array([train_ys[ex_idx]])})
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {sent: [serial_input[ex_idx]],
                                                                                   label: np.array([train_labels_arr[ex_idx]]), 
                                                                                   sent_len: np.array([train_seq_lens[ex_idx]])})
                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)

            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            #for ex_idx in xrange(0, len(serial_input)):
#            while epoch_step*batch_size < len(serial_input):
#                try:
#                    print "Im trying!"
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                # [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {fx: train_xs[ex_idx],
                #                                                                   label: np.array([train_ys[ex_idx]])})
##                    [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {sent: serial_input[epoch_step*batch_size: (epoch_step+1)*batch_size],
##                                                                                   label: np.array(train_labels_arr[epoch_step*batch_size: (epoch_step+1)*batch_size]), 
##                                                                                   sent_len: np.array(train_seq_lens[epoch_step*batch_size: (epoch_step+1)*batch_size])})
#                    step_idx += 1
#                    epoch_step += 1
#                    loss_this_iter += loss_this_instance
                    
                #print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
                
#                except:
#                    print len(serial_input[epoch_step*batch_size: (epoch_step+1)*batch_size])
            
            ###
            ##
        
        # Evaluate on the train set
            dev_correct = 0
            for ex_idx in xrange(0, len(dev_serial_input)):
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and z
                [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                                                  feed_dict={sent: [dev_serial_input[ex_idx]],
                                                                                  sent_len: np.array([dev_seq_lens[ex_idx]])})
                if (dev_labels_arr[ex_idx] == pred_this_instance):
                    dev_correct += 1
            # print "Example " + repr(test_serial_input[ex_idx]) + "; gold = " + repr(test_labels_arr[ex_idx]) + "; pred = " +\
            #       repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
            # print "  Hidden layer activations for this example: " + repr(z_this_instance)
            print repr(dev_correct) + "/" + repr(len(dev_labels_arr)) + " correct after training"
            print 1.0*dev_correct/len(dev_labels_arr)
            
        for ex_idx in xrange(0, len(test_serial_input)):
            
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                                  feed_dict={sent: [test_serial_input[ex_idx]]})
            
            test_exs[ex_idx].label = pred_this_instance
        
        return test_exs
            
def train_fancy6(train_exs, dev_exs, test_exs, word_embeddings):
    print "HEY"
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])
    print train_labels_arr[5]

    dim = len(word_embeddings.get_embedding_byidx(0))
    serial_input = np.zeros([len(train_exs), seq_max_len, dim])



    print "TRAIN Extraction begins!"

    for i in range(len(train_mat)):
        for wi in range(len(train_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(train_mat[i][wi])
            #for j in range(len(word_vec)):
            serial_input[i][wi] = word_vec

    #serial_input = serial_input.mean(axis=1)

                #print word_vec[j]
                


    print "TRAIN Extraction ends!"

    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    # Labels
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    dev_serial_input = np.zeros([len(dev_exs), seq_max_len, dim])

    print "DEV Extraction begins!"

    for i in range(len(dev_mat)):
        for wi in range(len(dev_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(dev_mat[i][wi])
            dev_serial_input[i][wi] = word_vec

    #dev_serial_input = dev_serial_input.mean(axis=1)

    print "DEV Extraction ends!"
    
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    # Labels
    test_labels_arr = np.array([ex.label for ex in test_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    test_serial_input = np.zeros([len(test_exs), seq_max_len, dim])

    print "TEST Extraction begins!"

    for i in range(len(test_mat)):
        for wi in range(len(test_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(test_mat[i][wi])
            test_serial_input[i][wi] = word_vec

    #dev_serial_input = dev_serial_input.mean(axis=1)

    print "TEST Extraction ends!"


    
    #serial input (seq of 300 corresponding to each word)
    batch_size = 1
    feat_vec_size = dim
    # Let's use 10 hidden units
    embedding_size = feat_vec_size
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    #fx = tf.placeholder(tf.float32, feat_vec_size)

    sent = tf.placeholder(tf.float32, [batch_size, seq_max_len, feat_vec_size])
    label = tf.placeholder(tf.int32, 1)
    sent_len = tf.placeholder(tf.int32, 1)
    num_layers = 3

    num_cells = 100

    def myLSTMcell():
        lstm = tf.nn.rnn_cell.LSTMCell(num_cells, reuse=tf.get_variable_scope().reuse)
        return lstm
        #return tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=tf.constant(0.7))

    #myLSTMcell = myLSTMcell()
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [myLSTMcell() for _ in range(num_layers)])

    initialS = myLSTMcell().zero_state(1, tf.float32)
    #sent_input = tf.unstack(sent)
    sent_input = sent
    output, _ = tf.nn.dynamic_rnn(stacked_lstm, sent_input, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)
    print "AKAMATH", output.shape, type(output)
    #print output.shape
    #output = tf.transpose(output, perm=[1, 0, 2])
    #print output.shape, output[0].shape
    #z = output[0][-1]
    z = tf.reduce_mean(output[0], axis=0)
    print "Z's shape is ", z.shape
    #print "Hey!", output.shape
    W = tf.get_variable("W", [num_classes, num_cells], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))
    print "HI SEXY", W.shape, z.shape
    #b = tf.get_variable("b", [num_classes])
    #probs = tf.contrib.layers.fully_connected([output[0][-1]], num_classes, activation_fn=tf.nn.softmax)
    #print probs.shape
    #probs = tf.reshape(probs, shape=[1,num_classes])
    probs = tf.nn.softmax(tf.tensordot(W, z, 1))
    one_best = tf.argmax(probs)

    label_onehot = tf.reshape(tf.one_hot(label, num_classes), shape=[num_classes])
    print probs.shape, label_onehot.shape
    

    loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))


    decay_steps = 10
    learning_rate_decay_factor = 0.99995
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.001
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    #opt = tf.train.AdamOptimizer(lr)
    opt = tf.train.AdamOptimizer()
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    num_epochs = 10
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            print "Epoch:", i
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(serial_input)):
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                # [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {fx: train_xs[ex_idx],
                #                                                                   label: np.array([train_ys[ex_idx]])})
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {sent: [serial_input[ex_idx]],
                                                                                   label: np.array([train_labels_arr[ex_idx]]), 
                                                                                   sent_len: np.array([train_seq_lens[ex_idx]])})
                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        
        # Evaluate on the train set
            dev_correct = 0
            for ex_idx in xrange(0, len(dev_serial_input)):
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and z
                [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                                                  feed_dict={sent: [dev_serial_input[ex_idx]],
                                                                                  sent_len: np.array([dev_seq_lens[ex_idx]])})
                if (dev_labels_arr[ex_idx] == pred_this_instance):
                    dev_correct += 1
            # print "Example " + repr(test_serial_input[ex_idx]) + "; gold = " + repr(test_labels_arr[ex_idx]) + "; pred = " +\
            #       repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
            # print "  Hidden layer activations for this example: " + repr(z_this_instance)
            print repr(dev_correct) + "/" + repr(len(dev_labels_arr)) + " correct after training"
            print 1.0*dev_correct/len(dev_labels_arr)

        for ex_idx in xrange(0, len(test_serial_input)):
            
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                                  feed_dict={sent: [test_serial_input[ex_idx]]})
            
            test_exs[ex_idx].label = pred_this_instance
        
        return test_exs

            
def train_fancy3(train_exs, dev_exs, test_exs, word_embeddings):
    print "HEY"
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])
    print train_labels_arr[5]

    dim = len(word_embeddings.get_embedding_byidx(0))
    serial_input = np.zeros([len(train_exs), seq_max_len, dim])



    print "TRAIN Extraction begins!"

    for i in range(len(train_mat)):
        for wi in range(len(train_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(train_mat[i][wi])
            #for j in range(len(word_vec)):
            serial_input[i][wi] = word_vec

    #serial_input = serial_input.mean(axis=1)

                #print word_vec[j]
                


    print "TRAIN Extraction ends!"

    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    # Labels
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    dev_serial_input = np.zeros([len(dev_exs), seq_max_len, dim])

    print "DEV Extraction begins!"

    for i in range(len(dev_mat)):
        for wi in range(len(dev_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(dev_mat[i][wi])
            dev_serial_input[i][wi] = word_vec

    #dev_serial_input = dev_serial_input.mean(axis=1)

    print "DEV Extraction ends!"
    
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    # Labels
    test_labels_arr = np.array([ex.label for ex in test_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    test_serial_input = np.zeros([len(test_exs), seq_max_len, dim])

    print "TEST Extraction begins!"

    for i in range(len(test_mat)):
        for wi in range(len(test_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(test_mat[i][wi])
            test_serial_input[i][wi] = word_vec

    #dev_serial_input = dev_serial_input.mean(axis=1)

    print "TEST Extraction ends!"


    
    #serial input (seq of 300 corresponding to each word)
    batch_size = 1
    feat_vec_size = dim
    # Let's use 10 hidden units
    embedding_size = feat_vec_size
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    #fx = tf.placeholder(tf.float32, feat_vec_size)

    sent = tf.placeholder(tf.float32, [batch_size, seq_max_len, feat_vec_size])
    label = tf.placeholder(tf.int32, 1)
    sent_len = tf.placeholder(tf.int32, 1)

    num_cells = 100

    def myLSTMcell():
        lstm = tf.nn.rnn_cell.LSTMCell(num_cells, reuse=tf.get_variable_scope().reuse)
        return lstm
        #return tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=tf.constant(0.7))

    myLSTMcell1 = myLSTMcell()
    myLSTMcell2 = myLSTMcell()

    #initialS = myLSTMcell.zero_state(1, tf.float32)
    #sent_input = tf.unstack(sent)
    sent_input = sent
    output, _ = tf.nn.bidirectional_dynamic_rnn(myLSTMcell1, myLSTMcell2, sent_input, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)
    print "Anikesh", type(output)
    print output[0].shape, output[1].shape
    #z1 = output[0][0][-1]
    
    z1 = tf.reduce_mean(output[0][0], axis=0)
    #z2 = output[1][0][-1]
    z2 = tf.reduce_mean(output[1][0], axis=0)
    print z1.shape, z2.shape
    z = tf.concat([z1, z2], 0)
    #z = tf.reduce_mean(tf.concat([output[0], output[1]], 0), axis=0)
    print "Z's shape is ", z.shape
    #print "Hey!", output.shape
    W = tf.get_variable("W", [num_classes, num_cells*2], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))
    #b = tf.get_variable("b", [num_classes])
    #probs = tf.contrib.layers.fully_connected([output[0][-1]], num_classes, activation_fn=tf.nn.softmax)
    #print probs.shape
    #probs = tf.reshape(probs, shape=[1,num_classes])
    probs = tf.nn.softmax(tf.tensordot(W, z, 1))
    one_best = tf.argmax(probs)

    label_onehot = tf.reshape(tf.one_hot(label, num_classes), shape=[num_classes])
    print probs.shape, label_onehot.shape
    

    loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))


    decay_steps = 10
    learning_rate_decay_factor = 0.995
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.005
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    opt = tf.train.AdamOptimizer()
    #opt = tf.train.AdamOptimizer()
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    num_epochs = 10
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            print "Epoch:", i
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(serial_input)):
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                # [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {fx: train_xs[ex_idx],
                #                                                                   label: np.array([train_ys[ex_idx]])})
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {sent: [serial_input[ex_idx]],
                                                                                   label: np.array([train_labels_arr[ex_idx]]), 
                                                                                   sent_len: np.array([train_seq_lens[ex_idx]])})
                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        
        # Evaluate on the train set
            dev_correct = 0
            for ex_idx in xrange(0, len(dev_serial_input)):
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and z
                [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                                                  feed_dict={sent: [dev_serial_input[ex_idx]],
                                                                                  sent_len: np.array([dev_seq_lens[ex_idx]])})
                if (dev_labels_arr[ex_idx] == pred_this_instance):
                    dev_correct += 1
            # print "Example " + repr(test_serial_input[ex_idx]) + "; gold = " + repr(test_labels_arr[ex_idx]) + "; pred = " +\
            #       repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
            # print "  Hidden layer activations for this example: " + repr(z_this_instance)
            print repr(dev_correct) + "/" + repr(len(dev_labels_arr)) + " correct after training"
            print 1.0*dev_correct/len(dev_labels_arr)
            
        for ex_idx in xrange(0, len(test_serial_input)):
            
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                                  feed_dict={sent: [test_serial_input[ex_idx]]})
            
            test_exs[ex_idx].label = pred_this_instance
        
        return test_exs

def train_fancy7(train_exs, dev_exs, test_exs, word_embeddings):
    print "HEY"
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])
    print train_labels_arr[5]

    dim = len(word_embeddings.get_embedding_byidx(0))
    serial_input = np.zeros([len(train_exs), seq_max_len, dim])



    print "TRAIN Extraction begins!"

    for i in range(len(train_mat)):
        for wi in range(len(train_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(train_mat[i][wi])
            #for j in range(len(word_vec)):
            serial_input[i][wi] = word_vec

    #serial_input = serial_input.mean(axis=1)

                #print word_vec[j]
                


    print "TRAIN Extraction ends!"

    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    # Labels
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    dev_serial_input = np.zeros([len(dev_exs), seq_max_len, dim])

    print "DEV Extraction begins!"

    for i in range(len(dev_mat)):
        for wi in range(len(dev_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(dev_mat[i][wi])
            dev_serial_input[i][wi] = word_vec

    #dev_serial_input = dev_serial_input.mean(axis=1)

    print "DEV Extraction ends!"
    
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    # Labels
    test_labels_arr = np.array([ex.label for ex in test_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    test_serial_input = np.zeros([len(test_exs), seq_max_len, dim])

    print "TEST Extraction begins!"

    for i in range(len(test_mat)):
        for wi in range(len(test_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(test_mat[i][wi])
            test_serial_input[i][wi] = word_vec

    #dev_serial_input = dev_serial_input.mean(axis=1)

    print "TEST Extraction ends!"


    
    #serial input (seq of 300 corresponding to each word)
    batch_size = 1
    feat_vec_size = dim
    # Let's use 10 hidden units
    embedding_size = feat_vec_size
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    #fx = tf.placeholder(tf.float32, feat_vec_size)

    sent = tf.placeholder(tf.float32, [batch_size, seq_max_len, feat_vec_size])
    label = tf.placeholder(tf.int32, 1)
    sent_len = tf.placeholder(tf.int32, 1)
    num_layers = 3

    num_cells = 100

    def myLSTMcell():
        lstm = tf.nn.rnn_cell.LSTMCell(num_cells, reuse=tf.get_variable_scope().reuse)
        return lstm
        #return tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=tf.constant(0.7))

    stacked_lstm1 = tf.contrib.rnn.MultiRNNCell(
        [myLSTMcell() for _ in range(num_layers)])
    stacked_lstm2 = tf.contrib.rnn.MultiRNNCell(
        [myLSTMcell() for _ in range(num_layers)])

    #initialS = myLSTMcell.zero_state(1, tf.float32)
    #sent_input = tf.unstack(sent)
    sent_input = sent
    output, _ = tf.nn.bidirectional_dynamic_rnn(stacked_lstm1, stacked_lstm2, sent_input, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)
    print "Anikesh", type(output)
    print len(output)
    print output[0].shape, output[1].shape
    #z1 = output[0][0][-1]
    
    z1 = tf.reduce_mean(output[0][0], axis=0)
    #z2 = output[1][0][-1]
    z2 = tf.reduce_mean(output[1][0], axis=0)
    print z1.shape, z2.shape
    z = tf.concat([z1, z2], 0)
    #z = tf.reduce_mean(tf.concat([output[0], output[1]], 0), axis=0)
    print "Z's shape is ", z.shape
    #print "Hey!", output.shape
    W = tf.get_variable("W", [num_classes, num_cells*2], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))
    #b = tf.get_variable("b", [num_classes])
    #probs = tf.contrib.layers.fully_connected([output[0][-1]], num_classes, activation_fn=tf.nn.softmax)
    #print probs.shape
    #probs = tf.reshape(probs, shape=[1,num_classes])
    probs = tf.nn.softmax(tf.tensordot(W, z, 1))
    one_best = tf.argmax(probs)

    label_onehot = tf.reshape(tf.one_hot(label, num_classes), shape=[num_classes])
    print probs.shape, label_onehot.shape
    

    loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))


    decay_steps = 10
    learning_rate_decay_factor = 0.995
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.005
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    opt = tf.train.AdamOptimizer()
    #opt = tf.train.AdamOptimizer()
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    num_epochs = 10
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            print "Epoch:", i
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(serial_input)):
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                # [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {fx: train_xs[ex_idx],
                #                                                                   label: np.array([train_ys[ex_idx]])})
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {sent: [serial_input[ex_idx]],
                                                                                   label: np.array([train_labels_arr[ex_idx]]), 
                                                                                   sent_len: np.array([train_seq_lens[ex_idx]])})
                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        
        # Evaluate on the train set
            dev_correct = 0
            for ex_idx in xrange(0, len(dev_serial_input)):
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and z
                [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                                                  feed_dict={sent: [dev_serial_input[ex_idx]],
                                                                                  sent_len: np.array([dev_seq_lens[ex_idx]])})
                if (dev_labels_arr[ex_idx] == pred_this_instance):
                    dev_correct += 1
            # print "Example " + repr(test_serial_input[ex_idx]) + "; gold = " + repr(test_labels_arr[ex_idx]) + "; pred = " +\
            #       repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
            # print "  Hidden layer activations for this example: " + repr(z_this_instance)
            print repr(dev_correct) + "/" + repr(len(dev_labels_arr)) + " correct after training"
            print 1.0*dev_correct/len(dev_labels_arr)
            
        for ex_idx in xrange(0, len(test_serial_input)):
            
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                                  feed_dict={sent: [test_serial_input[ex_idx]]})
            
            test_exs[ex_idx].label = pred_this_instance
        
        return test_exs


def train_fancy4(train_exs, dev_exs, test_exs, word_embeddings):
    print "HEY"
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])
    print train_labels_arr[5]

    dim = len(word_embeddings.get_embedding_byidx(0))
    serial_input = np.zeros([len(train_exs), seq_max_len, dim])



    print "TRAIN Extraction begins!"

    for i in range(len(train_mat)):
        for wi in range(len(train_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(train_mat[i][wi])
            #for j in range(len(word_vec)):
            serial_input[i][wi] = word_vec

    #serial_input = serial_input.mean(axis=1)

                #print word_vec[j]
                


    print "TRAIN Extraction ends!"

    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    # Labels
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    dev_serial_input = np.zeros([len(dev_exs), seq_max_len, dim])

    print "DEV Extraction begins!"

    for i in range(len(dev_mat)):
        for wi in range(len(dev_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(dev_mat[i][wi])
            dev_serial_input[i][wi] = word_vec

    #dev_serial_input = dev_serial_input.mean(axis=1)

    print "DEV Extraction ends!"
    
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    # Labels
    test_labels_arr = np.array([ex.label for ex in test_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    test_serial_input = np.zeros([len(test_exs), seq_max_len, dim])

    print "TEST Extraction begins!"

    for i in range(len(test_mat)):
        for wi in range(len(test_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(test_mat[i][wi])
            test_serial_input[i][wi] = word_vec

    #dev_serial_input = dev_serial_input.mean(axis=1)

    print "TEST Extraction ends!"


    
    #serial input (seq of 300 corresponding to each word)
    batch_size = 1
    feat_vec_size = dim
    # Let's use 10 hidden units
    embedding_size = feat_vec_size
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    #fx = tf.placeholder(tf.float32, feat_vec_size)

    sent = tf.placeholder(tf.float32, [batch_size, seq_max_len, feat_vec_size])
    filter1 = tf.zeros([1, 300, 5])
    
    label = tf.placeholder(tf.int32, 1)
    sent_len = tf.placeholder(tf.int32, 1)

    num_cells = 100

    def myLSTMcell():
        lstm = tf.nn.rnn_cell.LSTMCell(num_cells, reuse=tf.get_variable_scope().reuse)
        return tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=tf.constant(0.7))

    myLSTMcell1 = myLSTMcell()
    myLSTMcell2 = myLSTMcell()

    #initialS = myLSTMcell.zero_state(1, tf.float32)
    #sent_input = tf.unstack(sent)
    sent_input = sent
    
    print sent_input.shape, filter1.shape
    
    output, _ = tf.nn.conv1d(sent_input, filter1, strides=1,padding='VALID') #sequence_length=sent_len, dtype=tf.float32)
    print "Anikesh", type(output)
    print output.shape
    #z1 = output[0][0][-1]
    #z2 = output[1][0][-1]
    #z = tf.concat([z1, z2], 0)
    print "Z's shape is ", z.shape
    #print "Hey!", output.shape
    W = tf.get_variable("W", [num_classes, num_cells*2], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))
    #b = tf.get_variable("b", [num_classes])
    #probs = tf.contrib.layers.fully_connected([output[0][-1]], num_classes, activation_fn=tf.nn.softmax)
    #print probs.shape
    #probs = tf.reshape(probs, shape=[1,num_classes])
    probs = tf.nn.softmax(tf.tensordot(W, z, 1))
    one_best = tf.argmax(probs)

    label_onehot = tf.reshape(tf.one_hot(label, num_classes), shape=[num_classes])
    print probs.shape, label_onehot.shape
    

    loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))


    decay_steps = 10
    learning_rate_decay_factor = 0.99995
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.001
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    #opt = tf.train.AdamOptimizer(lr)
    opt = tf.train.AdamOptimizer()
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    num_epochs = 10
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            print "Epoch:", i
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(serial_input)):
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                # [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {fx: train_xs[ex_idx],
                #                                                                   label: np.array([train_ys[ex_idx]])})
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {sent: [serial_input[ex_idx]],
                                                                                   label: np.array([train_labels_arr[ex_idx]]), 
                                                                                   sent_len: np.array([train_seq_lens[ex_idx]])})
                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        
        # Evaluate on the train set
            dev_correct = 0
            for ex_idx in xrange(0, len(dev_serial_input)):
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and z
                [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                                                  feed_dict={sent: [dev_serial_input[ex_idx]],
                                                                                  sent_len: np.array([dev_seq_lens[ex_idx]])})
                if (dev_labels_arr[ex_idx] == pred_this_instance):
                    dev_correct += 1
            # print "Example " + repr(test_serial_input[ex_idx]) + "; gold = " + repr(test_labels_arr[ex_idx]) + "; pred = " +\
            #       repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
            # print "  Hidden layer activations for this example: " + repr(z_this_instance)
            print repr(dev_correct) + "/" + repr(len(dev_labels_arr)) + " correct after training"
            print 1.0*dev_correct/len(dev_labels_arr)
            
        for ex_idx in xrange(0, len(test_serial_input)):
            
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                                  feed_dict={sent: [test_serial_input[ex_idx]]})
            
            test_exs[ex_idx].label = pred_this_instance
        
        return test_exs
            
def train_fancy5(train_exs, dev_exs, test_exs, word_embeddings):
    print "HEY"
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])
    print train_labels_arr[5]

    dim = len(word_embeddings.get_embedding_byidx(0))
    serial_input = np.zeros([len(train_exs), seq_max_len, dim])



    print "TRAIN Extraction begins!"

    for i in range(len(train_mat)):
        for wi in range(len(train_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(train_mat[i][wi])
            #for j in range(len(word_vec)):
            serial_input[i][wi] = word_vec

    #serial_input = serial_input.mean(axis=1)

                #print word_vec[j]
                


    print "TRAIN Extraction ends!"

    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    # Labels
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    dev_serial_input = np.zeros([len(dev_exs), seq_max_len, dim])

    print "DEV Extraction begins!"

    for i in range(len(dev_mat)):
        for wi in range(len(dev_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(dev_mat[i][wi])
            dev_serial_input[i][wi] = word_vec

    #dev_serial_input = dev_serial_input.mean(axis=1)

    print "DEV Extraction ends!"
    
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    # Labels
    test_labels_arr = np.array([ex.label for ex in test_exs])

    dim = len(word_embeddings.get_embedding_byidx(0))
    test_serial_input = np.zeros([len(test_exs), seq_max_len, dim])

    print "TEST Extraction begins!"

    for i in range(len(test_mat)):
        for wi in range(len(test_mat[i])):
            word_vec = word_embeddings.get_embedding_byidx(test_mat[i][wi])
            test_serial_input[i][wi] = word_vec

    #dev_serial_input = dev_serial_input.mean(axis=1)

    print "TEST Extraction ends!"


    
    #serial input (seq of 300 corresponding to each word)
    batch_size = 1
    feat_vec_size = dim
    # Let's use 10 hidden units
    embedding_size = feat_vec_size
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    #fx = tf.placeholder(tf.float32, feat_vec_size)

    sent = tf.placeholder(tf.float32, [batch_size, seq_max_len, feat_vec_size])
    label = tf.placeholder(tf.int32, 1)
    sent_len = tf.placeholder(tf.int32, 1)

    num_cells = 100

    def myLSTMcell():
        lstm = tf.nn.rnn_cell.LSTMCell(num_cells, reuse=tf.get_variable_scope().reuse)
        return tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=tf.constant(0.7))

    myLSTMcell1 = myLSTMcell()
    myLSTMcell2 = myLSTMcell()

    #initialS = myLSTMcell.zero_state(1, tf.float32)
    #sent_input = tf.unstack(sent)
    sent_input = sent
    output, _ = tf.nn.bidirectional_dynamic_rnn(myLSTMcell1, myLSTMcell2, sent_input, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)
    print "Anikesh", type(output)
    print output[0].shape, output[1].shape
    #z1 = output[0][0][-1]
    
    #z1 = tf.reduce_mean(output[0][0], axis=0)
    #z2 = output[1][0][-1]
    #z2 = tf.reduce_mean(output[1][0], axis=0)
    
    z1 = tf.concat([output[0][0], output[1][0]], 1)
    
    def obtain_max_norm(ten):
        #t_t = tf.transpose(ten)
        max_norm_ind = 0
        #print type(ten[max_norm_ind])
        max_mag = tf.norm(ten[max_norm_ind])
        #print max_mag
        #print max_mag.shape, type(max_mag[0])
        for i in range(1,ten.shape[0]):
            if tf.less(max_mag, ten[max_norm_ind]) is not None:
                max_norm_ind = i
                max_mag = tf.norm(ten[max_norm_ind])
        return ten[max_norm_ind]
    
    #z1 = obtain_max_norm(output[0][0])
    #z2 = obtain_max_norm(output[1][0])
    
    #print z1.shape, z2.shape
    print z1.shape
    z = obtain_max_norm(z1)
    #z = tf.reduce_mean(tf.concat([output[0], output[1]], 0), axis=0)
    print "Z's shape is ", z.shape
    #print "Hey!", output.shape
    W = tf.get_variable("W", [num_classes, num_cells*2], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))
    #b = tf.get_variable("b", [num_classes])
    #probs = tf.contrib.layers.fully_connected([output[0][-1]], num_classes, activation_fn=tf.nn.softmax)
    #print probs.shape
    #probs = tf.reshape(probs, shape=[1,num_classes])
    probs = tf.nn.softmax(tf.tensordot(W, z, 1))
    one_best = tf.argmax(probs)

    label_onehot = tf.reshape(tf.one_hot(label, num_classes), shape=[num_classes])
    print probs.shape, label_onehot.shape
    

    loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))


    decay_steps = 10
    learning_rate_decay_factor = 0.995
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.005
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    #opt = tf.train.AdamOptimizer(lr)
    opt = tf.train.AdamOptimizer()
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    num_epochs = 10
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            print "Epoch:", i
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(serial_input)):
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                # [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {fx: train_xs[ex_idx],
                #                                                                   label: np.array([train_ys[ex_idx]])})
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {sent: [serial_input[ex_idx]],
                                                                                   label: np.array([train_labels_arr[ex_idx]]), 
                                                                                   sent_len: np.array([train_seq_lens[ex_idx]])})
                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        
        # Evaluate on the train set
            dev_correct = 0
            for ex_idx in xrange(0, len(dev_serial_input)):
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and z
                [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                                                  feed_dict={sent: [dev_serial_input[ex_idx]],
                                                                                  sent_len: np.array([dev_seq_lens[ex_idx]])})
                if (dev_labels_arr[ex_idx] == pred_this_instance):
                    dev_correct += 1
            # print "Example " + repr(test_serial_input[ex_idx]) + "; gold = " + repr(test_labels_arr[ex_idx]) + "; pred = " +\
            #       repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
            # print "  Hidden layer activations for this example: " + repr(z_this_instance)
            print repr(dev_correct) + "/" + repr(len(dev_labels_arr)) + " correct after training"
            print 1.0*dev_correct/len(dev_labels_arr)
            
        for ex_idx in xrange(0, len(test_serial_input)):
            
                # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                                  feed_dict={sent: [test_serial_input[ex_idx]]})
            
            test_exs[ex_idx].label = pred_this_instance
        
        return test_exs





