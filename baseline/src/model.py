# models.py

import tensorflow as tf
import numpy as np
import random
from data_gen import *
import random
#from tf.nn.rnn_cell import LSTMCell


# Returns a new numpy array with the data from np_arr padded to be of length length. If length is less than the
# length of the base array, truncates instead.
def pad_to_length(np_arr, length):
    result = np.ones(length)*(-1)
    result[0:np_arr.shape[0]] = np_arr
    return result

def pad(seq, length):
    seq = np.asarray(seq)

    if length < np.size(seq, 0):
        return seq[:length]
    result = np.zeros(length)
    result[0:seq.shape[0]] = seq
    return result


def train_bench1(train_exs, test_exs, word_embeddings, initial_learning_rate = 0.01, learning_rate_decay_factor=0.995):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 237 words to make it a square matrix.

    #TRAINING DATA
    print "TRAIN Extraction begins!"
    print len(train_exs)

    ## Matrix with seq word indices and word vectors
    dim = len(word_embeddings.get_embedding_byidx(0))
    #trainq1_s_input = np.zeros([len(train_exs), seq_max_len, dim])
    #trainq2_s_input = np.zeros([len(train_exs), seq_max_len, dim])

    _q1 = tf.placeholder(tf.int32, [None, seq_max_len])
    _q2 = tf.placeholder(tf.int32, [None, seq_max_len])
    label = tf.placeholder(tf.int32, None)
    q1_len = tf.placeholder(tf.int32, None)
    q2_len = tf.placeholder(tf.int32, None)

    embeddings = tf.Variable(word_embeddings.vectors)
    q1 = tf.cast(tf.nn.embedding_lookup(embeddings, _q1), tf.float32)
    q2 = tf.cast(tf.nn.embedding_lookup(embeddings, _q2), tf.float32)

    batch_size = 64
    feat_vec_size = dim

    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # DEFINING THE COMPUTATION GRAPH

    num_cells = 100

    def myLSTMcell(Preuse):
        lstm = tf.nn.rnn_cell.LSTMCell(num_cells, reuse=Preuse) #tf.get_variable_scope().reuse)
        return lstm

    output1, _ = tf.nn.dynamic_rnn(myLSTMcell(tf.get_variable_scope().reuse), q1, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)
    output2, _ = tf.nn.dynamic_rnn(myLSTMcell(True), q2, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)

    print output1.shape, output2.shape

    #on the basis of conclusions from last assignment, we use the mean vector instead of the last vector
    z1 = tf.reduce_mean(output1, axis=1)
    z2 = tf.reduce_mean(output2, axis=1)
    
    #combining the LSTM representation of the 2 questions
    z = tf.concat([z1, z2], 1)

    W = tf.get_variable("W", [num_cells*2, num_classes], 
        initializer=tf.contrib.layers.xavier_initializer())#seed=0))
    probs = tf.nn.softmax(tf.tensordot(z, W, 1))
    one_best = tf.argmax(probs, axis=1)
    label_onehot = tf.one_hot(label, num_classes)
    
    loss = tf.losses.softmax_cross_entropy(label_onehot, probs)

    decay_steps = 100
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    #initial_learning_rate = 0.01
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
    name = random.randint(0, 4242)
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        #train_writer = tf.summary.FileWriter('../logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
	#saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        #saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
        for i in range(0, num_epochs):
            step_idx = 0
	    #saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
            print "Epoch:", i
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(train_exs)/batch_size):
                if step_idx % 100 == 0:
                    print 'step:', step_idx
                q1_ = []
                q2_ = []
                label_ = []
                q1_sq_len_ = []
                q2_sq_len_ = []

                for b in xrange(0, batch_size):
                    curr_idx = ex_idx * batch_size + b
                    q1_.append(pad(train_exs[curr_idx].indexed_q1, seq_max_len))
                    q2_.append(pad(train_exs[curr_idx].indexed_q2, seq_max_len))
                    label_.append(train_exs[curr_idx].label)
                    q1_sq_len_.append(len(train_exs[curr_idx].indexed_q1))
                    q2_sq_len_.append(len(train_exs[curr_idx].indexed_q2))
                
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {_q1: q1_,
                                                                                    _q2: q2_,
                                                                                   label: np.array(label_),
                                                                                   q2_len: np.array(q2_sq_len_), 
                                                                                   q1_len: np.array(q1_sq_len_)})

                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        
        # Evaluate on the test set
            test_correct = 0
            batch_test = 100
            if i == num_epochs - 1:
                f = open(str(name)+"bench1.txt", "w+")
		f1 = open("gold_labels.txt", "w+")

            for ex_idx in xrange(0, len(test_exs)/batch_test):
                q1_ = []
                q2_ = []
                label_ = []
                q1_sq_len_ = []
                q2_sq_len_ = []

                for b in xrange(0, batch_test):
                    curr_idx = ex_idx * batch_test + b
                    q1_.append(pad(test_exs[curr_idx].indexed_q1, seq_max_len))
                    q2_.append(pad(test_exs[curr_idx].indexed_q2, seq_max_len))
                    label_.append(test_exs[curr_idx].label)
                    q1_sq_len_.append(len(test_exs[curr_idx].indexed_q1))
                    q2_sq_len_.append(len(test_exs[curr_idx].indexed_q2))
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different[word_embeddings.get_embedding_byidx(testQ1_mat[ex_idx])]
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
                [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                  feed_dict={_q1: q1_, #[testq1_s_input[ex_idx]],
                                                _q2: q2_,
                                                   label: np.array(label_),
                                                                                               q2_len: np.array(q2_sq_len_), 
                                                                                               q1_len: np.array(q1_sq_len_)}) 
                for b in xrange(0, len(pred_this_instance)):
                    curr_idx = ex_idx * batch_test + b
                    if (test_exs[curr_idx].label == pred_this_instance[b]):
                        test_correct += 1

                    if i == num_epochs - 1:
                        f.write(str(pred_this_instance[b]))
			f1.write(str(test_exs[curr_idx].label))



            str1 =  repr(test_correct) + "/" + repr(len(test_exs)) + " correct after training"
            str2 =  1.0*test_correct/len(test_exs)
            print str1
            print str2

            if i == num_epochs - 1:
                f.close()
		f1.close()
	#saver.save(sess, str(name)+'bench1_epoch', global_step=10)


def train_bench2(train_exs, test_exs, word_embeddings, initial_learning_rate = 0.01, learning_rate_decay_factor=0.995):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 237 words to make it a square matrix.

    #TRAINING DATA
    print "TRAIN Extraction begins!"
    print len(train_exs)


    ## Matrix with seq word indices and word vectors
    dim = len(word_embeddings.get_embedding_byidx(0))


    ## Matrix with seq word indices and word vectors
    dim = len(word_embeddings.get_embedding_byidx(0))
    #serial input (seq of 300 corresponding to each word)
    
    batch_size = 64
    feat_vec_size = dim
    hidden_dim = 30
    num_classes = 2
    num_cells = 100

    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    

    # DEFINING THE COMPUTATION GRAPH

    q1 = tf.placeholder(tf.float32, [None, seq_max_len, dim])
    q2 = tf.placeholder(tf.float32, [None, seq_max_len, dim])
    label = tf.placeholder(tf.int32, None)
    q1_len = tf.placeholder(tf.int32, None)
    q2_len = tf.placeholder(tf.int32, None)

    

    def myLSTMcell(Preuse):
        lstm = tf.nn.rnn_cell.LSTMCell(num_cells, reuse=Preuse) #tf.get_variable_scope().reuse)
        return lstm

    output1, _ = tf.nn.dynamic_rnn(myLSTMcell(tf.get_variable_scope().reuse), q1, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)
    output2, _ = tf.nn.dynamic_rnn(myLSTMcell(True), q2, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)

    print output1.shape, output2.shape

    #on the basis of conclusions from last assignment, we use the mean vector instead of the last vector
    z1 = tf.reduce_mean(output1, axis=1)
    z2 = tf.reduce_mean(output2, axis=1)

    #combining the LSTM representation of the 2 questions
    #z = tf.concat([z1, z2], 1)
    distance = tf.squared_difference(z1, z2)
    angle = tf.multiply(z1, z2)

    print distance.shape, angle.shape

    W_angle = tf.get_variable("W_angle", [num_cells, hidden_dim], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W_dist = tf.get_variable("W_dist", [num_cells, hidden_dim], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))

    lyr_1 = tf.nn.sigmoid(tf.tensordot(distance, W_dist, 1) + tf.tensordot(angle, W_angle, 1))

    print lyr_1.shape

    W_2 = tf.get_variable("W_2", [hidden_dim, num_classes], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))
    probs = tf.nn.softmax(tf.tensordot(lyr_1, W_2, 1))

    one_best = tf.argmax(probs, axis=1)
    label_onehot = tf.one_hot(label, num_classes)

    loss = tf.losses.softmax_cross_entropy(label_onehot, probs)



    decay_steps = 100
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
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
        #train_writer = tf.summary.FileWriter('../logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            step_idx = 0
            print "Epoch:", i
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(train_exs)/batch_size):
                if step_idx % 100 == 0:
                    print 'step:', step_idx
                q1_ = []
                q2_ = []
                label_ = []
                q1_sq_len_ = []
                q2_sq_len_ = []

                for b in xrange(0, batch_size):
                    curr_idx = ex_idx * batch_size + b
                    q1_.append(pad(map(word_embeddings.get_embedding_byidx, train_exs[curr_idx].indexed_q1), seq_max_len))
                    q2_.append(pad(map(word_embeddings.get_embedding_byidx, train_exs[curr_idx].indexed_q2), seq_max_len))
                    label_.append(train_exs[curr_idx].label)
                    q1_sq_len_.append(len(train_exs[curr_idx].indexed_q1))
                    q2_sq_len_.append(len(train_exs[curr_idx].indexed_q2))
                
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {q1: q1_,
                                                                                    q2: q2_,
                                                                                   label: np.array(label_),
                                                                                   q2_len: np.array(q2_sq_len_), 
                                                                                   q1_len: np.array(q1_sq_len_)})

                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        
        # Evaluate on the test set
        test_correct = 0
        for ex_idx in xrange(0, len(test_exs)):
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different[word_embeddings.get_embedding_byidx(testQ1_mat[ex_idx])]
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                                              feed_dict={q1: [pad(map(word_embeddings.get_embedding_byidx, test_exs[ex_idx].indexed_q1), seq_max_len)], #[testq1_s_input[ex_idx]],
                                                                                q2: [pad(map(word_embeddings.get_embedding_byidx, test_exs[ex_idx].indexed_q2), seq_max_len)],
                                                                               label: np.array([test_exs[ex_idx].label]),
                                                                               q2_len: np.array([len(test_exs[ex_idx].indexed_q2)]), 
                                                                               q1_len: np.array([len(test_exs[ex_idx].indexed_q1)])}) 
            if ex_idx % 500 == 0:
                print probs_this_instance, pred_this_instance
                print pred_this_instance[0], test_exs[ex_idx].label
            
            if (test_exs[ex_idx].label == pred_this_instance[0]):
                test_correct += 1
        str1 =  repr(test_correct) + "/" + repr(len(test_exs)) + " correct after training"
        str2 =  1.0*test_correct/len(test_exs)
        print str1
        print str2
        return str(str1)+ "\t" + str(str2)

def train_bench3(train_exs, test_exs, word_embeddings, initial_learning_rate = 0.01, learning_rate_decay_factor=0.995):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 237 words to make it a square matrix.

    #TRAINING DATA
    print "TRAIN Extraction begins!"
    print len(train_exs)


    ## Matrix with seq word indices and word vectors
    dim = len(word_embeddings.get_embedding_byidx(0))


    print "TEST Extraction ends!"

    #serial input (seq of 300 corresponding to each word)
    
    batch_size = 64
    feat_vec_size = dim
    hidden_dim = 30
    num_classes = 2
    num_cells = 100
    num_layers = 3

    # DEFINING THE COMPUTATION GRAPH

    _q1 = tf.placeholder(tf.int32, [None, seq_max_len])
    _q2 = tf.placeholder(tf.int32, [None, seq_max_len])
    label = tf.placeholder(tf.int32, None)
    q1_len = tf.placeholder(tf.int32, None)
    q2_len = tf.placeholder(tf.int32, None)

    embeddings = tf.Variable(word_embeddings.vectors)
    q1 = tf.cast(tf.nn.embedding_lookup(embeddings, _q1), tf.float32)
    q2 = tf.cast(tf.nn.embedding_lookup(embeddings, _q2), tf.float32)

    def myLSTMcell(Preuse):
        lstm = tf.nn.rnn_cell.LSTMCell(num_cells, reuse=Preuse) #tf.get_variable_scope().reuse)
        return lstm

    stacked_lstm1 = tf.contrib.rnn.MultiRNNCell(
        [myLSTMcell(tf.get_variable_scope().reuse) for _ in range(num_layers)])
    stacked_lstm2 = tf.contrib.rnn.MultiRNNCell(
        [myLSTMcell(True) for _ in range(num_layers)])

    output1, _ = tf.nn.dynamic_rnn(stacked_lstm1, q1, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)
    output2, _ = tf.nn.dynamic_rnn(stacked_lstm2, q2, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)

    print output1.shape, output2.shape

    #on the basis of conclusions from last assignment, we use the mean vector instead of the last vector
    z1 = tf.reduce_mean(output1, axis=1)
    z2 = tf.reduce_mean(output2, axis=1)

    #combining the LSTM representation of the 2 questions
    distance = tf.squared_difference(z1, z2)
    angle = tf.multiply(z1, z2)

    print distance.shape, angle.shape

    W_angle = tf.get_variable("W_angle", [num_cells, hidden_dim], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W_dist = tf.get_variable("W_dist", [num_cells, hidden_dim], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))

    lyr_1 = tf.nn.sigmoid(tf.tensordot(distance, W_dist, 1) + tf.tensordot(angle, W_angle, 1))

    print lyr_1.shape

    W_2 = tf.get_variable("W_2", [hidden_dim, num_classes], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))
    probs = tf.nn.softmax(tf.tensordot(lyr_1, W_2, 1))

    one_best = tf.argmax(probs, axis=1)
    print "hey sexy", tf.shape(probs)
    label_onehot = tf.one_hot(label, num_classes)

    loss = tf.losses.softmax_cross_entropy(label_onehot, probs)



    decay_steps = 100
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    #initial_learning_rate = 0.01
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
        #train_writer = tf.summary.FileWriter('../logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            step_idx = 0
            print "Epoch:", i
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(train_exs)/batch_size):
                if step_idx % 100 == 0:
                    print 'step:', step_idx
                q1_ = []
                q2_ = []
                label_ = []
                q1_sq_len_ = []
                q2_sq_len_ = []

                for b in xrange(0, batch_size):
                    curr_idx = ex_idx * batch_size + b
                    q1_.append(pad(train_exs[curr_idx].indexed_q1, seq_max_len))
                    q2_.append(pad(train_exs[curr_idx].indexed_q2, seq_max_len))
                    label_.append(train_exs[curr_idx].label)
                    q1_sq_len_.append(len(train_exs[curr_idx].indexed_q1))
                    q2_sq_len_.append(len(train_exs[curr_idx].indexed_q2))
                
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {_q1: q1_,
                                                                                    _q2: q2_,
                                                                                   label: np.array(label_),
                                                                                   q2_len: np.array(q2_sq_len_), 
                                                                                   q1_len: np.array(q1_sq_len_)})

                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        
        # Evaluate on the test set
        test_correct = 0
        for ex_idx in xrange(0, len(test_exs)):
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different[word_embeddings.get_embedding_byidx(testQ1_mat[ex_idx])]
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                                              feed_dict={_q1: [pad(test_exs[ex_idx].indexed_q1, seq_max_len)], #[testq1_s_input[ex_idx]],
                                                                                _q2: [pad(test_exs[ex_idx].indexed_q2, seq_max_len)],
                                                                               label: np.array([test_exs[ex_idx].label]),
                                                                               q2_len: np.array([len(test_exs[ex_idx].indexed_q2)]), 
                                                                               q1_len: np.array([len(test_exs[ex_idx].indexed_q1)])}) 
            if ex_idx % 500 == 0:
                print probs_this_instance, pred_this_instance
                print pred_this_instance[0], test_exs[ex_idx].label
            
            if (test_exs[ex_idx].label == pred_this_instance[0]):
                test_correct += 1
        str1 =  repr(test_correct) + "/" + repr(len(test_exs)) + " correct after training"
        str2 =  1.0*test_correct/len(test_exs)
        print str1
        print str2
        return str(str1)+ "\t" + str(str2)

def train_bench4(train_exs, test_exs, word_embeddings, initial_learning_rate = 0.01, learning_rate_decay_factor=0.995):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 237 words to make it a square matrix.

    #TRAINING DATA
    print "TRAIN Extraction begins!"
    print len(train_exs)


    ## Matrix with seq word indices and word vectors
    dim = len(word_embeddings.get_embedding_byidx(0))


    ## Matrix with seq word indices and word vectors
    dim = len(word_embeddings.get_embedding_byidx(0))

    print "TEST Extraction ends!"

    #serial input (seq of 300 corresponding to each word)
    
    batch_size = 64
    feat_vec_size = dim
    hidden_dim = 30
    num_classes = 2
    num_cells = 100
    num_layers = 3

    # DEFINING THE COMPUTATION GRAPH

    _q1 = tf.placeholder(tf.int32, [None, seq_max_len])
    _q2 = tf.placeholder(tf.int32, [None, seq_max_len])
    label = tf.placeholder(tf.int32, None)
    q1_len = tf.placeholder(tf.int32, None)
    q2_len = tf.placeholder(tf.int32, None)

    embeddings = tf.Variable(word_embeddings.vectors)
    q1 = tf.cast(tf.nn.embedding_lookup(embeddings, _q1), tf.float32)
    q2 = tf.cast(tf.nn.embedding_lookup(embeddings, _q2), tf.float32)

    def myLSTMcell(Preuse):
        lstm = tf.nn.rnn_cell.LSTMCell(num_cells, reuse=Preuse) #tf.get_variable_scope().reuse)
        return lstm

    output1, _ = tf.nn.bidirectional_dynamic_rnn(myLSTMcell(tf.get_variable_scope().reuse), myLSTMcell(tf.get_variable_scope().reuse), q1, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)
    output2, _ = tf.nn.bidirectional_dynamic_rnn(myLSTMcell(True), myLSTMcell(True), q2, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)

    #on the basis of conclusions from last assignment, we use the mean vector instead of the last vector
    z1_1 = tf.reduce_mean(output1[0], axis=1)
    z1_2 = tf.reduce_mean(output1[1], axis=1)
    z1 = tf.concat([z1_1, z1_2], 1)

    z2_1 = tf.reduce_mean(output2[0], axis=1)
    z2_2 = tf.reduce_mean(output2[1], axis=1)
    z2 = tf.concat([z2_1, z2_2], 1)

    #combining the LSTM representation of the 2 questions
    distance = tf.squared_difference(z1, z2)
    angle = tf.multiply(z1, z2)

    print distance.shape, angle.shape

    W_angle = tf.get_variable("W_angle", [2*num_cells, hidden_dim], 
        initializer=tf.contrib.layers.xavier_initializer()) #seed=0))
    W_dist = tf.get_variable("W_dist", [2*num_cells, hidden_dim], 
        initializer=tf.contrib.layers.xavier_initializer()) #seed=0))

    lyr_1 = tf.nn.sigmoid(tf.tensordot(distance, W_dist, 1) + tf.tensordot(angle, W_angle, 1))

    print lyr_1.shape

    W_2 = tf.get_variable("W_2", [hidden_dim, num_classes], 
        initializer=tf.contrib.layers.xavier_initializer()) #seed=0))
    probs = tf.nn.softmax(tf.tensordot(lyr_1, W_2, 1))

    one_best = tf.argmax(probs, axis=1)
    label_onehot = tf.one_hot(label, num_classes)
    
    loss = tf.losses.softmax_cross_entropy(label_onehot, probs)



    decay_steps = 100
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    #initial_learning_rate = 0.01
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
        #train_writer = tf.summary.FileWriter('../logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            step_idx = 0
            print "Epoch:", i
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(train_exs)/batch_size):
                if step_idx % 100 == 0:
                    print 'step:', step_idx
                q1_ = []
                q2_ = []
                label_ = []
                q1_sq_len_ = []
                q2_sq_len_ = []

                for b in xrange(0, batch_size):
                    curr_idx = ex_idx * batch_size + b
                    q1_.append(pad(train_exs[curr_idx].indexed_q1, seq_max_len))
                    q2_.append(pad(train_exs[curr_idx].indexed_q2, seq_max_len))
                    label_.append(train_exs[curr_idx].label)
                    q1_sq_len_.append(len(train_exs[curr_idx].indexed_q1))
                    q2_sq_len_.append(len(train_exs[curr_idx].indexed_q2))
                
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {_q1: q1_,
                                                                                    _q2: q2_,
                                                                                   label: np.array(label_),
                                                                                   q2_len: np.array(q2_sq_len_), 
                                                                                   q1_len: np.array(q1_sq_len_)})

                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        
        # Evaluate on the test set
        test_correct = 0
        for ex_idx in xrange(0, len(test_exs)):
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different[word_embeddings.get_embedding_byidx(testQ1_mat[ex_idx])]
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                                              feed_dict={_q1: [pad(test_exs[ex_idx].indexed_q1, seq_max_len)], #[testq1_s_input[ex_idx]],
                                                                                _q2: [pad(test_exs[ex_idx].indexed_q2, seq_max_len)],
                                                                               label: np.array([test_exs[ex_idx].label]),
                                                                               q2_len: np.array([len(test_exs[ex_idx].indexed_q2)]), 
                                                                               q1_len: np.array([len(test_exs[ex_idx].indexed_q1)])}) 
            if ex_idx % 500 == 0:
                print probs_this_instance, pred_this_instance
                print pred_this_instance[0], test_exs[ex_idx].label
            
            if (test_exs[ex_idx].label == pred_this_instance[0]):
                test_correct += 1
        str1 =  repr(test_correct) + "/" + repr(len(test_exs)) + " correct after training"
        str2 =  1.0*test_correct/len(test_exs)
        print str1
        print str2
        return str(str1)+ "\t" + str(str2)


def train_bench5(train_exs, test_exs, word_embeddings, initial_learning_rate = 0.01, learning_rate_decay_factor=0.995):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 237 words to make it a square matrix.

    #TRAINING DATA
    print "TRAIN Extraction begins!"
    print len(train_exs)


    ## Matrix with seq word indices and word vectors
    dim = len(word_embeddings.get_embedding_byidx(0))

    print "TEST Extraction ends!"
    
    batch_size = 64
    feat_vec_size = dim
    hidden_dim = 30
    num_classes = 2
    num_cells = 100

    # DEFINING THE COMPUTATION GRAPH

    _q1 = tf.placeholder(tf.int32, [None, seq_max_len])
    _q2 = tf.placeholder(tf.int32, [None, seq_max_len])
    label = tf.placeholder(tf.int32, None)
    q1_len = tf.placeholder(tf.int32, None)
    q2_len = tf.placeholder(tf.int32, None)

    embeddings = tf.Variable(word_embeddings.vectors)
    q1 = tf.cast(tf.nn.embedding_lookup(embeddings, _q1), tf.float32)
    q2 = tf.cast(tf.nn.embedding_lookup(embeddings, _q2), tf.float32)

    def myLSTMcell(Preuse):
        lstm = tf.nn.rnn_cell.LSTMCell(num_cells, reuse=Preuse) #tf.get_variable_scope().reuse)
        return lstm

    output1, _ = tf.nn.dynamic_rnn(myLSTMcell(tf.get_variable_scope().reuse), q1, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)
    output2, _ = tf.nn.dynamic_rnn(myLSTMcell(True), q2, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)

    print output1.shape, output2.shape

    #on the basis of conclusions from last assignment, we use the mean vector instead of the last vector
    z1 = tf.reduce_mean(output1, axis=1)
    z2 = tf.reduce_mean(output2, axis=1)

    #combining the LSTM representation of the 2 questions
    distance = tf.squared_difference(z1, z2)
    angle = tf.multiply(z1, z2)

    print distance.shape, angle.shape

    W_angle = tf.get_variable("W_angle", [num_cells, hidden_dim], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W_dist = tf.get_variable("W_dist", [num_cells, hidden_dim], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))

    lyr_1 = tf.nn.sigmoid(tf.tensordot(distance, W_dist, 1) + tf.tensordot(angle, W_angle, 1))

    print lyr_1.shape

    W_2 = tf.get_variable("W_2", [hidden_dim, num_classes], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))
    probs = tf.nn.softmax(tf.tensordot(lyr_1, W_2, 1))

    one_best = tf.argmax(probs, axis=1)
    label_onehot = tf.one_hot(label, num_classes)

    loss = tf.losses.softmax_cross_entropy(label_onehot, probs)



    decay_steps = 100
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
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
        #train_writer = tf.summary.FileWriter('../logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            step_idx = 0
            print "Epoch:", i
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(train_exs)/batch_size):
                if step_idx % 100 == 0:
                    print 'step:', step_idx
                q1_ = []
                q2_ = []
                label_ = []
                q1_sq_len_ = []
                q2_sq_len_ = []

                for b in xrange(0, batch_size):
                    curr_idx = ex_idx * batch_size + b
                    q1_.append(pad(train_exs[curr_idx].indexed_q1, seq_max_len))
                    q2_.append(pad(train_exs[curr_idx].indexed_q2, seq_max_len))
                    label_.append(train_exs[curr_idx].label)
                    q1_sq_len_.append(len(train_exs[curr_idx].indexed_q1))
                    q2_sq_len_.append(len(train_exs[curr_idx].indexed_q2))
                
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {_q1: q1_,
                                                                                    _q2: q2_,
                                                                                   label: np.array(label_),
                                                                                   q2_len: np.array(q2_sq_len_), 
                                                                                   q1_len: np.array(q1_sq_len_)})

                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        
        # Evaluate on the test set
        test_correct = 0
        for ex_idx in xrange(0, len(test_exs)):
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different[word_embeddings.get_embedding_byidx(testQ1_mat[ex_idx])]
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                                              feed_dict={_q1: [pad(test_exs[ex_idx].indexed_q1, seq_max_len)], #[testq1_s_input[ex_idx]],
                                                                                _q2: [pad(test_exs[ex_idx].indexed_q2, seq_max_len)],
                                                                               label: np.array([test_exs[ex_idx].label]),
                                                                               q2_len: np.array([len(test_exs[ex_idx].indexed_q2)]), 
                                                                               q1_len: np.array([len(test_exs[ex_idx].indexed_q1)])}) 
            if ex_idx % 500 == 0:
                print probs_this_instance, pred_this_instance
                print pred_this_instance[0], test_exs[ex_idx].label
            
            if (test_exs[ex_idx].label == pred_this_instance[0]):
                test_correct += 1
        str1 =  repr(test_correct) + "/" + repr(len(test_exs)) + " correct after training"
        str2 =  1.0*test_correct/len(test_exs)
        print str1
        print str2
        return str(str1)+ "\t" + str(str2)

def train_bench6(train_exs, test_exs, word_embeddings, initial_learning_rate = 0.01, learning_rate_decay_factor=0.995):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 237 words to make it a square matrix.

    #TRAINING DATA
    print "TRAIN Extraction begins!"
    print len(train_exs)


    ## Matrix with seq word indices and word vectors
    dim = len(word_embeddings.get_embedding_byidx(0))

    print "TEST Extraction ends!"
    
    batch_size = 64
    feat_vec_size = dim
    hidden_dim = 30
    num_classes = 2
    num_cells = 100
    lstm_size = num_cells

    # DEFINING THE COMPUTATION GRAPH

    _q1 = tf.placeholder(tf.int32, [None, seq_max_len])
    _q2 = tf.placeholder(tf.int32, [None, seq_max_len])
    label = tf.placeholder(tf.int32, None)
    q1_len = tf.placeholder(tf.int32, None)
    q2_len = tf.placeholder(tf.int32, None)

    embeddings = tf.Variable(word_embeddings.vectors)
    q1 = tf.cast(tf.nn.embedding_lookup(embeddings, _q1), tf.float32)
    q2 = tf.cast(tf.nn.embedding_lookup(embeddings, _q2), tf.float32)

    def myLSTMcell(Preuse):
        lstm = tf.nn.rnn_cell.LSTMCell(num_cells, reuse=Preuse) #tf.get_variable_scope().reuse)
        return lstm

    output1, _ = tf.nn.dynamic_rnn(myLSTMcell(tf.get_variable_scope().reuse), q1, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)
    output2, _ = tf.nn.dynamic_rnn(myLSTMcell(True), q2, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)

    def less_than(output, q_len):
        output = tf.reshape(output, [seq_max_len, -1, output.shape[2]])
        output = output[:q_len]
        output = tf.reshape(output, [-1, q_len, output.shape[2]])
        return tf.reduce_mean(output, axis=1)

    def greateqthan(output):
        return tf.reduce_mean(output1, axis=1)

    z1 = tf.divide(tf.reduce_sum(output1, axis=1), tf.tile(tf.expand_dims(tf.cast(q1_len, dtype=tf.float32), axis=1), tf.constant([1, lstm_size])))
    z2 = tf.divide(tf.reduce_sum(output2, axis=1), tf.tile(tf.expand_dims(tf.cast(q2_len, dtype=tf.float32), axis=1), tf.constant([1, lstm_size])))

    #combining the LSTM representation of the 2 questions
    distance = tf.squared_difference(z1, z2)
    angle = tf.multiply(z1, z2)

    print distance.shape, angle.shape

    W_angle = tf.get_variable("W_angle", [num_cells, hidden_dim], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W_dist = tf.get_variable("W_dist", [num_cells, hidden_dim], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))

    lyr_1 = tf.nn.sigmoid(tf.tensordot(distance, W_dist, 1) + tf.tensordot(angle, W_angle, 1))

    print lyr_1.shape

    W_2 = tf.get_variable("W_2", [hidden_dim, num_classes], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))
    probs = tf.nn.softmax(tf.tensordot(lyr_1, W_2, 1))

    one_best = tf.argmax(probs, axis=1)
    label_onehot = tf.one_hot(label, num_classes)
    
    loss = tf.losses.softmax_cross_entropy(label_onehot, probs)

    decay_steps = 100
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
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
        #train_writer = tf.summary.FileWriter('../logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            step_idx = 0
            print "Epoch:", i
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(train_exs)/batch_size):
                if step_idx % 100 == 0:
                    print 'step:', step_idx
                q1_ = []
                q2_ = []
                label_ = []
                q1_sq_len_ = []
                q2_sq_len_ = []

                for b in xrange(0, batch_size):
                    curr_idx = ex_idx * batch_size + b
                    q1_.append(pad(train_exs[curr_idx].indexed_q1, seq_max_len))
                    q2_.append(pad(train_exs[curr_idx].indexed_q2, seq_max_len))
                    label_.append(train_exs[curr_idx].label)
                    q1_sq_len_.append(len(train_exs[curr_idx].indexed_q1))
                    q2_sq_len_.append(len(train_exs[curr_idx].indexed_q2))
                
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {_q1: q1_,
                                                                                    _q2: q2_,
                                                                                   label: np.array(label_),
                                                                                   q2_len: np.array(q2_sq_len_), 
                                                                                   q1_len: np.array(q1_sq_len_)})

                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        
        # Evaluate on the test set
        test_correct = 0
        for ex_idx in xrange(0, len(test_exs)):
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different[word_embeddings.get_embedding_byidx(testQ1_mat[ex_idx])]
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                                              feed_dict={_q1: [pad(test_exs[ex_idx].indexed_q1, seq_max_len)], #[testq1_s_input[ex_idx]],
                                                                                _q2: [pad(test_exs[ex_idx].indexed_q2, seq_max_len)],
                                                                               label: np.array([test_exs[ex_idx].label]),
                                                                               q2_len: np.array([len(test_exs[ex_idx].indexed_q2)]), 
                                                                               q1_len: np.array([len(test_exs[ex_idx].indexed_q1)])}) 
            if ex_idx % 500 == 0:
                print probs_this_instance, pred_this_instance
                print pred_this_instance[0], test_exs[ex_idx].label
            
            if (test_exs[ex_idx].label == pred_this_instance[0]):
                test_correct += 1
        str1 =  repr(test_correct) + "/" + repr(len(test_exs)) + " correct after training"
        str2 =  1.0*test_correct/len(test_exs)
        print str1
        print str2
        return str(str1)+ "\t" + str(str2)

def train_bench7(train_exs, test_exs, word_embeddings, initial_learning_rate = 0.01, learning_rate_decay_factor=0.995):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 237 words to make it a square matrix.

    #TRAINING DATA
    print "TRAIN Extraction begins!"
    print len(train_exs)


    ## Matrix with seq word indices and word vectors
    dim = len(word_embeddings.get_embedding_byidx(0))
    print "TEST Extraction ends!"

    batch_size = 64
    feat_vec_size = dim
    hidden_dim = 60
    num_classes = 2
    num_cells = 100

    # DEFINING THE COMPUTATION GRAPH

    _q1 = tf.placeholder(tf.int32, [None, seq_max_len])
    _q2 = tf.placeholder(tf.int32, [None, seq_max_len])
    label = tf.placeholder(tf.int32, None)
    q1_len = tf.placeholder(tf.int32, None)
    q2_len = tf.placeholder(tf.int32, None)

    embeddings = tf.Variable(word_embeddings.vectors)
    q1 = tf.cast(tf.nn.embedding_lookup(embeddings, _q1), tf.float32)
    q2 = tf.cast(tf.nn.embedding_lookup(embeddings, _q2), tf.float32)

    def myLSTMcell(Preuse):
        lstm = tf.nn.rnn_cell.LSTMCell(num_cells, reuse=Preuse) #tf.get_variable_scope().reuse)
        return lstm

    output1, _ = tf.nn.dynamic_rnn(myLSTMcell(tf.get_variable_scope().reuse), q1, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)
    output2, _ = tf.nn.dynamic_rnn(myLSTMcell(True), q2, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)

    print output1.shape, output2.shape

    #on the basis of conclusions from last assignment, we use the mean vector instead of the last vector
    z1 = tf.reduce_mean(output1, axis=1)
    z2 = tf.reduce_mean(output2, axis=1)

    
    #combining the LSTM representation of the 2 questions
    distance = tf.squared_difference(z1, z2)
    angle = tf.multiply(z1, z2)

    print distance.shape, angle.shape

    z = tf.concat([distance, angle], 1)
    W = tf.get_variable("W_angle", [num_cells*2, hidden_dim], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))

    lyr_1 = tf.nn.sigmoid(tf.tensordot(z, W, 1))

    print lyr_1.shape

    W_2 = tf.get_variable("W_2", [hidden_dim, num_classes], 
        initializer=tf.contrib.layers.xavier_initializer(seed=0))
    probs = tf.nn.softmax(tf.tensordot(lyr_1, W_2, 1))

    one_best = tf.argmax(probs, axis=1)
    label_onehot = tf.one_hot(label, num_classes)
    

    loss = tf.losses.softmax_cross_entropy(label_onehot, probs)



    decay_steps = 100
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
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
        #train_writer = tf.summary.FileWriter('../logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            step_idx = 0
            print "Epoch:", i
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(train_exs)/batch_size):
                if step_idx % 100 == 0:
                    print 'step:', step_idx
                q1_ = []
                q2_ = []
                label_ = []
                q1_sq_len_ = []
                q2_sq_len_ = []

                for b in xrange(0, batch_size):
                    curr_idx = ex_idx * batch_size + b
                    q1_.append(pad(train_exs[curr_idx].indexed_q1, seq_max_len))
                    q2_.append(pad(train_exs[curr_idx].indexed_q2, seq_max_len))
                    label_.append(train_exs[curr_idx].label)
                    q1_sq_len_.append(len(train_exs[curr_idx].indexed_q1))
                    q2_sq_len_.append(len(train_exs[curr_idx].indexed_q2))
                
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {_q1: q1_,
                                                                                    _q2: q2_,
                                                                                   label: np.array(label_),
                                                                                   q2_len: np.array(q2_sq_len_), 
                                                                                   q1_len: np.array(q1_sq_len_)})

                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        
        # Evaluate on the test set
        test_correct = 0
        for ex_idx in xrange(0, len(test_exs)):
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different[word_embeddings.get_embedding_byidx(testQ1_mat[ex_idx])]
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                                              feed_dict={_q1: [pad(test_exs[ex_idx].indexed_q1, seq_max_len)], #[testq1_s_input[ex_idx]],
                                                                                _q2: [pad(test_exs[ex_idx].indexed_q2, seq_max_len)],
                                                                               label: np.array([test_exs[ex_idx].label]),
                                                                               q2_len: np.array([len(test_exs[ex_idx].indexed_q2)]), 
                                                                               q1_len: np.array([len(test_exs[ex_idx].indexed_q1)])}) 
            if ex_idx % 500 == 0:
                print probs_this_instance, pred_this_instance
                print pred_this_instance[0], test_exs[ex_idx].label
            
            if (test_exs[ex_idx].label == pred_this_instance[0]):
                test_correct += 1
        str1 =  repr(test_correct) + "/" + repr(len(test_exs)) + " correct after training"
        str2 =  1.0*test_correct/len(test_exs)
        print str1
        print str2
        return str(str1)+ "\t" + str(str2)

def train_bench8(train_exs, test_exs, word_embeddings, initial_learning_rate = 0.01, learning_rate_decay_factor=0.995):
    print "HEY"
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 237 words to make it a square matrix.

    #TRAINING DATA
    print "TRAIN Extraction begins!"
    print len(train_exs)


    ## Matrix with seq word indices and word vectors
    dim = len(word_embeddings.get_embedding_byidx(0))

    print "TEST Extraction ends!"

    batch_size = 64
    feat_vec_size = dim
    hidden_dim = 60
    num_classes = 2
    num_cells = 100

    # DEFINING THE COMPUTATION GRAPH

    _q1 = tf.placeholder(tf.int32, [None, seq_max_len])
    _q2 = tf.placeholder(tf.int32, [None, seq_max_len])
    label = tf.placeholder(tf.int32, None)
    q1_len = tf.placeholder(tf.int32, None)
    q2_len = tf.placeholder(tf.int32, None)

    embeddings = tf.Variable(word_embeddings.vectors)
    q1 = tf.cast(tf.nn.embedding_lookup(embeddings, _q1), tf.float32)
    q2 = tf.cast(tf.nn.embedding_lookup(embeddings, _q2), tf.float32)

    def myLSTMcell(Preuse):
        lstm = tf.nn.rnn_cell.LSTMCell(num_cells, reuse=Preuse) #tf.get_variable_scope().reuse)
        return lstm#tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=tf.constant(0.75))

    output1, _ = tf.nn.dynamic_rnn(myLSTMcell(tf.get_variable_scope().reuse), q1, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)
    output2, _ = tf.nn.dynamic_rnn(myLSTMcell(True), q2, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)
    
    print output1.shape, output2.shape

    #on the basis of conclusions from last assignment, we use the mean vector instead of the last vector
    z1 = tf.reduce_mean(output1, axis=1)
    z2 = tf.reduce_mean(output2, axis=1)

    #combining the LSTM representation of the 2 questions
    distance = tf.squared_difference(z1, z2)
    angle = tf.multiply(z1, z2)

    print distance.shape, angle.shape

    z = tf.concat([distance, angle], 1)
    W = tf.get_variable("W_angle", [num_cells*2, hidden_dim], 
        initializer=tf.contrib.layers.xavier_initializer())#seed=0))

    lyr_1 = tf.nn.sigmoid(tf.tensordot(z, W, 1))

    print lyr_1.shape

    W_2 = tf.get_variable("W_2", [hidden_dim, num_classes], 
        initializer=tf.contrib.layers.xavier_initializer())#seed=0))
    probs = tf.nn.softmax(tf.tensordot(lyr_1, W_2, 1))

    one_best = tf.argmax(probs, axis=1)
    label_onehot = tf.one_hot(label, num_classes)
    

    loss = tf.losses.softmax_cross_entropy(label_onehot, probs)

    decay_steps = 100
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
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
    name = random.randint(0, 4242)
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        #train_writer = tf.summary.FileWriter('../logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
		step_idx = 0
		print "Epoch:", i
		loss_this_iter = 0
		# batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
    		for ex_idx in xrange(0, len(train_exs)/batch_size):
    			if step_idx % 100 == 0:
    			    print 'step:', step_idx
    			q1_ = []
    			q2_ = []
    			label_ = []
    			q1_sq_len_ = []
    			q2_sq_len_ = []

    			for b in xrange(0, batch_size):
    			    curr_idx = ex_idx * batch_size + b
    			    q1_.append(pad(train_exs[curr_idx].indexed_q1, seq_max_len))
    			    q2_.append(pad(train_exs[curr_idx].indexed_q2, seq_max_len))
    			    label_.append(train_exs[curr_idx].label)
    			    q1_sq_len_.append(len(train_exs[curr_idx].indexed_q1))
    			    q2_sq_len_.append(len(train_exs[curr_idx].indexed_q2))

    			[_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {_q1: q1_,
    											    _q2: q2_,
    											   label: np.array(label_),
    											   q2_len: np.array(q2_sq_len_), 
    											   q1_len: np.array(q1_sq_len_)})

    			step_idx += 1
    			loss_this_iter += loss_this_instance
    		print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)

            # Evaluate on the test set
    		test_correct = 0
    		batch_test = 100
            	if i == num_epochs -1:
                	f= open(str(name)+"bench7.txt","w+")

    		for ex_idx in xrange(0, len(test_exs)/batch_test):
    		    q1_ = []
    		    q2_ = []
    		    label_ = []
    		    q1_sq_len_ = []
    		    q2_sq_len_ = []

    		    for b in xrange(0, batch_test):
    			curr_idx = ex_idx * batch_test + b
    			q1_.append(pad(test_exs[curr_idx].indexed_q1, seq_max_len))
    			q2_.append(pad(test_exs[curr_idx].indexed_q2, seq_max_len))
    			label_.append(test_exs[curr_idx].label)
    			q1_sq_len_.append(len(test_exs[curr_idx].indexed_q1))
    			q2_sq_len_.append(len(test_exs[curr_idx].indexed_q2))
    		# Note that we only feed in the x, not the y, since we're not training. We're also extracting different[word_embeddings.get_embedding_byidx(testQ1_mat[ex_idx])]
    		# quantities from the running of the computation graph, namely the probabilities, prediction, and z
    		    [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
    										      feed_dict={_q1: q1_, #[testq1_s_input[ex_idx]],
    											_q2: q2_,
    										       label: np.array(label_),
                                                                                               q2_len: np.array(q2_sq_len_), 
                                                                                               q1_len: np.array(q1_sq_len_)}) 
    		    for b in xrange(0, len(pred_this_instance)):
                    	curr_idx = ex_idx * batch_test + b
                    	if (test_exs[curr_idx].label == pred_this_instance[b]):
                        	test_correct += 1

                        if i == num_epochs -1:
                            f.write(str(pred_this_instance[b]))

    		str1 =  repr(test_correct) + "/" + repr(len(test_exs)) + " correct after training"
    		str2 =  1.0*test_correct/len(test_exs)
    		print str1
    		print str2

            	if i == num_epochs -1:
                	f.close() 

def train_bench9(train_exs, test_exs, word_embeddings, initial_learning_rate = 0.01, learning_rate_decay_factor=0.995):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 237 words to make it a square matrix.

    #TRAINING DATA
    print "TRAIN Extraction begins!"
    print len(train_exs)

    dim = len(word_embeddings.get_embedding_byidx(0))

    print "TEST Extraction ends!"

    _q1 = tf.placeholder(tf.int32, [None, seq_max_len])
    _q2 = tf.placeholder(tf.int32, [None, seq_max_len])
    label = tf.placeholder(tf.int32, None)
    q1_len = tf.placeholder(tf.int32, None)
    q2_len = tf.placeholder(tf.int32, None)

    embeddings = tf.Variable(word_embeddings.vectors)
    q1 = tf.cast(tf.nn.embedding_lookup(embeddings, _q1), tf.float32)
    q2 = tf.cast(tf.nn.embedding_lookup(embeddings, _q2), tf.float32)

    batch_size = 64
    feat_vec_size = dim

    num_classes = 2

    # DEFINING THE COMPUTATION GRAPH

    #num_cells = 100
    num_cells = 300
    def myLSTMcell(Preuse):
        lstm = tf.nn.rnn_cell.LSTMCell(num_cells, reuse=Preuse) #tf.get_variable_scope().reuse)
        return lstm

    output1, _ = tf.nn.dynamic_rnn(myLSTMcell(tf.get_variable_scope().reuse), q1, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)
    output2, _ = tf.nn.dynamic_rnn(myLSTMcell(True), q2, dtype=tf.float32) #sequence_length=sent_len, dtype=tf.float32)
    output1 = q1
    output2 = q2
    print output1.shape, output2.shape

    hidden_ff = 10
    F = tf.get_variable("F", [hidden_ff, num_cells], 
        initializer=tf.contrib.layers.xavier_initializer())#seed=0))

    units = 100
    sent1_f = tf.layers.dense(output1, hidden_ff)
    sent2_f = tf.layers.dense(output2, hidden_ff)

    att = tf.matmul(tf.transpose(sent1_f, (0, 2, 1)), tf.transpose(sent2_f, (0, 2, 1)), 1)
    exp_att = tf.exp(att)
    along_a = tf.reduce_sum(exp_att, axis=1)
    along_b = tf.reduce_sum(exp_att, axis=2)

    along_a = tf.expand_dims(along_a, 2)
    along_a = tf.tile(along_a, tf.constant([1, 1, num_cells]))

    along_b = tf.expand_dims(along_b, 2)
    along_b = tf.tile(along_b, tf.constant([1, 1, num_cells]))

    unnorm_beta = tf.matmul(exp_att, output2, 2) #1)
    beta = tf.div(unnorm_beta, along_a)

    unnorm_alpha = tf.matmul(exp_att, output1, 2) #1)
    alpha = tf.div(unnorm_alpha, along_b)

    modif_a = tf.concat((output1, beta), axis=2)
    modif_b = tf.concat((output2, alpha), axis=2)

    hidden_g = 100
    G = tf.get_variable("G", [hidden_g, num_cells*2], 
        initializer=tf.contrib.layers.xavier_initializer())

    V1 = tf.tensordot(modif_a, G, 1)
    V2 = tf.tensordot(modif_b, G, 1)

    v1 = tf.reduce_sum(V1, axis=1)
    v2 = tf.reduce_sum(V2, axis=1)

    v = tf.concat((v1, v2), axis=1)
    H = tf.get_variable("H", [2*hidden_g, num_classes], 
        initializer=tf.contrib.layers.xavier_initializer())

    probs = tf.nn.softmax(tf.tensordot(v, H, 1))

    one_best = tf.argmax(probs, axis=1)
    label_onehot = tf.one_hot(label, num_classes)
    
    loss = tf.losses.softmax_cross_entropy(label_onehot, probs)

    decay_steps = 100
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
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
    name = random.randint(0, 4242)
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        #train_writer = tf.summary.FileWriter('../logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            step_idx = 0
        #saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
            print "Epoch:", i
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(train_exs)/batch_size):
                if step_idx % 100 == 0:
                    print 'step:', step_idx
                q1_ = []
                q2_ = []
                label_ = []
                q1_sq_len_ = []
                q2_sq_len_ = []

                for b in xrange(0, batch_size):
                    curr_idx = ex_idx * batch_size + b
                    q1_.append(pad(train_exs[curr_idx].indexed_q1, seq_max_len))
                    q2_.append(pad(train_exs[curr_idx].indexed_q2, seq_max_len))
                    label_.append(train_exs[curr_idx].label)
                    q1_sq_len_.append(len(train_exs[curr_idx].indexed_q1))
                    q2_sq_len_.append(len(train_exs[curr_idx].indexed_q2))
                
                [_, loss_this_instance, attention, summary, alpha1, beta1, v1] = sess.run([train_op, loss, exp_att, merged, alpha, beta, v], feed_dict = {_q1: q1_,
                                                                                    _q2: q2_,
                                                                                   label: np.array(label_),
                                                                                   q2_len: np.array(q2_sq_len_), 
                                                                                   q1_len: np.array(q1_sq_len_)})

                print v1 #alpha1, beta1
		step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        
        # Evaluate on the test set
            test_correct = 0
            batch_test = 100
            if i == num_epochs - 1:
                f = open(str(name)+"bench1.txt", "w+")
        	f1 = open("gold_labels.txt", "w+")

            for ex_idx in xrange(0, len(test_exs)/batch_test):
                q1_ = []
                q2_ = []
                label_ = []
                q1_sq_len_ = []
                q2_sq_len_ = []

                for b in xrange(0, batch_test):
                #print b
                    curr_idx = ex_idx * batch_test + b
                    q1_.append(pad(test_exs[curr_idx].indexed_q1, seq_max_len))
            #print q1_[0]
                    q2_.append(pad(test_exs[curr_idx].indexed_q2, seq_max_len))
                    label_.append(test_exs[curr_idx].label)
                    q1_sq_len_.append(len(test_exs[curr_idx].indexed_q1))
                    q2_sq_len_.append(len(test_exs[curr_idx].indexed_q2))
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different[word_embeddings.get_embedding_byidx(testQ1_mat[ex_idx])]
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
                [probs_this_instance, pred_this_instance] = sess.run([probs, one_best],
                                                  feed_dict={_q1: q1_, #[testq1_s_input[ex_idx]],
                                                _q2: q2_,
                                                   label: np.array(label_),
                                                                                               q2_len: np.array(q2_sq_len_), 
                                                                                               q1_len: np.array(q1_sq_len_)}) 
                for b in xrange(0, len(pred_this_instance)):
                    curr_idx = ex_idx * batch_test + b
                    if (test_exs[curr_idx].label == pred_this_instance[b]):
                        test_correct += 1

                    if i == num_epochs - 1:
                        f.write(str(pred_this_instance[b]))
            		f1.write(str(test_exs[curr_idx].label))



            str1 =  repr(test_correct) + "/" + repr(len(test_exs)) + " correct after training"
            str2 =  1.0*test_correct/len(test_exs)
            print str1
            print str2

            if i == num_epochs - 1:
                f.close()
        	f1.close()

def train_svm(train_exs, test_exs, word_embeddings, initial_learning_rate = 0.01, learning_rate_decay_factor=0.995):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 237 words to make it a square matrix.

    #TRAINING DATA
    print len(train_exs), "Training set"
    train_X = []
    train_Y = []

    for qp in train_exs:
	q1 = map(word_embeddings.get_embedding_byidx, qp.indexed_q1)
	avg_q1 = np.mean(np.asarray(q1), axis=0)
        avg_q2 = np.mean(np.asarray(map(word_embeddings.get_embedding_byidx, qp.indexed_q2)), axis=0)
        qp1 = np.asarray([avg_q1, avg_q2])
	train_X.append(np.ndarray.flatten(qp1, 'F'))
        train_Y.append(qp.label)

    print len(test_exs), "Test set"
    test_X = []
    test_Y = []

    for qp in test_exs:
        avg_q1 = np.mean(np.asarray(map(word_embeddings.get_embedding_byidx, qp.indexed_q1)), axis=0)
        avg_q2 = np.mean(np.asarray(map(word_embeddings.get_embedding_byidx, qp.indexed_q2)), axis=0)
        test_X.append(np.ndarray.flatten(np.asarray([avg_q1, avg_q2]), 'F'))
        test_Y.append(qp.label)

    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV
    
    clf = svm.SVC(C= 0.4, gamma= 0.4)
    clf.fit(train_X, train_Y)
    print clf.score(train_X, train_Y)
    print clf.dual_coef_
    f1 = open("alphas_0.95_0.95_.txt", "w+")
    for i in range(len(clf.dual_coef_[0])):
	f1.write(str(clf.dual_coef_ [0][i]) + ' ')
    f1.close() 
    y_pred = clf.predict(test_X)
    print np.sum(test_Y), np.sum(y_pred)
    print accuracy_score(test_Y, y_pred)
