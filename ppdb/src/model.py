import numpy as np
from data_gen import *
#from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from random import shuffle

def generate_avg_embedding(indexed_q, word_vectors):
    dim = len(word_vectors.get_embedding_byidx(0))
    emb = np.zeros(dim)
    for word_idx in indexed_q:
        emb += word_vectors.get_embedding_byidx(word_idx)
    return emb/len(indexed_q)

class AvgEmbeddingModel:
    def __init__(self, model, word_vectors):
        self.model = model
        self.word_vectors = word_vectors

    def predict(self, test_exs):
        X = []
        y = []
        # generate X's for test examples
        for ex in test_exs:
            q1_avg_emb = generate_avg_embedding(ex.indexed_q1, self.word_vectors)
            q2_avg_emb = generate_avg_embedding(ex.indexed_q1, self.word_vectors)
            X.append(q1_avg_emb * q2_avg_emb)
            y.append(ex.label)
        print 'Test accuracy:', self.model.score(X, y)
           

def train_avg_model(train_exs, word_vectors):
    # generate average embeddings
    X = []
    y = []
    for ex in train_exs:
        q1_avg_emb = generate_avg_embedding(ex.indexed_q1, word_vectors)
        q2_avg_emb = generate_avg_embedding(ex.indexed_q1, word_vectors)
        X.append(q1_avg_emb * q2_avg_emb)
        y.append(ex.label)

    model = LogisticRegression()
    model = model.fit(X, y)
    print 'Training accuracy:', model.score(X, y)
    return AvgEmbeddingModel(model, word_vectors)

def pad(seq, length):
    seq = np.asarray(seq)
    if length < np.size(seq, 0):
        return seq[:length, :]
    result = np.zeros((length, np.size(seq, 1)))
    result[0:seq.shape[0], :] = seq
    return result

def train_lstm_model(train_exs, word_vectors, ppdb_pairs, test_exs):
    n_classes = 2
    lstm_size = 100
    seq_max_len = 50
    batch_size = 256
    dim = len(word_vectors.get_embedding_byidx(0))

    #defining the computation graph
    q1 = tf.placeholder(tf.float32, [None, seq_max_len, dim])
    len1 = tf.placeholder(tf.int32, None)
    q2 = tf.placeholder(tf.float32, [None , seq_max_len, dim])
    len2 = tf.placeholder(tf.int32, None)

    def myLSTMCell(reuse_):
        lstm = tf.nn.rnn_cell.LSTMCell(lstm_size, reuse=reuse_)
        return lstm

    outputs1, states1 = tf.nn.dynamic_rnn(myLSTMCell(tf.get_variable_scope().reuse), q1, sequence_length=len1, dtype=tf.float32)
    outputs2, states2 = tf.nn.dynamic_rnn(myLSTMCell(True), q2, sequence_length=len2, dtype=tf.float32)

    z1 = tf.divide(tf.reduce_sum(outputs1, axis=1), tf.tile(tf.expand_dims(tf.cast(len1, dtype=tf.float32), axis=1), tf.constant([1, lstm_size])))
    z2 = tf.divide(tf.reduce_sum(outputs2, axis=1), tf.tile(tf.expand_dims(tf.cast(len2, dtype=tf.float32), axis=1), tf.constant([1, lstm_size])))

    states_concat = tf.concat([z1, z2], 1)

    W = tf.get_variable("W", [2 * lstm_size, n_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    probs = tf.tensordot(states_concat, W, 1)

    prediction = tf.argmax(probs, axis=1)

    label = tf.placeholder(tf.int32, None)
    label_onehot = tf.one_hot(label, n_classes)

    loss = tf.losses.softmax_cross_entropy(label_onehot, probs)

    # training algorithm parameters
    decay_steps = 1000
    learning_rate_decay_factor = 0.95
    global_step = tf.contrib.framework.get_or_create_global_step()
    initial_learning_rate = 0.001
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(loss)
    apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradients_op]):
        train_op = tf.no_op(name='train')

    init = tf.global_variables_initializer()
    n_epochs = 10
    merged = tf.summary.merge_all()

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        train_writer = tf.summary.FileWriter('../logs/', sess.graph)
        tf.set_random_seed(0)
        sess.run(init)

        #train_exs = ppdb_pairs + train_exs
        for i in range(0, n_epochs):
            step_idx = 0
            print 'Epoch:', i,
            loss_this_iter = 0
            shuffle(train_exs)
            for ex_idx in xrange(0, len(train_exs)/batch_size):
                q1_ = []
                q2_ = []
                label_ = []
                len1_ = []
                len2_ = []
                for b in xrange(0, batch_size):
                    curr_idx = ex_idx * batch_size + b
                    q1_.append(pad(map(word_vectors.get_embedding_byidx, train_exs[curr_idx].indexed_q1), seq_max_len))
                    q2_.append(pad(map(word_vectors.get_embedding_byidx, train_exs[curr_idx].indexed_q2), seq_max_len))
                    label_.append(train_exs[curr_idx].label)
                    len1_.append(len(train_exs[curr_idx].indexed_q1))
                    len2_.append(len(train_exs[curr_idx].indexed_q2))

                [_, loss_this_instance, summary, z1_, z2_, probs_, prediction_, outputs1_, z1_corr_] = sess.run([train_op, loss, merged, z1, z2, probs, prediction, outputs1, z1_corr], feed_dict = {
                    q1: q1_, 
                    q2: q2_, 
                    label: np.array(label_), 
                    len1: np.array(len1_),
                    len2: np.array(len2_)})
                if step_idx % 100 == 0:
                    lr = sess.run(opt._lr)
                    print 'step:', step_idx, 'of', len(train_exs)/batch_size, ex_idx
                    #print 'lr:', lr
                    #print 'z1:', np.linalg.norm(z1_)
                    #print 'z2:', np.linalg.norm(z2_)
                    #print 'probs:', probs_
                    #print 'pred:', prediction_
                step_idx += 1
                loss_this_iter += loss_this_instance
            print 'Loss ' + repr(i) + ': ' + repr(loss_this_iter),

            # evaluate
            train_correct = 0
            batch_size_pred = 100
            for ex_idx in xrange(0, len(train_exs)/batch_size_pred):
                q1_ = []
                q2_ = []
                len1_ = []
                len2_ = []
                for b in xrange(0, batch_size_pred):
                    curr_idx = ex_idx * batch_size_pred + b
                    q1_.append(pad(map(word_vectors.get_embedding_byidx, train_exs[curr_idx].indexed_q1), seq_max_len))
                    q2_.append(pad(map(word_vectors.get_embedding_byidx, train_exs[curr_idx].indexed_q2), seq_max_len))
                    len1_.append(len(train_exs[curr_idx].indexed_q1))
                    len2_.append(len(train_exs[curr_idx].indexed_q2))

                [pred_this_instance] = sess.run([prediction], feed_dict = {
                    q1: q1_, 
                    q2: q2_,
                    len1: np.array(len1_),
                    len2: np.array(len2_)})
                #print train_exs[ex_idx].label, pred_this_instance[0]
                for b in xrange(0, batch_size_pred):
                    curr_idx = ex_idx * batch_size_pred + b
                    if (train_exs[curr_idx].label == pred_this_instance[b]):
                        train_correct += 1
            print 'Train accuracy', 
            #print repr(train_correct) + '/' + repr(len(train_exs)) + ' correct after training'
            print 100.0*train_correct / len(train_exs),

            # evaluate
            test_correct = 0
            batch_size_pred = 100
            for ex_idx in xrange(0, len(test_exs)/batch_size_pred):
                q1_ = []
                q2_ = []
                len1_ = []
                len2_ = []
                for b in xrange(0, batch_size_pred):
                    curr_idx = ex_idx * batch_size_pred + b
                    q1_.append(pad(map(word_vectors.get_embedding_byidx, test_exs[curr_idx].indexed_q1), seq_max_len))
                    q2_.append(pad(map(word_vectors.get_embedding_byidx, test_exs[curr_idx].indexed_q2), seq_max_len))
                    len1_.append(len(test_exs[curr_idx].indexed_q1))
                    len2_.append(len(test_exs[curr_idx].indexed_q2))

                [pred_this_instance] = sess.run([prediction], feed_dict = {
                    q1: q1_, 
                    q2: q2_,
                    len1: np.array(len1_),
                    len2: np.array(len2_)})
                #print test_exs[ex_idx].label, pred_this_instance[0]
                for b in xrange(0, batch_size_pred):
                    curr_idx = ex_idx * batch_size_pred + b
                    if (test_exs[curr_idx].label == pred_this_instance[b]):
                        test_correct += 1
            print 'Test accuracy',
            #print repr(test_correct) + '/' + repr(len(test_exs)) + ' correct after testing'
            print 100.0*test_correct / len(test_exs)

        '''
        # evaluate
        test_correct = 0
        for ex_idx in xrange(0, len(test_exs)):
            [pred_this_instance] = sess.run([prediction], feed_dict = {
                q1: [pad(map(word_vectors.get_embedding_byidx, test_exs[ex_idx].indexed_q1), seq_max_len)], 
                q2: [pad(map(word_vectors.get_embedding_byidx, test_exs[ex_idx].indexed_q2), seq_max_len)], 
                len1: np.array([len(test_exs[ex_idx].indexed_q1)]),
                len2: np.array([len(test_exs[ex_idx].indexed_q2)])})
            #print test_exs[ex_idx].label, pred_this_instance[0]
            if (test_exs[ex_idx].label == pred_this_instance[0]):
                test_correct += 1
        print repr(test_correct) + '/' + repr(len(test_exs)) + ' correct after training'
        print 100.0*test_correct / len(test_exs)
        '''
