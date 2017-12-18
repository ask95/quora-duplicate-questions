import numpy as np
from data_gen import *
#from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from random import shuffle
from timeit import default_timer as timer

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
           

def train_svm_model(train_exs, valid_exs, test_exs, word_vectors, C_, gamma_, seed_):
    def create_XY(exs):
        X, Y = [], []
        for qp in exs:
            q1 = map(word_vectors.get_embedding_byidx, qp.indexed_q1)
            q2 = map(word_vectors.get_embedding_byidx, qp.indexed_q2)
            avg_q1 = np.mean(np.asarray(q1), axis=0)
            avg_q2 = np.mean(np.asarray(q2), axis=0)
            avg_qp = np.asarray([avg_q1, avg_q2])

            X.append(np.ndarray.flatten(avg_qp, 'F'))
            Y.append(qp.label)
        return X, Y

    train_X, train_Y = create_XY(train_exs)
    valid_X, valid_Y = create_XY(valid_exs)
    test_X, test_Y = create_XY(test_exs)

    from sklearn import svm
    from sklearn.metrics import accuracy_score
    
    clf = svm.SVC(C = C_, gamma = gamma_, random_state=seed_)
    clf.fit(train_X, train_Y)
    print clf.score(train_X, train_Y)
    print clf.score(valid_X, valid_Y)
    print clf.score(test_X, test_Y)

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

def train_lstm_model(quora_pairs, word_vectors, ppdb_pairs, valid_exs, test_exs, lstm_size, initial_learning_rate, decay_steps, learning_rate_decay_factor, variant, scaling_factor):
    n_classes = 2
    seq_max_len = 50
    batch_size = 256
    dim = len(word_vectors.get_embedding_byidx(0))

    #defining the computation graph
    q1 = tf.placeholder(tf.float32, [None, seq_max_len, dim])
    q2 = tf.placeholder(tf.float32, [None , seq_max_len, dim])
    len1 = tf.placeholder(tf.int32, None)
    len2 = tf.placeholder(tf.int32, None)
    scaling = tf.placeholder(tf.float32, None)

    def myLSTMCell(reuse_):
        lstm = tf.nn.rnn_cell.LSTMCell(lstm_size, reuse=reuse_)
        return lstm

    outputs1, states1 = tf.nn.dynamic_rnn(myLSTMCell(tf.get_variable_scope().reuse), q1, sequence_length=len1, dtype=tf.float32)
    outputs2, states2 = tf.nn.dynamic_rnn(myLSTMCell(True), q2, sequence_length=len2, dtype=tf.float32)

    z1 = tf.divide(tf.reduce_sum(outputs1, axis=1), tf.tile(tf.expand_dims(tf.cast(len1, dtype=tf.float32), axis=1), tf.constant([1, lstm_size])))
    z2 = tf.divide(tf.reduce_sum(outputs2, axis=1), tf.tile(tf.expand_dims(tf.cast(len2, dtype=tf.float32), axis=1), tf.constant([1, lstm_size])))

    distance = tf.squared_difference(z1, z2)
    angle = tf.multiply(z1, z2)

    n_hidden = 100

    W_dist = tf.get_variable("W_dist", [lstm_size, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W_angle = tf.get_variable("W_angle", [lstm_size, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    z_concat = tf.concat([tf.tensordot(distance, W_dist, 1), tf.tensordot(angle, W_angle, 1)], 1)

    W = tf.get_variable("W", [2 * n_hidden, n_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    probs = tf.tensordot(tf.nn.sigmoid(z_concat), W, 1)

    prediction = tf.argmax(probs, axis=1)

    label = tf.placeholder(tf.int32, None)
    label_onehot = tf.one_hot(label, n_classes)

    loss = tf.losses.softmax_cross_entropy(label_onehot, probs, weights=scaling)

    # training algorithm parameters
    global_step = tf.contrib.framework.get_or_create_global_step()
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
    merged = tf.summary.merge_all()

    n_epochs = 10
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        train_writer = tf.summary.FileWriter('../logs/', sess.graph)
        tf.set_random_seed(0)
        sess.run(init)

        for i in range(0, n_epochs):
            step_idx = 0
            print 'Epoch:', i,
            loss_this_iter = 0
            if variant == 'alt':
                shuffle(quora_pairs)
                shuffle(ppdb_pairs)
                train_exs = ppdb_pairs + quora_pairs
            elif variant == 'mix':
                train_exs = ppdb_pairs + quora_pairs
                shuffle(train_exs)
            elif variant == '1by1':
                if i < n_epochs/2:
                    shuffle(ppdb_pairs)
                    train_exs = [] + ppdb_pairs
                    shuffle(ppdb_pairs)
                    train_exs = train_exs + ppdb_pairs
                else:
                    shuffle(quora_pairs)
                    train_exs = [] + quora_pairs
                    shuffle(quora_pairs)
                    train_exs = train_exs + quora_pairs
            elif variant == 'noppdb':
                train_exs = quora_pairs
                shuffle(train_exs)
            else:
                print 'Invalid variant'
            for ex_idx in xrange(0, len(train_exs)/batch_size):
                q1_ = []
                q2_ = []
                label_ = []
                len1_ = []
                len2_ = []
                scaling_ = []
                for b in xrange(0, batch_size):
                    curr_idx = ex_idx * batch_size + b
                    q1_.append(pad(map(word_vectors.get_embedding_byidx, train_exs[curr_idx].indexed_q1), seq_max_len))
                    q2_.append(pad(map(word_vectors.get_embedding_byidx, train_exs[curr_idx].indexed_q2), seq_max_len))
                    label_.append(train_exs[curr_idx].label)
                    len1_.append(min(seq_max_len, len(train_exs[curr_idx].indexed_q1)))
                    len2_.append(min(seq_max_len, len(train_exs[curr_idx].indexed_q2)))
                    scaling_.append(1.0 if train_exs[curr_idx].dataset == 'quora' else scaling_factor)

                [_, loss_this_instance, summary, z1_, z2_, probs_, prediction_] = sess.run([train_op, loss, merged, z1, z2, probs, prediction], feed_dict = {
                    q1: q1_, 
                    q2: q2_, 
                    label: np.array(label_), 
                    len1: np.array(len1_),
                    len2: np.array(len2_),
                    scaling: np.array(scaling_)})
                if step_idx % 100 == 0:
                    lr = sess.run(opt._lr)
                step_idx += 1
                loss_this_iter += loss_this_instance
            print 'Loss ' + repr(i) + ': ' + repr(loss_this_iter),

            def evaluate(exs):
                correct = 0
                batch_size_pred = 100
                for ex_idx in xrange(0, len(exs)/batch_size_pred):
                    q1_ = []
                    q2_ = []
                    len1_ = []
                    len2_ = []
                    for b in xrange(0, batch_size_pred):
                        curr_idx = ex_idx * batch_size_pred + b
                        q1_.append(pad(map(word_vectors.get_embedding_byidx, exs[curr_idx].indexed_q1), seq_max_len))
                        q2_.append(pad(map(word_vectors.get_embedding_byidx, exs[curr_idx].indexed_q2), seq_max_len))
                        len1_.append(min(seq_max_len, len(exs[curr_idx].indexed_q1)))
                        len2_.append(min(seq_max_len, len(exs[curr_idx].indexed_q2)))

                    [pred_this_instance] = sess.run([prediction], feed_dict = {
                        q1: q1_, 
                        q2: q2_,
                        len1: np.array(len1_),
                        len2: np.array(len2_)})
                    for b in xrange(0, batch_size_pred):
                        curr_idx = ex_idx * batch_size_pred + b
                        if (exs[curr_idx].label == pred_this_instance[b]):
                            correct += 1
                return 100.0*correct / len(exs)
            
            print 'Train acc: ', evaluate(train_exs),
            print 'Valid acc: ', evaluate(valid_exs),
            print 'Test acc: ', evaluate(test_exs)


def train_lstm_model_two_step(quora_pairs, word_vectors, ppdb_pairs, valid_exs, test_exs, lstm_size, initial_learning_rate, decay_steps, learning_rate_decay_factor):
    n_classes = 2
    seq_max_len = 50
    batch_size = 256
    dim = len(word_vectors.get_embedding_byidx(0))

    def myLSTMCell(reuse_):
        lstm = tf.nn.rnn_cell.LSTMCell(lstm_size, reuse=reuse_)
        return lstm

    q1 = tf.placeholder(tf.float32, [None, seq_max_len, dim])
    q2 = tf.placeholder(tf.float32, [None , seq_max_len, dim])
    len1 = tf.placeholder(tf.int32, None)
    len2 = tf.placeholder(tf.int32, None)
    label = tf.placeholder(tf.int32, None)
    w_q = tf.placeholder(tf.float32, None)

    n_hidden = 100

    label_onehot = tf.one_hot(label, n_classes)

    #defining the computation graph
    with tf.variable_scope("net_quora"):
        outputs1, _ = tf.nn.dynamic_rnn(myLSTMCell(tf.get_variable_scope().reuse), \
                      q1, sequence_length=len1, dtype=tf.float32)
        outputs2, _ = tf.nn.dynamic_rnn(myLSTMCell(True), \
                      q2, sequence_length=len2, dtype=tf.float32)

        z1_q = tf.divide(tf.reduce_sum(outputs1, axis=1), \
               tf.tile(tf.expand_dims(tf.cast(len1, dtype=tf.float32), axis=1), \
               tf.constant([1, lstm_size])))
        z2_q = tf.divide(tf.reduce_sum(outputs2, axis=1), \
               tf.tile(tf.expand_dims(tf.cast(len2, dtype=tf.float32), axis=1), \
               tf.constant([1, lstm_size])))

        distance = tf.squared_difference(z1_q, z2_q)
        angle = tf.multiply(z1_q, z2_q)

        W_dist = tf.get_variable("W_dist", [lstm_size, n_hidden], \
                 initializer=tf.contrib.layers.xavier_initializer(seed=0))
        W_angle = tf.get_variable("W_angle", [lstm_size, n_hidden], \
                  initializer=tf.contrib.layers.xavier_initializer(seed=0))

        z_concat_q = tf.concat([tf.tensordot(distance, W_dist, 1), \
                   tf.tensordot(angle, W_angle, 1)], 1)

        W = tf.get_variable("W", [2 * n_hidden, n_classes], \
            initializer=tf.contrib.layers.xavier_initializer(seed=0))
        probs_q = tf.tensordot(tf.nn.sigmoid(z_concat_q), W, 1)

        loss_q = tf.losses.softmax_cross_entropy(label_onehot, probs_q, weights = w_q)

    with tf.variable_scope("net_ppdb"):
        outputs1, _ = tf.nn.dynamic_rnn(myLSTMCell(tf.get_variable_scope().reuse), \
                      q1, sequence_length=len1, dtype=tf.float32)
        outputs2, _ = tf.nn.dynamic_rnn(myLSTMCell(True), \
                      q2, sequence_length=len2, dtype=tf.float32)

        z1_p = tf.divide(tf.reduce_sum(outputs1, axis=1), \
               tf.tile(tf.expand_dims(tf.cast(len1, dtype=tf.float32), axis=1), \
               tf.constant([1, lstm_size])))
        z2_p = tf.divide(tf.reduce_sum(outputs2, axis=1), \
               tf.tile(tf.expand_dims(tf.cast(len2, dtype=tf.float32), axis=1), \
               tf.constant([1, lstm_size])))

        distance = tf.squared_difference(z1_p, z2_p)
        angle = tf.multiply(z1_p, z2_p)

        W_dist = tf.get_variable("W_dist", [lstm_size, n_hidden], \
                 initializer=tf.contrib.layers.xavier_initializer(seed=0))
        W_angle = tf.get_variable("W_angle", [lstm_size, n_hidden], \
                  initializer=tf.contrib.layers.xavier_initializer(seed=0))

        z_concat_p = tf.concat([tf.tensordot(distance, W_dist, 1), \
                   tf.tensordot(angle, W_angle, 1)], 1)

        W = tf.get_variable("W", [2 * n_hidden, n_classes], \
            initializer=tf.contrib.layers.xavier_initializer(seed=0))
        probs_p = tf.tensordot(tf.nn.sigmoid(z_concat_p), W, 1)

        loss_p = tf.losses.softmax_cross_entropy(label_onehot, probs_p, weights = 1-w_q)

    pred_q = tf.argmax(probs_q, axis=1)
    pred_p = tf.argmax(probs_p, axis=1)
    
    loss_single = loss_q + loss_p

    z1 = tf.concat([z1_q, z1_p], 1)
    z2 = tf.concat([z2_q, z2_p], 1)

    distance = tf.squared_difference(z1, z2)
    angle = tf.multiply(z1, z2)
   
    W_dist = tf.get_variable("W_dist", [2 * lstm_size, n_hidden], \
             initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W_angle = tf.get_variable("W_angle", [2 * lstm_size, n_hidden], \
              initializer=tf.contrib.layers.xavier_initializer(seed=0))

    z_combined = tf.concat([tf.tensordot(distance, W_dist, 1), \
                            tf.tensordot(angle, W_angle, 1)], 1)

    W_combined = tf.get_variable("W_comb", [2 * n_hidden, n_classes], \
                 initializer=tf.contrib.layers.xavier_initializer(seed=0))

    probs_combined = tf.tensordot(tf.nn.sigmoid(z_combined), W_combined, 1)
    loss_combined = tf.losses.softmax_cross_entropy(label_onehot, probs_combined)
    prediction = tf.argmax(probs_combined, axis=1)

    # training algorithm parameters
    global_step = tf.contrib.framework.get_or_create_global_step()
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss_single)
    opt = tf.train.AdamOptimizer(lr)

    var_list1 = filter(lambda v: 'net' in v.name, tf.trainable_variables())
    var_list2 = tf.trainable_variables()

    grads1 = opt.compute_gradients(loss_single, var_list = var_list1)
    apply_gradients_op1 = opt.apply_gradients(grads1, global_step=global_step)
    with tf.control_dependencies([apply_gradients_op1]):
        train_op1 = tf.no_op(name='train')

    grads2 = opt.compute_gradients(loss_combined, var_list = var_list2)
    apply_gradients_op2 = opt.apply_gradients(grads2, global_step=global_step)
    with tf.control_dependencies([apply_gradients_op2]):
        train_op2 = tf.no_op(name='train')

    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    n_epochs = 20
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        train_writer = tf.summary.FileWriter('../logs/', sess.graph)
        tf.set_random_seed(0)
        sess.run(init)

        for i in range(0, n_epochs):
            step_idx = 0
            print 'Epoch:', i,
            loss_this_iter = 0
            if i % 2 == 0:
                train_exs = ppdb_pairs + quora_pairs
            else:
                train_exs = quora_pairs
            shuffle(train_exs)
            for ex_idx in xrange(0, len(train_exs)/batch_size):
                q1_ = []
                q2_ = []
                label_ = []
                len1_ = []
                len2_ = []
                w_q_ = []
                for b in xrange(0, batch_size):
                    curr_idx = ex_idx * batch_size + b
                    q1_.append(pad(map(word_vectors.get_embedding_byidx, train_exs[curr_idx].indexed_q1), seq_max_len))
                    q2_.append(pad(map(word_vectors.get_embedding_byidx, train_exs[curr_idx].indexed_q2), seq_max_len))
                    label_.append(train_exs[curr_idx].label)
                    len1_.append(min(seq_max_len, len(train_exs[curr_idx].indexed_q1)))
                    len2_.append(min(seq_max_len, len(train_exs[curr_idx].indexed_q2)))
                    w_q_.append(1.0 if train_exs[curr_idx].dataset == 'quora' else 0.0)

                if i % 2 == 0:
                    [_, loss_this_instance] = sess.run([train_op1, loss_single], feed_dict = {
                        q1: q1_, 
                        q2: q2_, 
                        label: np.array(label_), 
                        len1: np.array(len1_),
                        len2: np.array(len2_),
                        w_q: np.array(w_q_)})
                else:
                    [_, loss_this_instance] = sess.run([train_op2, loss_single], feed_dict = {
                        q1: q1_, 
                        q2: q2_, 
                        label: np.array(label_), 
                        len1: np.array(len1_),
                        len2: np.array(len2_),
                        w_q: np.array(w_q_)})
                if step_idx % 100 == 0:
                    lr = sess.run(opt._lr)
                step_idx += 1
                loss_this_iter += loss_this_instance
            print 'Loss ' + repr(i) + ': ' + repr(loss_this_iter),

            def evaluate(exs):
                correct = 0
                batch_size_pred = 100
                for ex_idx in xrange(0, len(exs)/batch_size_pred):
                    q1_ = []
                    q2_ = []
                    len1_ = []
                    len2_ = []
                    for b in xrange(0, batch_size_pred):
                        curr_idx = ex_idx * batch_size_pred + b
                        q1_.append(pad(map(word_vectors.get_embedding_byidx, exs[curr_idx].indexed_q1), seq_max_len))
                        q2_.append(pad(map(word_vectors.get_embedding_byidx, exs[curr_idx].indexed_q2), seq_max_len))
                        len1_.append(min(seq_max_len, len(exs[curr_idx].indexed_q1)))
                        len2_.append(min(seq_max_len, len(exs[curr_idx].indexed_q2)))

                    [pred_this_instance] = sess.run([prediction], feed_dict = {
                        q1: q1_, 
                        q2: q2_,
                        len1: np.array(len1_),
                        len2: np.array(len2_)})
                    for b in xrange(0, batch_size_pred):
                        curr_idx = ex_idx * batch_size_pred + b
                        if (exs[curr_idx].label == pred_this_instance[b]):
                            correct += 1
                return 100.0*correct / len(exs)
            
            print 'Train acc: ', evaluate(train_exs),
            print 'Valid acc: ', evaluate(valid_exs),
            print 'Test acc: ', evaluate(test_exs)
