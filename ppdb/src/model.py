import numpy as np
from data_gen import *
#from sklearn.linear_model import LogisticRegression
import tensorflow as tf

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
    #print np.shape(seq)
    if length < np.size(seq, 0):
        return seq[:length, :]
    result = np.zeros((length, np.size(seq, 1)))
    result[0:seq.shape[0], :] = seq
    return result

def train_lstm_model(train_exs, word_vectors, ppdb_pairs):


    n_classes = 2
    lstm_size = 100
    seq_max_len = 50
    batch_size = 1
    dim = len(word_vectors.get_embedding_byidx(0))

    #defining the computation graph
    q1 = tf.placeholder(tf.float32, [batch_size, seq_max_len, dim])
    len1 = tf.placeholder(tf.int32, 1)
    q2 = tf.placeholder(tf.float32, [batch_size, seq_max_len, dim])
    len2 = tf.placeholder(tf.int32, 1)

    def myLSTMCell(reuse_):
        lstm = tf.nn.rnn_cell.LSTMCell(lstm_size, reuse=reuse_)
        return lstm

    #lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    #x1 = tf.unstack(q1, seq_max_len, 1)
    #x2 = tf.unstack(q2, seq_max_len, 1)
    outputs1, states1 = tf.nn.dynamic_rnn(myLSTMCell(tf.get_variable_scope().reuse), q1, dtype=tf.float32)
    outputs2, states2 = tf.nn.dynamic_rnn(myLSTMCell(True), q2, dtype=tf.float32)

    z1 = tf.reduce_mean(outputs1[0], axis=0)
    z2 = tf.reduce_mean(outputs2[0], axis=0)

    #states_concat = tf.concat((tf.squeeze(states1[-1]), tf.squeeze(states2[-1])), axis=-1)
    states_concat = tf.concat([z1, z2], 0)

    W = tf.get_variable("W", [n_classes, 2 * lstm_size])
    probs = tf.tensordot(W, states_concat, 1)

    prediction = tf.argmax(probs)

    label = tf.placeholder(tf.int32, 1)
    label_onehot = tf.reshape(tf.one_hot(label, n_classes), shape=[n_classes])

    loss = tf.losses.softmax_cross_entropy(label_onehot, probs)

    # training algorithm parameters
    decay_steps = 1000
    learning_rate_decay_factor = 0.99
    global_step = tf.contrib.framework.get_or_create_global_step()
    initial_learning_rate = 0.01
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

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('../logs/', sess.graph)
        tf.set_random_seed(0)
        sess.run(init)
        # merge ppdb and train exs

        #print (train_exs[:100])
        #input()

        #print (train_exs[1])
        #print (ppdb_pairs[1])
        #input()
        #print len(train_exs)
        train_exs = ppdb_pairs + train_exs
        #print len(train_exs)
        for i in range(0, n_epochs):
            step_idx = 0
            print 'Epoch:', i
            loss_this_iter = 0
            #print len(train_exs)
            for ex_idx in xrange(0, len(train_exs)):
                if step_idx % 1000 == -1:
                    print 'step:', step_idx
                #print len(train_exs[ex_idx].indexed_q1)
                #print len(train_exs[ex_idx].indexed_q2)
                #print train_exs[ex_idx].label
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {
                    q1: [pad(map(word_vectors.get_embedding_byidx, train_exs[ex_idx].indexed_q1), seq_max_len)], 
                    q2: [pad(map(word_vectors.get_embedding_byidx, train_exs[ex_idx].indexed_q2), seq_max_len)], 
                    label: np.array([train_exs[ex_idx].label]), 
                    len1: np.array([len(train_exs[ex_idx].indexed_q1)]),
                    len2: np.array([len(train_exs[ex_idx].indexed_q2)])})
                step_idx += 1
                loss_this_iter += loss_this_instance
            print 'Loss for iteration ' + repr(i) + ': ' + repr(loss_this_iter)
