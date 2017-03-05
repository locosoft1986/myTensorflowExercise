import tensorflow as tf
import numpy as np


def weight_variable(shape, std):
    initial = tf.truncated_normal(shape, stddev=std, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(1.0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)

def one_hot(y, output_size):
    out = np.zeros((y.size, output_size))
    out[range(y.size), y] = 1
    return out

class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size, reg=0.5, learning_rate=1e-3, std=1e-4):
        # build the tensorflow graph
        self.params = {}
        X = tf.placeholder(tf.float32, [None, input_size])
        y = tf.placeholder(tf.int32, [None, output_size])

        self.params['W1'] = weight_variable([input_size, hidden_size], std)
        self.params['b1'] = bias_variable([hidden_size])
        self.params['W2'] = weight_variable([hidden_size, output_size], std)
        self.params['b2'] = bias_variable([output_size])

        self.X = X
        self.y = y

        affine1 = tf.matmul(X, self.params['W1']) + self.params['b1']
        relu = tf.nn.relu(affine1)
        affine2 = tf.matmul(relu, self.params['W2']) + self.params['b2']
        pred_y = tf.nn.softmax(affine2)

        power_sum_w1 = tf.reduce_sum(tf.pow(self.params['W1'], 2))
        power_sum_w2 = tf.reduce_sum(tf.pow(self.params['W2'], 2))
        reg_part = 0.5 * reg * power_sum_w1 * power_sum_w2

        loss = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred_y)), reg_part)

        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        prediction = tf.argmax(pred_y, dimension=1)
        correct_predictions = tf.equal(prediction, tf.argmax(y, dimension=1))

        self.output_size = output_size
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        self.prediction = prediction
        self.pred_y = pred_y
        self.loss = loss
        self.train_step = train_step
        self.session = None

    def train(self, session, X, y, X_val, y_val,
            num_iters=100,
            batch_size=200, verbose=False):
        if session is None:
            raise 'Session is not an object'

        session.run(tf.global_variables_initializer())
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        one_hot_y_val = one_hot(y_val, self.output_size)

        for it in range(num_iters):
            x_batch = None
            y_batch = None
            idx = np.random.choice(num_train, batch_size, replace=True)
            x_batch = X[idx]
            y_batch = one_hot(y[idx], self.output_size)

            _, loss = session.run([self.train_step, self.loss], feed_dict={self.X: x_batch, self.y: y_batch})

            loss_history.append(loss)

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = session.run(self.accuracy, feed_dict={self.X: x_batch, self.y: y_batch})
                val_acc = session.run(self.accuracy, feed_dict={self.X: X_val, self.y: one_hot_y_val})
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, session, X):
        if session is None:
            raise 'Session is not an object'

        return session.run(self.prediction, feed_dict={self.X: X})






