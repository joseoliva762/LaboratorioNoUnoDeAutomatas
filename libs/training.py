import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split as tts

class Training():
    def __init__(self, lr=0.52, n_ep=1000, dispstep=100):
        self.learning_rate = lr
        self.num_epochs = n_ep
        self.display_step = dispstep

    def get_hiperparameters(self, _x_features_training):
        self.rows = np.shape(_x_features_training)[0]
        self.columns = np.shape(_x_features_training)[1]
        print('[OK]    Get Hierparameters')
        return self.learning_rate, self.num_epochs, self.display_step, self.rows, self.columns


    def get_placeholders(self, rows=1, colms=1):
        tf.reset_default_graph()
        self.X = tf.placeholder(tf.float32, [rows, colms])
        self.Y = tf.placeholder(tf.float32, [rows, 1])
        print('[OK]    Create PlaceHolder')
        return self.X, self.Y

    def get_variables(self, colms=1):
        self.theta = tf.get_variable("theta",[colms, 1])
        self.theta_cero = tf.get_variable("theta0",[1])
        print('[OK]    Create Thetas')
        return self.theta, self.theta_cero

    def _h_func(self, _X_features_ph):
        self.hipotesis = tf.add(tf.matmul(_X_features_ph, self.theta), self.theta_cero)
        print('[OK]    Hipotesis.')
        return self.hipotesis

    def _J_func(self, _y_labels_training, _X_features_ph):
        self.J = tf.pow(self._h_func(_X_features_ph) - _y_labels_training, 2)
        print('[OK]    Loss Function.')
        return self.J

    def get_mean_j(self, _y_labels_training, _X_features_ph):
        self.mean_J = (1/(2*self.rows))*tf.reduce_sum(self._J_func(_y_labels_training, _X_features_ph))
        print('[OK]    Mean of Loss Function.')
        return self.mean_J

    def tftGDO(self):
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.mean_J)
        print('[OK]    Gradient Descent Optimizer.')
        return self.optimizer

    def model(self, _ep, _train_cost, X_features_ph, x_features_training, Y_labels_ph, y_labels_training, theta, theta_cero):
        self.ep = _ep
        self.train_cost = _train_cost
        print('\n\tStart Training...\r\n')
        with tf.name_scope("starting_tensorflow_session"):
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for _epoch in range(self.num_epochs):
                    _, self.t_cost = sess.run([self.optimizer, self.mean_J], feed_dict={X_features_ph: x_features_training, Y_labels_ph: y_labels_training})
                    self.train_cost.append(self.t_cost)
                    self.ep.append(_epoch)
                    if (_epoch+1) % self.display_step == 0:
                        print("Epoch: {}".format(_epoch + 1), "train_cost={:0.4f}".format(self.t_cost))
                print('\n\tTraining Completed.')
                self.th = sess.run(theta)
                self.th0 = sess.run(theta_cero)
        return self.ep, self.train_cost, self.th, self.th0


# ************************************** Area de trabajo ********************************************************************************
    def _J_func_test(self):
        self.J_test = np.power(self.h_test - self.y_labels_test, 2)
        return self.J_test

    def mean_J_test_func(self, test_x_rows):
        self.mean_J_test = (1/(2*test_x_rows))*np.sum(self._J_func_test())
        return self.mean_J_test

    def apply_test(self, _x_features_test, _y_labels_test):
        print("\n\tStart Testing...\n")
        self.x_features_test = _x_features_test
        self.y_labels_test = _y_labels_test
        test_x_rows = np.shape(self.x_features_test)[0]
        self.h_test = np.dot(self.x_features_test, self.th) + self.th0
        #print(self.y_labels_test.shape)
        self.predice = self.mean_J_test_func(test_x_rows)
        print('>> Prediction: {:.4f}.'.format(self.predice[0]))
        #predice =  (1/(2*tam))*np.sum(np.power(hipo - y_labels_test, 2))
        self.compara = np.abs(self.y_labels_test - self.predice) # errores
        n_mal = np.sum(self.compara)/test_x_rows  # Saco el promedio de los errores. suma de todos los errores/np.shape(self.compara)[0]
        self.t_accuracy = (test_x_rows - n_mal)/test_x_rows
        return self.t_accuracy
