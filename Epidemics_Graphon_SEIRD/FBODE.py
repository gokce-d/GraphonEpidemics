import tensorflow as tf
import numpy as np


class FBODEEquation(object):
    def __init__(self, beta, gamma, rho, epsilon, lambda1, lambda2, lambda3, lambda4, cost_I, cost_lambda1, cost_lambda2, cost_dead, g, Delta_t):
        # parameters of the MFG
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.epsilon = epsilon
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.cost_I = cost_I # penalty for being in I
        self.cost_lambda1 = cost_lambda1
        self.cost_lambda2 = cost_lambda2
        self.cost_dead = cost_dead
        self.g = g
        self.Delta_t = Delta_t

    def driver_P(self, Z_empirical, P, ALPHA):
        n_samples = tf.shape(P)[0]
        Nstates = tf.shape(P)[1]
        driver = np.zeros((n_samples, Nstates))
        driver[:,0] = tf.reshape(-self.beta * ALPHA * Z_empirical * tf.reshape(P[:, 0], [n_samples, 1]), [n_samples])
        driver[:,1] = tf.reshape( self.beta * ALPHA * Z_empirical * tf.reshape(P[:, 0], [n_samples, 1]) - self.epsilon * tf.reshape(P[:, 1], [ n_samples, 1]), [n_samples])
        driver[:,2] = tf.reshape( self.epsilon * tf.reshape(P[:, 1], [ n_samples, 1]) - self.gamma * tf.reshape(P[:, 2], [n_samples, 1]),[n_samples])
        driver[:,3] = tf.reshape( self.rho * self.gamma * tf.reshape(P[:, 2], [n_samples, 1]), [n_samples])
        driver[:,4] = tf.reshape( (1-self.rho) * self.gamma * tf.reshape(P[:, 2], [n_samples, 1]), [n_samples])
        return driver

    def driver_U(self, Z_empirical, U, ALPHA):
        n_samples = tf.shape(U)[0]
        Nstates = tf.shape(U)[1]
        driver = np.zeros((n_samples, Nstates))
        driver[:,0] = tf.reshape(self.beta * ALPHA * Z_empirical * (tf.reshape(U[:, 0], [n_samples, 1]) - tf.reshape(U[:, 1], [n_samples, 1])) - \
                                  0.5 * self.cost_lambda1 * ((self.lambda1 - ALPHA) ** 2), [n_samples])
        driver[:,1] = tf.reshape(self.epsilon * (tf.reshape(U[:, 1], [n_samples, 1]) - tf.reshape(U[:, 2], [n_samples, 1])), [n_samples])
        driver[:,2] = tf.reshape(self.rho * self.gamma * (tf.reshape(U[:, 2], [n_samples, 1]) - tf.reshape(U[:, 3], [n_samples, 1])) + \
                                (1-self.rho) * self.gamma * (tf.reshape(U[:, 2], [n_samples, 1]) - tf.reshape(U[:, 4], [n_samples, 1])) - self.cost_I, [n_samples])
        driver[:,4] = -1 * self.cost_dead * np.ones((n_samples))
        return driver

    def optimal_ALPHA(self, U, Z_empirical):
        n_samples = tf.shape(U)[0]
        alpha = self.lambda1 + (self.beta / self.cost_lambda1) * Z_empirical * (
                    tf.reshape(U[:, 0], [n_samples, 1]) - tf.reshape(U[:, 1], [n_samples, 1]))
        return alpha


