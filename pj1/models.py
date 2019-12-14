import numpy as np
from scipy.cluster.vq import kmeans


class BpNet(object):

    def __init__(self, config):

        input_size = config.input_size
        hidden_size = config.hidden_size
        output_size = config.output_size
        self.params = {}
        self.params['w1'] = np.random.randn(input_size, hidden_size) * 0.1
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = np.random.randn(hidden_size, hidden_size) * 0.1
        self.params['b2'] = np.zeros(hidden_size)
        self.params['w3'] = np.random.randn(hidden_size, output_size) * 0.1
        self.params['b3'] = np.zeros(output_size)
        self.Eiters = 0
        self.loss_history = {'iter': [], 'hist': []}
        self.err_history = {'iter': [], 'hist': []}
        self.learning_rate = config.learning_rate
        self.lr_decay = config.lr_decay
        self.decay_step = config.decay_step
        self.num_iters = config.num_iters
        self.record_step = config.record_step
        self.val_step = config.val_step
        self.log_step = config.log_step
    
    def forward(self, x, d=None):
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        w3, b3 = self.params['w3'], self.params['b3']
        
        hid = np.matmul(x, w1) + b1
        hid_nl = 1 / (1 + np.exp(-hid)) # fc1 through non-linear layer
        hid_2 = np.matmul(hid_nl, w2) + b2
        hid_nl_2 = 1 / (1 + np.exp(-hid_2))
        out = np.matmul(hid_nl_2, w3) + b3
        if d is None:
            return out
        N = x.shape[0]
        loss = np.sum((out - d.reshape(-1, 1)) ** 2) * 0.5 / N
        return out, hid_nl_2, hid_nl, loss

    def train(self, x, d, x_val=None, d_val=None):

        for it in range(self.num_iters):
            
            # w1, b1 = self.params['w1'], self.params['b1']
            w2, b2 = self.params['w2'], self.params['b2']
            w3, b3 = self.params['w3'], self.params['b3']
            N = x.shape[0]
            # forward
            out, hid_nl_2, hid_nl, loss = self.forward(x, d)
            
            # backward # backprop of activation f(x) = x is omitted
            # dfc2 = (out - d.reshape(-1, 1)) / N # dloss by dfc2
            # dw2 = np.matmul(hid_nl.T, dfc2)
            # db2 = np.sum(dfc2, axis=0)
            # dfc1_nl = np.matmul(dfc2, w2.T)
            # dfc1 = dfc1_nl * ((1 - hid_nl) * hid_nl) # for sigmoid gate
            # dw1 = np.matmul(x.T, dfc1)
            # db1 = np.sum(dfc1, axis=0)
            
            dfc3 = (out - d.reshape(-1, 1)) / N
            dw3 = np.matmul(hid_nl_2.T, dfc3)
            db3 = np.sum(dfc3, axis=0)
            # print(d.shape, dfc3.shape, w3.shape)
            dfc2_nl = np.matmul(dfc3, w3.T)
            dfc2 = dfc2_nl * ((1 - hid_nl_2) * hid_nl_2)
            dw2 = np.matmul(hid_nl.T, dfc2)
            db2 = np.sum(dfc2, axis=0)
            dfc1_nl = np.matmul(dfc2, w2.T)
            dfc1 = dfc1_nl * ((1 - hid_nl) * hid_nl) # for sigmoid gate
            dw1 = np.matmul(x.T, dfc1)
            db1 = np.sum(dfc1, axis=0)
            # update
            self.params['w3'] -= self.learning_rate * dw3
            self.params['b3'] -= self.learning_rate * db3
            self.params['w2'] -= self.learning_rate * dw2
            self.params['b2'] -= self.learning_rate * db2
            self.params['w1'] -= self.learning_rate * dw1
            self.params['b1'] -= self.learning_rate * db1
            
            if (self.Eiters+1) % self.decay_step == 0:
                self.learning_rate *= self.lr_decay
                # print('learning_rate: %f' % self.learning_rate)
            if self.Eiters % self.record_step == 0:
                self.loss_history['iter'].append(self.Eiters)
                self.loss_history['hist'].append(loss)
            if self.Eiters % self.val_step == 0 and (x_val is not None) and (d_val is not None):
                y_val = self.predict(x_val)
                mse = np.mean((y_val - d_val) ** 2)
                self.err_history['iter'].append(self.Eiters)
                self.err_history['hist'].append(mse)
            if self.Eiters % self.log_step == 0:
                print('Eiter: %d, Loss: %f, Lr: %f' % (self.Eiters, loss, self.learning_rate))
#            if it % val_step == 0:
#                self.predict15100(x_val, d_val, it)
            self.Eiters += 1
        return self.loss_history, self.err_history

    def predict(self, x):
        return self.forward(x)[:, 0]


class RBF(object):

    def __init__(self, input_size, hidden_size, output_size):

        self.params = {}
        self.params['w_aug'] = None          
        self.params['sigma_2'] = None
        self.params['t'] = None
        self.sizes = {'D': input_size, 'H': hidden_size, 'C': output_size}

    def _pick_gaussian(self, x):
        k = self.sizes['H']
        t, _ = kmeans(x, k)
        dis_mat = np.zeros((k, k)) 
        dis_mat += np.sum(t ** 2, axis=1)
        dis_mat += np.sum(t ** 2, axis=1).reshape(-1, 1)
        dis_mat -= 2 * np.matmul(t, t.T)
        dis_mat *= (np.ones((k, k)) - np.eye(k)) 
        dis_mat = np.sqrt(dis_mat)
        sigma_2 = (np.max(dis_mat, axis=1) / np.sqrt(k)).reshape(-1, 1) ** 2
        self.params['t'], self.params['sigma_2'] = t, sigma_2

    def _kernel_mat(self, x):
        t, sigma_2 = self.params['t'], self.params['sigma_2']
        N, H = x.shape[0], t.shape[0]
        l2norm_2 = np.zeros((N, H)) 
        l2norm_2 += np.sum(x ** 2, axis=1).reshape(-1, 1)
        l2norm_2 += np.sum(t ** 2, axis=1)
        l2norm_2 -= 2 * np.matmul(x, t.T)        
        phi = np.exp(-l2norm_2 / sigma_2.T)
        phi_aug = np.concatenate([phi, np.ones((N, 1))], axis=1)
        return phi_aug

    def train(self, x, d):
        self._pick_gaussian(x)
        phi_aug = self._kernel_mat(x)
        phi_aug_inv = np.matmul(
                np.asarray(np.asmatrix(np.matmul(phi_aug.T, phi_aug)).I), 
                phi_aug.T)
        w_aug = np.matmul(phi_aug_inv, d.reshape(phi_aug.shape[0], -1))
        self.params['w_aug'] = w_aug
        return self.params
    
    def predict(self, x):
        phi_aug = self._kernel_mat(x)
        w_aug = self.params['w_aug']
        y = np.matmul(phi_aug, w_aug)[:, 0]
        return y

