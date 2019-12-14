# coding:utf-8
import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import FastICA
from collections import Counter


class PCA(object):

    def __init__(self, imgs, method):
        self.x = imgs
        self.method = method
        self.N, self.n = self.x.shape
        

    def pca_cx(self, x, m):

        mv = np.mean(x, axis=0) # mean value
        xp = x - mv # samples removed mean
        cx = np.matmul(xp.T, xp) / self.N # covariance mat of x
        _, fv = eigh(cx, eigvals=(self.n-m, self.n-1)) # feature vector
        fv = fv.T # ||eigvector|| = 1.0
        fv = fv.real
        us = fv[::-1] # m by n, principal ws, ie. selected m ws
        # ys = np.matmul(xp, us.T) # N by m, coordinate under new bases
        ys = np.matmul(np.matmul(xp, us.T), us) # N by n, samples presented under old bases
        return us, ys, mv

    def pca_nn(self, x, m, lr, iters, iters_add_interval, iters_add):

        loss_hs = [] # loss histories 2-d list
        mv = np.mean(x, axis=0) # mean value
        xp = x - mv # samples removed mean
        us = np.array([]).reshape((0, self.n)) # m by n, principal ws
        rs = xp # init residual
        for m_ in range(1, m+1):
            
            loss_h = [] # loss history
            w = np.random.randn(1, self.n) # 1 by n
            w = w / np.sqrt(np.sum(w*w)) # normalize
            for ite in range(1, iters+1):
                a = np.sum(w*rs, axis=1, keepdims=True) # N by 1, coordinate under new base
                delta = rs - np.matmul(a, w)
                dw = np.mean(- a * delta, axis=0, keepdims=True)
                w -= lr * dw # update w
                w = w / np.sqrt(np.sum(w*w)) # again normalize w

                loss = np.mean(np.sum(delta*delta, axis=1))
                loss_h.append(loss)
            print('epoch: %d/%d' % (m_, m), 'loss:%f' % loss)
            # print('loss', loss)
            a = np.sum(w*rs, axis=1, keepdims=True) # N by 1, coordinate under new base
            y = np.matmul(a, w) # N by 1, samples presented under old bases
            rr = rs - y # N by n, residual

            us = np.concatenate([us, w], axis=0)
            loss_hs.append(loss_h)
            rs = rr
            if m_ % iters_add_interval == 0:
                iters += iters_add
                print('Num iters change to %d' % iters)
        # ass = np.matmul(xp, us.T) # N by m, coordinate under new bases
        ys = np.matmul(np.matmul(xp, us.T), us) # N by n, samples presented under old bases
        return us, ys, mv, loss_hs
        
    def exec(self, m, config=None):
        if self.N < m:
            print('error:m > N!')
            exit()
        if self.n < m:
            print('error:m > n!')
            exit()
        if self.method == 'cx':    
            self.us, self.ys, self.mv = self.pca_cx(self.x, m)
        elif self.method == 'nn':         
            self.us, self.ys, self.mv, self.loss_hs = self.pca_nn(
                self.x, m, config.lr, config.iters, config.iters_add_interval, config.iters_add)

    def get_results(self, m):
        us = self.us[0:m]
        mv = self.mv
        ys = np.matmul(self.x - self.mv, us.T)
        out = np.matmul(ys, us)
        return us, mv, ys, out


def whiten(x, thre=1e-10, truth=4):
    N = x.shape[1]
    x_ = x - np.mean(x, axis=1, keepdims=True)
    cx = np.matmul(x_, x_.T) / N
    _, s, vh = np.linalg.svd(cx)
    n = np.sum(s > thre)
    if n != truth:
        print('whithening: gussing channel number wrong!')
        n = truth
    s = s[:n]
    vh = vh[:n]
    M = np.matmul(np.diag(1/np.sqrt(s)), vh)
    z = np.matmul(M, x_)
    return z


class ICA(object):

    def __init__(self, x, method, config):
        self.x = x
        self.method = method
        self.config = config

    def hj(self, x, lr, iters):
        m, d = x.shape
        w = np.random.randn(m, m) / m
        w[range(m), range(m)] = 1.0
        I = np.eye(m)
        ys = np.matmul(np.linalg.inv(I+w), x)
        delta = 999.0
        delta_h = [delta, ]
        for it in range(iters):
            dw = np.matmul(np.power(ys, 3), ys.T) / d
            dw[range(m), range(m)] = 0.0
            w += lr * dw
            ys = np.matmul(np.linalg.inv(I+w), x)
            delta = np.sum(np.abs(dw))
            if it % 100 == 0:
                print('iter:%d, delta:%e' % (it, delta))
            delta_h.append(delta)
        return ys, delta_h
    
    def r4(self, x):
        D = x.shape[1]
        acc = 0.0
        # print(x[:,1000])
        for j in range(D):
            x_ = x[:, j]
            acc += np.sum(x_ * x_) * np.matmul(x_.reshape(-1, 1), x_.reshape(1, -1))
        r4 = acc / D
        _, BT = np.linalg.eig(r4)
        ys = np.matmul(BT, x)
        return ys

    def fast_ica(self, x, tol, max_iter):
        ica_fast = FastICA(whiten=False, tol=tol, max_iter=max_iter)
        ys = ica_fast.fit_transform(x.T).T
        return ys
    
    def exec(self):
        if self.method == 'hj':
            ys, delta_h = self.hj(self.x, self.config.lr, self.config.iter)
            return [ys, delta_h]
        elif self.method == 'r4':
            ys = self.r4(self.x)
            return [ys]
        elif self.method == 'fast':
            ys = self.fast_ica(self.x, 1e-18, self.config.iter)
            return [ys]

def match_and_recover(ss, ys, signal='image'):
    n, D = ss.shape
    coef_mat = np.zeros((n, n))
    outs = np.zeros_like(ys)
    # compute correlation coefficients
    for i in range(n):
        for j in range(n):
            coef_mat[i, j] = np.corrcoef(ys[i], ss[j])[0, 1]
    # assign labels
    coef_mat_abs = np.abs(coef_mat)
    labels = np.argmax(coef_mat_abs, axis=1)
    # coef_mat_sign = (coef_mat / coef_mat_abs)[range(n), labels]
    for i in range(n):
        y = ys[i]
        j = labels[i]
        s = ss[j]
        mu = np.mean(s)
        s0 = s - mu
        alpha = np.sum(s0 * y) / np.sum(s0 * s0)
        outs[i] = y / alpha + mu
        outs[outs<0.0] = 0.0
        outs[outs>1.0] = 1.0

    return coef_mat_abs, labels, outs

# SOM
def assign_labels_to_nerons(labels_record, map_size):
    result_map = np.zeros((map_size, map_size), dtype=np.int32)
    for i in range(map_size):
        for j in range(map_size):
            p = (i, j)
            if p in labels_record:
                result_map[i, j] = labels_record[p].most_common()[0][0]
            else:
                result_map[i, j] = -1
    mask = (result_map > -1)
    reliable_map = result_map.copy()
    for i in range(map_size):
        for j in range(map_size):
            if not mask[i, j]:
                result_map[i, j] = vote(reliable_map, (i, j))
    return result_map
                
                
def vote(lbs, p):
    map_size = lbs.shape[0]
    counter = Counter()
    for dist in range(1, map_size):
        for absdi in range(0, dist+1):
            absdj = dist - absdi
            didj = {(absdi, absdj), (absdi, -absdj), (-absdi, absdj), (-absdi, -absdj)}
            for d in didj:
                i, j = p[0]+d[0], p[1]+d[1]
                if i>=0 and i<map_size and j>=0 and j<map_size:
                    if lbs[i, j] > -1:
                        counter[lbs[i, j]] += 1
        most_common = counter.most_common()
        if (len(most_common)==1) or (len(most_common)>1 and most_common[0][1]>most_common[1][1]):
            return most_common[0][0]
    return -1


def som_acc(som, data_test, labels_test, result_map):
    acc = 0
    for i, x in enumerate(data_test):
        acc += (labels_test[i] == result_map[som.winner(x)])
    acc /= labels_test.shape[0]
    return acc