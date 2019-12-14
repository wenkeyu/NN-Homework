import numpy as np
import models
import argparse


def get_sets(features, train_num, val_num, test_num):# train_set、val_set、test_set index
	# default value 6 0 4
	_, dim = features.shape
	x_train = np.zeros((train_num*40, dim)) # 6, m
	d_train = np.zeros(40*train_num).astype('int')
	x_test = np.zeros((test_num*40, dim)) # 4, m
	d_test = np.zeros(40*test_num).astype('int')
	
	for i in range(40):# 40 classes to random split every 10 pics in 1 class
		features_c = features[10*i:10*(i+1)]
		arr = np.arange(10)
		np.random.seed(10*i)
		np.random.shuffle(arr)
		train_set = arr[0:train_num]
		test_set = arr[train_num:10]
		feature_train = features_c[train_set]
		feature_test = features_c[test_set]
		x_train[train_num * i: train_num * (i+1)] = feature_train
		d_train[train_num * i: train_num * (i+1)] = i
		x_test[test_num * i: test_num * (i+1)] = feature_test
		d_test[test_num * i: test_num * (i+1)] = i
	features_ = {'x_train': x_train, 'x_test': x_test,
				 'd_train': d_train, 'd_test': d_test}
	return features_


class BP_config(object):
    def __init__(self):
        self.input_size = 1
        self.hidden_size = 500
        self.output_size = 40
        self.learning_rate = 0.25
        self.lr_decay = 0.9
        self.decay_step = 100000
        self.num_iters = 3000
        self.record_step = 100
        self.val_step = 100
        self.log_step = 100
        self.reg = 0.0

    def change_input_size(self, m):
    	self.input_size = m

    def change_hideen_size(self, H):
    	self.hidden_size = H


class SVM_config(object):

	def __init__(self):
		self.gamma = 7 
		self.C = 5 
		self.tol = 1e-7
		self.verbose = False

	def change_gamma(self, gamma):
		self.gamma = gamma

	def change_C(self, C):
		self.C = C


class RBF_config(object):
    def __init__(self):
        self.input_size = 1
        self.hidden_size = 10
        self.output_size = 40
        self.learning_rate = 100
        self.lr_decay = 0.9
        self.decay_step = 100000
        self.num_iters = 300000
        self.record_step = 100
        self.val_step = 100
        self.log_step = 100
        self.reg = 0.0

    def change_input_size(self, m):
    	self.input_size = m

    def change_hideen_size(self, H):
    	self.hidden_size = H


def main():
	parser = argparse.ArgumentParser()    
	parser.add_argument('--feature_path', default='data/features/', type=str)
	parser.add_argument('--train_num', default=5, type=int)
	parser.add_argument('--val_num', default=0, type=int)
	parser.add_argument('--test_num', default=5, type=int)
	parser.add_argument('--model', default='bp', type=str)
	opt = parser.parse_args()
	ms = [50, 100, 150, 200, 250, 300, 350, 10304]# 10304,
	# gammas = list(np.arange(1.0, 10.0, 0.2))
	gammas = [5.6,3.4,2.6,2.4,2.2,2.8,2.8,1.4]
	cs = [0.001]#, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3, 5, 7, 10]
	Hs = [25, 50, 100, 200, 300, 400, 500, 750, 1000]
	R_h = list(range(6,30))
	svm_acc = []
	if opt.model == 'bp':
		config = BP_config()
	elif opt.model == 'rbf':
		config = RBF_config()
	elif opt.model == 'svm':
		config = SVM_config()
	for m in ms:
		features = np.load(opt.feature_path+'feature_'+str(m)+'.npy') # pca features
		features_ = get_sets(features, opt.train_num, opt.val_num, opt.test_num)
		if opt.model == 'bp':
			config.change_input_size(m) # to suit different size of input m
			for H in Hs:
				config.change_hideen_size(H)
				print('H:%d m:%d' % (H, m))
				model = models.BpNet(config)
				loss_h, acc_h = model.train(features_['x_train'], features_['d_train'], 
										  features_['x_test'], features_['d_test'])
				y_test = np.argmax(model.predict(features_['x_test']), axis=1)
				acc = np.sum(y_test == features_['d_test']) / features_['d_test'].shape[0]
				print('acc: %f' % acc)
		elif opt.model == 'svm':
			svm_1 = []
			for gamma in gammas:
				svm_2 = []
				config.change_gamma(gamma)
				for C in cs:
					config.change_C(C)
					model = models.SVM(config)
					model.train(features_['x_train'], features_['d_train'])
					y_test = model.predict(features_['x_test'])
					acc = np.sum(y_test == features_['d_test']) / features_['d_test'].shape[0]
					print('m:%d gamma:%1f C: % 3f acc:%f' % (m, gamma, C, acc), end='\r')
					svm_2.append(acc)
				svm_1.append(svm_2)
			svm_acc.append(svm_1)
		elif opt.model == 'rbf':
			config.change_input_size(m)
			for h in R_h:
				config.change_hideen_size(h)
				# model = models.RBF(m, 20, 40)
				model = models.RBF_bp(config, features_['x_train']) # 25 for fun_3 20 for func_2 22 for func_1
				model.train(features_['x_train'], features_['d_train'])
				y_test = np.argmax(model.predict(features_['x_test']), axis=1)
				acc = np.sum(y_test == features_['d_test']) / features_['d_test'].shape[0]
				print('m:%d h:%d' % (m, h))
				print(acc)
				rbf_acc.append(acc)
	svm_acc_np = np.array(svm_acc)
	
	np.save('svm_acc.npy', svm_acc_np)
	print(svm_acc)
	print(svm_acc_np.shape)

if __name__ == '__main__':
    main()