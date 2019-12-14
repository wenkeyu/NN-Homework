import numpy as np
from minisom import MiniSom
import argparse
import matplotlib.pyplot as plt
from engine import assign_labels_to_nerons, som_acc


def get_sets(features, train_num, val_num, test_num):# train_set、val_set、test_set index
	# default value 6 0 4
	_, dim = features.shape
	x_train = np.zeros((train_num*2, dim)) # 6, m
	d_train = np.zeros(2*train_num).astype('int')
	x_test = np.zeros((test_num*2, dim)) # 4, m
	d_test = np.zeros(2*test_num).astype('int')
	
	for i in range(2):# 40 classes to random split every 10 pics in 1 class
		features_c = features[10*i:10*(i+1)]
		arr = np.arange(10)
		np.random.seed(i)
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


def main():
	parser = argparse.ArgumentParser()    
	parser.add_argument('--feature_path', default='data/features/', type=str)
	parser.add_argument('--train_num', default=5, type=int)
	parser.add_argument('--val_num', default=0, type=int)
	parser.add_argument('--test_num', default=5, type=int)
	parser.add_argument('--iter', default=1000, type=int)
	parser.add_argument('--map_size', default=3, type=int)
	parser.add_argument('--neighborhood_function', default='triangle', type=str)
	parser.add_argument('--learning_rate', default=0.06, type=int)
	parser.add_argument('--sigma', default=2, type=int)
	opt = parser.parse_args()
	# ms = [50, 100, 150, 200, 250, 300, 350, 10304]
	ms = [350]
	faces = [0, 1] # face classes to be used
	for m in ms:
		features = np.load(opt.feature_path+'feature_'+str(m)+'.npy') # pca features
		features = np.concatenate((features[faces[0]*10: faces[0]*10+10],features[faces[1]*10: faces[1]*10+10]), axis=0) # 只取两类人脸
		# features = features[10:30]
		features_ = get_sets(features, opt.train_num, opt.val_num, opt.test_num)

		som = MiniSom(opt.map_size, opt.map_size, m, sigma=opt.sigma, learning_rate=opt.learning_rate, neighborhood_function=opt.neighborhood_function)
		som.random_weights_init(features_['x_train'])
		som.train_batch(features_['x_train'], opt.iter)
		# visualization - labels_record
		preds = som.labels_map(features_['x_train'], features_['d_train'])
		fig = plt.figure(figsize=(opt.map_size, opt.map_size))
		grid = plt.GridSpec(opt.map_size, opt.map_size)
		for position in preds.keys():
		    label_fracs = [preds[position][l] for l in [0, 1]]
		    plt.subplot(grid[position[0], position[1]], aspect=1)
		    patches, texts = plt.pie(label_fracs)
		fig.legend(patches, ['face%d' % p for p in [0, 1]], loc='upper center', ncol=2)
		fig.savefig('imgs/som_%d' % opt.map_size)

	# for m in ms:
	# 	accs = []
	# 	for i in range(40):
	# 		for j in range(i, 40):
	# 			faces = [i, j]
	# 			features = np.load(opt.feature_path+'feature_'+str(m)+'.npy') # pca features
	# 			features = np.concatenate((features[faces[0]*10: faces[0]*10+10],features[faces[1]*10: faces[1]*10+10]), axis=0) # 只取两类人脸
	# 			# features = features[10:30]
	# 			features_ = get_sets(features, opt.train_num, opt.val_num, opt.test_num)

	# 			som = MiniSom(opt.map_size, opt.map_size, m, sigma=opt.sigma, learning_rate=opt.learning_rate, neighborhood_function=opt.neighborhood_function)
	# 			som.random_weights_init(features_['x_train'])
	# 			som.train_batch(features_['x_train'], opt.iter)
	# 			# visualization - labels_record
	# 			preds = som.labels_map(features_['x_test'], features_['d_test'])
	# 			# fig = plt.figure(figsize=(opt.map_size, opt.map_size))
	# 			# grid = plt.GridSpec(opt.map_size, opt.map_size)
	# 			# for position in preds.keys():
	# 			#     label_fracs = [preds[position][l] for l in [0, 1]]
	# 			#     plt.subplot(grid[position[0], position[1]], aspect=1)
	# 			#     patches, texts = plt.pie(label_fracs)
	# 			# fig.legend(patches, ['face%d' % p for p in [0, 1]], 
	# 			#            loc='upper center', ncol=2)
	# 			# fig.savefig('imgs/som_%d' % m)
	# 			# plt.close(fig)

	# 			result_map = assign_labels_to_nerons(preds, opt.map_size)
	# 			acc = som_acc(som, features_['x_test'], features_['d_test'], result_map)
	# 			# print(i, j)
	# 			# print(acc)
	# 			accs.append(acc)
	# 	print(np.mean(accs))

# 10304 0.97817
# 50 0.980976
# 100 0.979146
# 150 0.979512
# 200 0.979024
# 250 0.97878
# 300 0.97756
# 350 0.97829
if __name__ == '__main__':
    main()