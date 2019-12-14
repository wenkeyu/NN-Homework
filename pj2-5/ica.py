import numpy as np
from engine import ICA, whiten, match_and_recover
import argparse
from PIL import Image


def get_mixed(s):
	np.random.seed(20) # 20 for r4
	A = np.random.rand(5, s.shape[0])
	# A = np.array(((0.5,0.25,0.25,0.25),(0.5,0.4,0.2,0.2),(0.5,0.2,0.4,0.2),(0.5,0.2,0.4,0.2),(0.5,0.2,0.2,0.4)))
	# print(A)
	A = np.exp(-A*A) / np.sum(np.exp(- A * A), axis=1, keepdims=True)
	x = np.matmul(A, s)
	return x


def save_img(arr, img_size, path):
	arr[arr > 1.0] = 1.0
	arr[arr < 0.0] = 0.0
	Image.fromarray((arr.reshape(img_size)*255.0).astype(np.uint8)).save(path)


class ICA_config(object):

	def __init__(self):
		self.lr = 2.0
		self.iter = 10000


def main():
	parser = argparse.ArgumentParser()    
	parser.add_argument('--img_path', default='data/ICA/', type=str)
	parser.add_argument('--s', default=4, type=int)
	parser.add_argument('--method', default='hj', type=str)
	# parser.add_argument('--test_num', default=4, type=int)
	# parser.add_argument('--model', default='bp', type=str)
	opt = parser.parse_args()
	IMG_SIZE = (256, 512)
	# get raw images
	for i in range(opt.s):
	    img = Image.open(opt.img_path+str(i+1)+'.bmp')
	    img = (np.array(img).reshape(1,-1))/255
	    if i==0:
	        imgs = img
	    else:
	        imgs = np.concatenate((imgs, img), axis=0)
	# print(imgs[0:4])
	
	x = get_mixed(imgs[0:4])[0:5]
	# save mixed imgs
	# for i in range(5):
	# 	path = 'imgs/mixed_' + str(i+1) + '.bmp'
	# 	save_img(x[i], IMG_SIZE, path)
	# whitten
	z = whiten(x, truth=4)
	config = ICA_config()
	ica = ICA(z, opt.method, config)
	ys = ica.exec()[0]
	# ys = ica.get_results()[0]
	coef_mat, labels, imgs_ = match_and_recover(imgs[0:4], ys)
	# for i in range(4):
	#     save_img(x[i], IMG_SIZE, 'imgs/mixed_%d.bmp' % i)
	print(labels,imgs_)
	print(coef_mat)
	for i in range(z.shape[0]):
		save_img(imgs_[i], IMG_SIZE, 'imgs/recv_%s_%d_%d.bmp' % (opt.method, x.shape[0], i))
		# save_img(imgs_[i], IMG_SIZE, 'imgs/recv_%s_%d_label.bmp' % (opt.method, labels[i]))


if __name__ == '__main__':
    main()