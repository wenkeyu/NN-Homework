import argparse
import numpy as np
import models
import matplotlib.pyplot as plt
from sklearn import svm


class bp_config(object):
    def __init__(self):
        self.input_size = 1
        self.hidden_size = 16
        self.output_size = 1
        self.learning_rate = 0.4
        self.lr_decay = 0.9
        self.decay_step = 100000
        self.num_iters = 500000
        self.record_step = 100
        self.val_step = 100
        self.log_step = 10000


def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--model', default='svm', type=str)
    parser.add_argument('--func', default='func_1', type=str)
    parser.add_argument('--train_num', default=2000, type=int)
    parser.add_argument('--val_num', default=2000, type=int)
    parser.add_argument('--test_num', default=200, type=int)
    opt = parser.parse_args()

    func = opt.func
    funcname = func.split('_')[1]
    if funcname == '1':
        function = func_1
    elif funcname == '2':
        function = func_2
    elif funcname == '3':
        function = func_3
    data = gen_data(function, opt.train_num, opt.val_num, opt.test_num)
    # x:inpit_data d:teacher
    x_train, d_train = data['x_train'], data['d_train']
    x_val, d_val = data['x_val'], data['d_val']
    x_test, d_test = data['x_test'], data['d_test']

    # if opt.model == 'per':
    #     model = models.Perceptron(1, 1)
    #     loss_history, err_history = model.train(x_train, d_train, x_val=x_val, d_val=d_val)
    if opt.model == 'bp':
        config = bp_config()
        model = models.BpNet(config)
        loss_history, err_history = model.train(x_train, d_train, x_val=x_val, d_val=d_val)
    elif opt.model == 'rbf':
        model = models.RBF(1, 20, 1) # 27 for fun_3 20 for func_2 21 for func_1
        params = model.train(x_train, d_train)
        sigma_2, t = params['sigma_2'], params['t']
    elif opt.model == 'svm':
        model = svm.SVR(kernel='rbf', C=1.0, epsilon=1e-9)
        model.fit(x_train, d_train)

    y_test = model.predict(x_test)
    mse = np.mean((y_test - d_test) ** 2)
    print('mse=%e' % mse)
    # print(mse)
    # if opt.model == 'per' or opt.model == 'bp':
    #     # loss fig
    #     fig = plt.figure(1, figsize=(6, 4.5))
    #     plt.plot(loss_history['iter'], loss_history['hist'])
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Loss')
    #     plt.title('Loss after Each Iteration')
    #     plt.savefig('imgs/'+opt.model+'_loss_'+str(funcname)+'.png')

    #     plt.figure(2, figsize=(6, 4.5))
    #     plt.plot(err_history['iter'], err_history['hist'])
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Mean squared error')
    #     plt.title('Mean Squared Error')
    #     plt.savefig('imgs/'+opt.model+'_err_'+str(funcname)+'.png')

    x, y = function(10000, random=False)
    fig = plt.figure(1, figsize=(10, 7.5))
    plt.plot(x, y)
    plt.scatter(x_test, y_test, marker='o', color='black', s=30)
    a1 = fig.axes[0]
    a1.set_xlabel('x')
    a1.set_ylabel('y')
    plt.legend(['Groundtruth', 'Predicted'], loc='upper right')
    a2 = a1.twinx()
    print(y_test.shape,d_test.shape)

    if funcname == '3':
        # print(funcname)
        y_test = y_test.reshape(-1,1)
    # print(y_test.shape,d_test.shape)
    err_test = y_test - d_test
    # print(err_test.shape)
    sort_mask = np.argsort(x_test[:, 0])
    # print(sort_mask)
    # print(x_test.shape)

    a2.plot(x_test[:, 0][sort_mask], err_test[sort_mask], ls=':', color='green')
    plt.legend(['Absolute Error'], loc='lower right')
    a2.set_ylabel('Absolute error')
    plt.title('Groundtruth, Predictions and Absolute Error')
    plt.show()
    fig.savefig('imgs/'+opt.model+'_test_'+str(funcname)+'.png')
    

def func_1(num, random=False):
    if random:
        x = np.random.uniform(-1 ,1, num) * 6 * np.pi 
        x[x==0] = 1e-7
    else:
        x = np.linspace(-6*np.pi, 6*np.pi, num)
        x[x==0] = 1e-7
    return x, 10*np.sin(x) / np.abs(x)

def func_2(num, random=False):
    if random:
        x = np.random.rand(num) * 20
        # print(np.max(x))
    else:
        x = np.linspace(0, 20, num)
        # print(x)
    return x, np.sin(x)

def func_3(num, random=False):
    if random:
        x1 = np.random.uniform(-10, -2, int(num/2))
        x2 = np.random.uniform(-2, -0.4, int(num/2))  
        # x = np.random.uniform(-10,-0.1,num)
        # x[x>=0 and x<0.5]=0.5
        # x[-0.5<=x<0]=-0.5
    else:
        x1 = np.linspace(-10, -2, int(num/2))
        x2 = np.linspace(-2, -0.4, int(num/2))  
        # x = np.linspace(-10,-0.7,num)
        # x[0<=x<0.5]=0.5
        # x[-0.5<=x<0]=-0.5
    x = np.append(x1,x2)
    return x, 1 / (x ** 3)


def gen_data(function, num_train, num_val, num_test):
    """
    generate train、val、test data
    # """
    if function == func_3:
        num_t = num_train + num_val + num_test
        xs, ds = function(num_t, random=True)
        xs = xs.reshape(-1, 1)
        x_train = np.append(xs[0:int(num_train/2)],xs[int(num_t/2):int(num_t/2)+int(num_train/2)])
        d_train = np.append(ds[0:int(num_train/2)],ds[int(num_t/2):int(num_t/2)+int(num_train/2)])
        x_val = np.append(xs[int(num_train/2):int(num_train/2)+int(num_val/2)],xs[int(num_t/2)+int(num_train/2):int(num_t/2)+int(num_train/2)+int(num_val/2)])
        d_val = np.append(ds[int(num_train/2):int(num_train/2)+int(num_val/2)],ds[int(num_t/2)+int(num_train/2):int(num_t/2)+int(num_train/2)+int(num_val/2)])
        x_test = np.append(xs[int(num_train/2)+int(num_val/2):int(num_train/2)+int(num_val/2)+int(num_test/2)],xs[int(num_t/2)+int(num_train/2)+int(num_val/2):int(num_t/2)+int(num_train/2)+int(num_val/2)+int(num_test/2)])
        d_test = np.append(ds[int(num_train/2)+int(num_val/2):int(num_train/2)+int(num_val/2)+int(num_test/2)],ds[int(num_t/2)+int(num_train/2)+int(num_val/2):int(num_t/2)+int(num_train/2)+int(num_val/2)+int(num_test/2)])
        x_train = x_train.reshape(-1, 1)
        d_train = d_train.reshape(-1, 1)
        x_val = x_val.reshape(-1, 1)
        d_val = d_val.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)
        d_test = d_test.reshape(-1, 1)
    else:
        xs, ds = function(num_train + num_val + num_test, random=True)
        xs = xs.reshape(-1, 1)
        x_train = xs[0:num_train]
        d_train = ds[0:num_train]
        x_val = xs[num_train:num_train+num_val]
        d_val = ds[num_train:num_train+num_val]
        x_test = xs[num_train+num_val:num_train+num_val+num_test]
        d_test = ds[num_train+num_val:num_train+num_val+num_test]    
    return {'x_train': x_train, 'd_train': d_train, 
            'x_val': x_val, 'd_val': d_val, 
            'x_test': x_test, 'd_test': d_test }


if __name__ == '__main__':
    main()

