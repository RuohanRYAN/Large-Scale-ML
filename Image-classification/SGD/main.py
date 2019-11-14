import os
import numpy as np 
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables


def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = numpy.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        Xs_tr = numpy.ascontiguousarray(Xs_tr)
        Ys_tr = numpy.ascontiguousarray(Ys_tr)
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = numpy.ascontiguousarray(Xs_te)
        Ys_te = numpy.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the gradient of the multinomial logistic regression objective, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    # TODO students should implement this
    d,n = Xs.shape
    c,_ = Ys.shape
    nSample = len(ii)
    Xsamp = np.reshape(Xs[:,ii],(d,nSample))
    Ysamp = np.reshape(Ys[:,ii],(c,nSample))
    sfmaxSum = np.sum(np.exp(np.dot(W,Xsamp)),axis = 0)
    sfmax = (np.exp(np.dot(W,Xsamp))/sfmaxSum)-Ysamp
    grad = np.zeros((c,d))
    for i in range(nSample):
        grad = grad+np.dot(np.reshape(sfmax[:,i],(c,1)),np.reshape(Xsamp[:,i],(1,d)))
    grad = grad/nSample + gamma*W
    return grad 

# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should use their implementation from programming assignment 1
    d,n = Xs.shape
    c,_ = Ys.shape
    yTrain = np.argmax(np.dot(W,Xs),axis=0)
    yExpect = np.argmax(Ys,axis=0)
    nWrong = len(np.argwhere(yTrain - yExpect))
    return nWrong/n

# ALGORITHM 1: run stochastic gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of iterations (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" iterations
def stochastic_gradient_descent(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
    # TODO students should implement this
    d,n = Xs.shape
    c,_ = Ys.shape
    W = W0
    T = num_epochs
    gradient = []
    for i in range(T*n):
        ii = [np.random.randint(0,n)]
        W = W - alpha* multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
        if((i+1)%monitor_period==0):
            gradient.append(W)
    return gradient

# ALGORITHM 2: run stochastic gradient descent with sequential sampling order
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of iterations (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" iterations
def sgd_sequential_scan(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
    # TODO students should implement this
    d,n = Xs.shape
    c,_ = Ys.shape
    W = W0
    gradient = []
    count = 0
    for t in range(num_epochs):
        for i in range(n):
            ii = [i]
            W = W - alpha* multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
            count+=1
            if((count)%monitor_period==0):
                gradient.append(W)
    return gradient 


# ALGORITHM 3: run stochastic gradient descent with minibatching
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_minibatch(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    d,n = Xs.shape
    c,_ = Ys.shape
    W = W0
    gradient = []
    count = 0
    T = int(num_epochs * n/B)
    for t in range(T):
#         ii = np.random.randint(0,n,size = B)
        ii = [np.random.randint(0,n)for j in range(B)]
        W = W - alpha* multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
#         W = W - alpha*np.mean([multinomial_logreg_grad_i(Xs, Ys, [i], gamma, W) for i in ii])
        count+=1
        if(count % monitor_period==0):
            gradient.append(W)
    return gradient 

# ALGORITHM 4: run stochastic gradient descent with minibatching and sequential sampling order
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_minibatch_sequential_scan(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    d,n = Xs.shape
    c,_ = Ys.shape
    W = W0
    gradient = []
    count = 0
    for t in range(num_epochs):
        for i in range(int(n/B)):
            b = i*B
#             ii = np.random.randint(b,b+B,size = B)
            rg = np.arange(b, b+B)
            ii = np.random.choice(rg,size = B, replace=False)
            W = W - alpha* multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
            count+=1
            if(count % monitor_period==0):
                gradient.append(W)
    return gradient 

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # TODO add code to produce figures
    d,n = Xs_tr.shape
    c,_ = Ys_tr.shape
    print('d is :',d)
    print('n is :',n)
    print('c is :',c)
    gamma = 0.0001
    alpha = 0.001
    W0 = np.zeros((c,d))
    num_epochs = 10
    monitor_period = 6000
    gradient = stochastic_gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, num_epochs, monitor_period)
    x1 = range(len(gradient));y1 = []
    for i in range(len(gradient)):
        y1.append(multinomial_logreg_error(Xs_tr, Ys_tr, gradient[i]))
        
    gradient2 = sgd_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, num_epochs, monitor_period)
    x2 = range(len(gradient2));y2 = []
    for i in range(len(gradient2)):
        y2.append(multinomial_logreg_error(Xs_tr, Ys_tr, gradient2[i]))

    pyplot.figure(1);pyplot.plot(x1,y1,x2,y2);pyplot.xlabel('number of iterations');pyplot.ylabel('error rate')
    pyplot.title('error rate with SGD alpha = 0.001');pyplot.legend(('SGD','SGD with sequential sampling'));
    pyplot.savefig('error rate SGD')
    
    alpha = 0.05
    B = 60
    monitor_period = 100
    gradient3 = sgd_minibatch(Xs_tr, Ys_tr, gamma, W0, alpha, B, num_epochs, monitor_period)
    x3 = range(len(gradient3));y3 = []
    for i in range(len(gradient3)):
        y3.append(multinomial_logreg_error(Xs_tr, Ys_tr, gradient3[i]))
        
    gradient4 = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, num_epochs, monitor_period)
    x4 = range(len(gradient4));y4 = []
    for i in range(len(gradient4)):
        y4.append(multinomial_logreg_error(Xs_tr, Ys_tr, gradient4[i]))

    pyplot.figure(2);pyplot.plot(x3,y3,x4,y4);pyplot.xlabel('number of iterations');pyplot.ylabel('error rate')
    pyplot.title('error rate with SGD mini batching');pyplot.legend(('SGD with mini batching','SGD mini batching with sequential sampling'));
    pyplot.savefig('error rate SGD with miniBatch')
    
    
    
    print('training error for SGD is: ',multinomial_logreg_error(Xs_tr, Ys_tr, gradient[-1]))
    print('training error for SGD with sequential sampling is: ',multinomial_logreg_error(Xs_tr, Ys_tr, gradient2[-1]))
    print('training error for SGD with mini batch is: ',multinomial_logreg_error(Xs_tr, Ys_tr, gradient3[-1]))
    print('training error for SGD with mini batch seqential is: ',multinomial_logreg_error(Xs_tr, Ys_tr, gradient4[-1]))
          
    print('testing error for SGD is: ',multinomial_logreg_error(Xs_te, Ys_te, gradient[-1]))
    print('testing error for SGD with sequential sampling is: ',multinomial_logreg_error(Xs_te, Ys_te, gradient2[-1]))
    print('testing error for SGD with mini batch is: ',multinomial_logreg_error(Xs_te, Ys_te, gradient3[-1]))
    print('testing error for SGD with mini batch seqential is: ',multinomial_logreg_error(Xs_te, Ys_te, gradient4[-1]))
          
    pyplot.figure(3);pyplot.plot(x1,y1,x2,y2,x3,y3,x4,y4);pyplot.xlabel('number of iterations');pyplot.ylabel('error rate')
    pyplot.title('error rate with different types of SGD ')
    pyplot.legend(('SGD','SGD with sequential sampling','SGD with mini batching','SGD mini batching with sequential sampling'));
    pyplot.savefig('error rate total')
#######testing results #####################


