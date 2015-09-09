#Author: Arjun Chakraborty
#Implementation of a one layer autoencoder and evaluation using KS test.


import numpy as np
import theano as th
from theano import tensor as T
from numpy import random as rng
import scipy
from scipy import stats
import genFeature
from genFeature import generate_features
train_set,test_set = generate_features()

class AutoEncoder(object):
    
    def __init__(self, X,Y, hidden_size, activation_function,
                 output_function,Weight= True):
        
        #Input for training set and conversion into shared variable for theano 
        self.X=X
        self.X=th.shared(name='X', value=np.asarray(self.X, 
        				dtype=th.config.floatX),borrow=True)

        # Input for test set and conversion into shared variable for theano
        self.Y=Y
        self.Y=th.shared(name='Y', value=np.asarray(self.Y, 
                        dtype=th.config.floatX),borrow=True)


        # Shape of training set
        self.n = X.shape[1]
        self.m = X.shape[0]
        
        # Shape of test set
        self.n1 = Y.shape[0]
        self.m1 = Y.shape[1]

        
        self.hidden_size=hidden_size
        
        # Choose between random and sparse initialization
        
        if weight==True:

            # Random Initialization
            initial_W = np.asarray(rng.uniform(
                     low=-4 * np.sqrt(6. / (self.hidden_size + self.n)),
                     high=4 * np.sqrt(6. / (self.hidden_size + self.n)),
                     size=(self.n, self.hidden_size)), dtype=th.config.floatX)
        else:

            # Sparse Initialization dependent on num_connections and scale
            num_connections = 10
            scale = 0.8

            indices = range(self.n)
                initial_W = numpy.zeros((self.n, self.hidden_size),dtype=theano.config.floatX)
                
                for i in range(self.hidden_size):
                    random.shuffle(indices)
                    for j in indices[:num_connections]:
                        initial_W[j,i] = random.gauss(0.0, scale)
        
       
        # Declaration of parameters
        self.W = th.shared(value=initial_W, name='W', borrow=True)
        
        self.b1 = th.shared(name='b1', value=np.zeros(shape=(self.hidden_size,),
                            dtype=th.config.floatX),borrow=True)
        
        self.b2 = th.shared(name='b2', value=np.zeros(shape=(self.n,),
                            dtype=th.config.floatX),borrow=True)
        
        self.activation_function=activation_function
        
        self.output_function=output_function



    """ This function is responsible for training the model. 
        It takes in number of epochs,mini-batch size and learning rate """

    def train(self, n_epochs=5, mini_batch_size=10, initial_learning_rate=0.1):
        
        index = T.lscalar()
        x=T.matrix('x')
        params = [self.W, self.b1, self.b2]
        hidden = self.activation_function(T.dot(x, self.W)+self.b1)
        
        output = T.dot(hidden,T.transpose(self.W))+self.b2
        
        output = self.output_function(output)

        # Learning rate is a shared variable so that it can be decayed
        learning_rate = theano.shared(np.asarray(initial_learning_rate,
        dtype=theano.config.floatX))
         
        #Use mean squared error
        L = T.sum((x-output)**2,axis=1)
        cost=L.mean()       
        updates=[]
         
        #Return gradient with respect to W, b1, b2.
        gparams = T.grad(cost,params)
         
        #Create a list of 2 tuples for updates.
        for param, gparam in zip(params, gparams):
            updates.append((param, param-learning_rate*gparam))
         
        #Train given a mini-batch of the data.
        train = th.function(inputs=[index], outputs=[cost], updates=updates,
                            givens={x:self.X[index:index+mini_batch_size,:]})
                             
        # Start training
        import time
        start_time = time.clock()
        samples = []
        
        epoch = 0
        while (epoch < n_epochs):
            
            epoch = epoch  + 1
            print "Epoch:",epoch
            
            for row in xrange(0,self.m, mini_batch_size):
                
                samples.append(train(row))
                samples_epoch = np.array(samples)

            print " The mean loss for each epoch is:"
            print np.mean(samples_epoch)
            

            # decay of learning rate
            new_learning_rate = learning_rate.get_value() * 0.985
            learning_rate.set_value(np.cast[th.config.floatX](new_learning_rate))

        end_time = time.clock()
        
        print "Average time per epoch=", (end_time-start_time)/n_epochs

        

     """ This function is responsible for testing the model. In this function
         the L2 norm of the reconstruction error of training and test set is 
         computed. In the KS_test function, the distribution is assessed. """
     
     def test(self,mini_batch_size=5):
        
        index = T.lscalar()
        x=T.matrix('x')
        y=T.matrix('y')
       
        hidden = self.activation_function(T.dot(x, self.W)+self.b1)
        
        output = T.dot(hidden,T.transpose(self.W))+self.b2
        
        output = self.output_function(output)

      
        #Finding L2 norm of the test set
        L = T.sum(T.sqrt((x-output)**0.5,axis=1))
        cost=L.mean()       
        updates=[]
        
        
        #Test given a mini-batch of the data.
        test= th.function(inputs=[index], outputs=[cost],
                            givens={x:self.Y[index:index+mini_batch_size,:]})
                             
 
        import time
        start_time = time.clock()
        samples_test = []
        
        for row in xrange(0,self.m1, mini_batch_size):
                
            samples_test.append(test(row))
            samples_test_epoch = np.array(samples_test)

        print " The mean loss for test set is:"
        print np.mean(samples_test_epoch)


        

        # Finding the L2 norm of the training set

        R = T.sum(T.sqrt((x-output)**0.5,axis=1))
        cost1=T.mean()       
        

        test= th.function(inputs=[index], outputs=[cost1],
                            givens={x:self.X[index:index+mini_batch_size,:]})
                             
        
        samples_train=[]

        for row in xrange(0,self.m, mini_batch_size):
                
            samples_train.append(test(row))
            samples_train_epoch = np.array(samples_train)

        print " The mean reconstruction error for training set is:"
        print np.mean(samples_train_epoch)


        # The samples of the L2 norm of the training and test set are converted into 1D numpy arrays
        samples_train_epoch = np.ravel(samples_train_epoch)
        samples_test_epoch =  np.ravel(samples_test_epoch) 
            
            
        return samples_train_epoch,samples_test_epoch

   
    """ This function is directly responsible for carrying out the KS test using a scipy function """
    def KS_test (samples_train_epoch,samples_test_epoch):

        print stats.ks_2samp(samples_train_epoch, samples_test_epoch)


""" Test function """
def final_test():
    X=training_set
    Y=test_set
    activation_function = T.nnet.sigmoid
    output_function=activation_function
    A = AutoEncoder(X, 100, activation_function, output_function,weight=True)
    A.train()
    samples_train_epoch,samples_test_epoch=A.test()
    A.KS_test(samples_train_epoch,samples_test_epoch)

if __name__ == "__main__":

	final_test()		



