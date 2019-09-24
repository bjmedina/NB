# Bryan Medina
# CAP5610 - Machine Learning
# Naive Bayes w/ Iris Dataset

####### Imports ########
import sys

import numpy as np
import pandas as pd
########################


# This plots the confusion matrix at the end of the program
show_matrix = True

if(show_matrix):
    import matplotlib.pyplot as plt
    
class NaiveBayes():
    '''
    Description
    -----------
    Class that can handle data and run the naive bayes classifier on the iris dataset. 
    '''
    
    def __init__(self, training_set, num_features, num_classes):
        # Store the training set
        self.training_set = training_set
        
        self.num_features = num_features
        self.num_classes  = num_classes

        # MLE estimates for Sigma. One sigma for each class and feature 
        self.sigma        = np.zeros((self.num_classes, self.num_features))
        # MLE estimates for Mu. One mu for each class and feature
        self.mu           = np.zeros((self.num_classes, self.num_features))


    def train(self):
        '''
        Description
        -----------
        'train' gets the likelihood and prior probabilities ready for testing.

        Input
        -----
        None
        
        Output
        ------
        None
        '''
        
        #### Priors ###
        # Prior probabilities for class d will be called theta(d), where d is the class
        self.theta = np.array([ np.sum(data.iloc[self.training_set][4] == cl) for cl in classes.keys() ]) / len(self.training_set)
        ###############

        
        ### Likelihoods ###        
        # Mask matrix where each row represents a different class,  
        in_classy = [[classes[data.iloc[int(i)][4]] == y for i in self.training_set] for y in range(self.num_classes)]

        # Vector of nth feature across all training examples
        xn = data.iloc[self.training_set] 
            
        for y in range(self.num_classes):
            # Number of examples in class y
            my = (self.theta[y]*len(self.training_set))
            
            for n in range(self.num_features):
                # MLE estimates for mu
                self.mu[y][n] = (1./my) * np.dot(xn[n], in_classy[y])
                
                # MLE estimates for sigma
                self.sigma[y][n] = (1. / (my-1)) * np.dot((xn[n]-self.mu[y][n])**2, in_classy[y])
        ###################


    def predict(self, test_example):
        '''
        'predict' calculates the most probable class given the data, using this idea
        Y* = argmax(Y in the set of classes) [P(X1|Y)*P(X2|Y)*...*P(Xn|Y)*P(Y)]
           = argmax(Y in the set of classes) [log(P(X1|Y))+log(P(X2|Y))+...+log(P(Xn|Y))+log(P(Y))]
        
        Input
        -----
        'test_ex':float 
        
        Output
        ------
        tuple of class prediction and actual class, both integers.
        '''      
        actual    = test_example[4]
        predicted = [-1,-1e9] # [class index, likelihood]

        for y in range(self.num_classes):
            prob = np.sum([self.logLikelihood(test_example, y, i) for i in range(self.num_features)]) + np.log(self.theta[y])
            
            if(prob >= predicted[1]):
                predicted[0] = y
                predicted[1] = prob

        return predicted[0], classes[actual]

    def logLikelihood(self, test_ex, y, n):
        '''
        Calculates the log likelihood of the test example given the MLE estimates of mu and sigma, and the features of the test example.

        Input
        -----
        'test_ex':float 
        'n': integer: specifies nth feature
        
        Output
        ------
        scalar probability
        '''
        a = ((2*np.pi)**(0.5)) * self.sigma[y][n]
        a = 1. / a
        b = -((test_ex[n] - self.mu[y][n]) ** 2)
        c = 2 * (self.sigma[y][n]**2)
        
        return np.log(a) + (b/c)
    

####### Getting data / initializing variables ###
data      = pd.read_csv('/home/bjm/Documents/School/fall2019/CAP5610/assignments/a1/data/iris.data', header=None) # CHANGE ME
classes   = {} # Dictionary will store class name with index
class_idx = 0 
num_feats = len(data.iloc[0])-1 # Number of features

# Code to get the index that we'll be using for a particular class
for cl in data.iloc[:][4]:
    if not (cl in classes.keys()):
        classes[cl] = class_idx
        class_idx = class_idx + 1
        
KFolds    = 5 # Specify number of folds for k-fold cross validation
K         = len(data) # Number of examples of which we are going to split
confusion = np.zeros((class_idx,class_idx)) # Three classes, so confusion matrix is 3 X 3
##################################################


###### Naive Bayes (with continuous features) ######
# 1. Shuffle indices for the data set
fold_size  = K / KFolds
all_splits = np.zeros((KFolds, K))

# 2. Divide it into K groups and get indices for K groups
for i in range(0, KFolds):
    # We'll use the first 'fold_size' group as the test set. Rest is training. 
    all_splits[i] = np.random.permutation(np.arange(K))

# 3. Repeat KFolds times
for fold in range(KFolds):
    
    # 4. Reserve first subgroup for testing. Train and everything that isn't is in the test set
    test_set = all_splits[fold][0:int(fold_size)]
    training = all_splits[fold][int(fold_size):]
    
    assert len(test_set) >= 1
    assert len(training) >= 1

    # 5. Run NB on training set
    NB = NaiveBayes(training, num_feats, class_idx)
    NB.train()

    for test in test_set:
        # 6. Test with the test set
        predicted, actual = NB.predict(data.iloc[int(test)])

        # 7. Save the rpecision, recall, and add to confusion matrix
        confusion[predicted][actual] += 1

# 8. Report
accuracy = (np.sum([confusion[i][i] for i in range(len(confusion))]) / (K)) * 100

if(show_matrix):
    plt.matshow(confusion)
    plt.title("Confusion Matrix for naive bayes %.2f %% Accuracy" % (accuracy))
    plt.colorbar()
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.show()
else:
    print("Confusion Matrix:")
    print(confusion)
    print("\nAccuracy: %.2f %%" % (accuracy))

