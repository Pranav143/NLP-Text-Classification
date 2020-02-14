import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  

models = ["SGDClassifier", "GaussianNB", "RandomForestClassifier", "MLPClassifier", "AdaBoostClassifier"]


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    # Get sum of all elements 
    total_classifications = C.sum()
    # Get total correct classifications, which is the sum of the diagonal
    correct_classifications = C.trace()
    if total_classifications == 0:
        return 0.0
    acc = correct_classifications / total_classifications
    return acc

def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    recall_list = []
    for i in range(C.shape[0]): # Iterate through all the rows 
        num_predictions = C[i].sum() # Get sum of that row 
        num_correct_predictions = C[i][i]
        if num_predictions == 0:
            recall_list.append(0.0)
        else:
            recall_list.append(num_correct_predictions/num_predictions)
    return recall_list 

def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    precision_list = []
    sum_of_columns = np.sum(C, axis=0) # Returns an array with sum of all columns 
    for i in range(len(sum_of_columns)):
        num_correct = C[i][i]
        if sum_of_columns[i] == 0:
            precision_list.append(0.0)
        else:
            precision_list.append(num_correct / sum_of_columns[i])
    return precision_list

def load_model(model_name):
    if model_name == "SGDClassifier":
        model = SGDClassifier()
        
    if model_name == "GaussianNB":
        model = GaussianNB()

    if model_name == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators = 10, max_depth = 5)
    
    if model_name == "MLPClassifier":
        model = MLPClassifier(alpha=0.05)
 
    if model_name == "AdaBoostClassifier":
        model = AdaBoostClassifier()
        
    return model
            
            
def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    acc = []
    prec = []
    rec = []
    conf = []
    for i in models:
        if i == "SGDClassifier":
            model = SGDClassifier()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            confusionMatrix = confusion_matrix(y_test, predictions)
            conf.append(confusionMatrix)
            acc.append(accuracy(confusionMatrix))
            prec.append(precision(confusionMatrix))
            rec.append(recall(confusionMatrix))
        
        if i == "GaussianNB":
            model = GaussianNB()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            confusionMatrix = confusion_matrix(y_test, predictions)
            conf.append(confusionMatrix)
            acc.append(accuracy(confusionMatrix))
            prec.append(precision(confusionMatrix))
            rec.append(recall(confusionMatrix))
        
        if i == "RandomForestClassifier":
            model = RandomForestClassifier(n_estimators = 10, max_depth = 5)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            confusionMatrix = confusion_matrix(y_test, predictions)
            conf.append(confusionMatrix)
            acc.append(accuracy(confusionMatrix))
            prec.append(precision(confusionMatrix))
            rec.append(recall(confusionMatrix))
        
        if i == "MLPClassifier":
            model = MLPClassifier(alpha=0.05)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            confusionMatrix = confusion_matrix(y_test, predictions)
            conf.append(confusionMatrix)
            acc.append(accuracy(confusionMatrix))
            prec.append(precision(confusionMatrix))
            rec.append(recall(confusionMatrix))
        
        if i == "AdaBoostClassifier":
            model = AdaBoostClassifier()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            confusionMatrix = confusion_matrix(y_test, predictions)
            conf.append(confusionMatrix)
            acc.append(accuracy(confusionMatrix))
            prec.append(precision(confusionMatrix))
            rec.append(recall(confusionMatrix))
            
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        for i in range(len(models)):
            outf.write(f'Results for {models[i]}:\n')  
            outf.write(f'\tAccuracy: {acc[i]:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in rec[i]]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in prec[i]]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf[i]}\n\n')
    
    # To calculate iBest, I'm simply going to take highest accuracy
    max_acc = max(acc)
    acc_dict = dict(zip(acc, np.arange(5))) # Make a small dictionary to get the index of max_acc
    iBest = acc_dict.get(max_acc)
    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    model_name = models[iBest] 
    model = load_model(model_name)
    num_train = [1000, 5000, 10000, 15000, 20000]
    
    # Just for return value 
    X_1k = X_train[:1000]
    y_1k = y_train[:1000]
    
    acc = []
    for i in num_train:
        train_data = X_train[:i]
        train_label = y_train[:i]
        model.fit(train_data, train_label)
        predictions = model.predict(X_test)
        confusionMatrix = confusion_matrix(y_test, predictions)
        acc.append(accuracy(confusionMatrix))
        
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        for i in range(len(num_train)):
            outf.write(f'{num_train[i]}: {acc[i]:.4f}\n')
    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    # PART 1 ##################################################################
    pp_vals = [] # store p values
    model = load_model(models[i]) # load model
 
    # Grab the pvalues_ for k = 5 to 50
    for i in range(2): # k = 5 to k = 50
        selector = SelectKBest(f_classif, 5*(10**i)) 
        selector.fit(X_train[:32000], y_train[0:32000])
        pp = selector.pvalues_
        pp_vals.append(list(pp))
    
    # PART 2 ##################################################################
    acc_part2 = [] # index 0 = full dataset. index 1 = 1k dataset
    
    # K = 5 features on 32k set
    selector1 = SelectKBest(f_classif, 5)
    X_new1 = selector1.fit_transform(X_train[:32000], y_train[0:32000])
    pp1 = selector1.pvalues_
    indices_32k = selector1.get_support(indices=True) # PART 4 SOL ############
    
    # 32k model and accuracy 
    model.fit(X_new1, y_train[0:32000])
    predictions = model.predict(selector1.transform(X_test))
    confusionMatrix = confusion_matrix(y_test, predictions)
    acc_part2.append(accuracy(confusionMatrix))
    
    # k = 5 on 1k set 
    selector2 = SelectKBest(f_classif, 5)
    X_new2 = selector2.fit_transform(X_1k, y_1k)
    pp2 = selector2.pvalues_
    indices_1k = selector2.get_support(indices=True)
    
    # 1k model and accuracy
    model.fit(X_new2, y_1k)
    predictions = model.predict(selector2.transform(X_test))
    confusionMatrix = confusion_matrix(y_test, predictions)
    acc_part2.append(accuracy(confusionMatrix))

    # PART 3 ##################################################################
    # Intersection of feature indices 
    intersection = np.intersect1d(indices_1k, indices_32k)
    
    # WRITE TO FILE ###########################################################
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        for k in [0,1]:
            outf.write(f'{5*(10**k)} p-values: {[round(pval, 4) for pval in pp_vals[k]]}\n')
        
        outf.write(f'Accuracy for 1k: {acc_part2[1]:.4f}\n')
        outf.write(f'Accuracy for full dataset: {acc_part2[0]:.4f}\n')
        outf.write(f'Chosen feature intersection: {intersection}\n')
        outf.write(f'Top-5 at higher: {indices_32k}\n')
    

def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    # Merge test and train for whole dataset
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train,y_test))
    
    kf = KFold(5, shuffle=True, random_state=42)
    mean_acc = [] # array to store mean accuracy per model
    acc = []
    kfold_accuracies = [[],[],[],[],[]] # array to track across folds 
    for i in range(5): # Iterate 5 times, once per model
        temp_acc = []
        index = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = load_model(models[i]) 
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            acc_score = accuracy_score(y_test, predictions)
            
            temp_acc.append(acc_score) # It goes to this array to get averaged
            kfold_accuracies[index].append(acc_score) # Tracks accuracy per fold
            index += 1
        mean_acc.append(np.average(temp_acc)) 
        acc.append(temp_acc) # This tracks in a batch the accuracy across all folds per model
    
    highest_acc_index = mean_acc.index(max(mean_acc)) # Index of best model 
    pvalues = []
    # Now, calculate the stats values 
    for i in range(5):
        if i == highest_acc_index:
            continue
        else:
            a = acc[i]
            b = acc[highest_acc_index]
            S = ttest_rel(a,b)
            pvalues.append(S.pvalue)
            
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        for i in range(5):
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies[i]]}\n')
        outf.write(f'p-values: {[round(pval, 4) for pval in pvalues]}\n')



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    
    data_matrix = np.load(args.input)
    key = data_matrix.files[0] # necessary for loading a .npz file
    data = data_matrix[key] # Data is a numpy matrix with shape (40000, 174)
    
    labels = data[:, 173] # Extracts the last column of labels
    features = np.delete(data, 173, axis = 1) # drops the last column of labels
    # Now, split the data into 20% test, 80% training, with a random shuffle thrown in
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state = 42)
    
    output_dir = args.output_dir

    iBest = class31(output_dir, X_train, X_test, y_train, y_test)
    X_1k, y_1k = class32(output_dir, X_train, X_test, y_train, y_test, iBest)
    class33(output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(output_dir, X_train, X_test, y_train, y_test, iBest)