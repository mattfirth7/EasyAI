import random
import numpy as np
#import tensorflow as tf
import minepy as mp
from sklearn import preprocessing, neighbors, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle
import tkinter
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split


'''To anyone reading this: Yes I know this code would be much shorter by just
looping over all the classifiers and their hyperparameters from an array or something
but this is vastly more readable as a first draft

NOT OPERATIONAL: MAIN METHOD INCOMPLETE AND UNTESTED
'''



#Need to find a way to take all of the first values of the first n columns

def convert_to_sklearn_shape(x):
    converted_data = []
    for index in range(len(x[0])):
        row = []
        for feature in x:
            row.append(feature[index])
        converted_data.append(row)
    return converted_data

def convert_to_analysis_shape(x):
    converted_data = []
    for entry in x[0]:
        converted_data.append([entry])
    for index in range(1,len(x)):
        for col_index in range(len(x[index])):
            converted_data[col_index].append(x[index][col_index])
    
    return converted_data

def data_norm(x):
    for feature in x:
        mean = np.mean(feature)
        stdev = np.std(feature)
        feature = (feature - mean) / stdev
        
    return x


#still need to add optimization over hyperparameters
def k_nearest(x_train, y_train, x_test, y_test):
    clf = KNeighborsClassifier(n_neighbors = 3)
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    model = pickle.dumps(clf)
    return ['k_nearest', accuracy, model]

def support_vector_machine(x_train, y_train, x_test, y_test):
    accuracy_arr = []
    for kernel in ('linear', 'poly', 'rbf'):
        clf = svm.SVC(kernel = kernel, gamma = 2)
        clf.fit(x_train,y_train)
        
        accuracy = clf.score(x_test, y_test)
        accuracy_arr.append(accuracy)
        
    accuracy_argmax = np.argmax(accuracy_arr)
    if accuracy_argmax == 0:
        clf = svm.SVC(kernel = 'linear', gamma = 2)
        clf.fit(x_train, y_train)
        accuracy = clf.score(x_test, y_test)
        model = pickle.dumps(clf)
        return ['linear', accuracy, model]
    elif accuracy_argmax == 1:
        clf = svm.SVC(kernel = 'poly', gamma = 2)
        clf.fit(x_train, y_train)
        accuracy = clf.score(x_test, y_test)
        model = pickle.dumps(clf)
        return ['poly', accuracy, model]
    else:
        model = pickle.dumps(clf)
        return ['rbf', accuracy, model]

#still need to add optimization over hyperparameters
def gaussian_process_classifier(x_train, y_train, x_test, y_test):
    kernel = 1.0 * RBF(1.0)
    clf = GaussianProcessClassifier(kernel = kernel, random_state = 0)
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    model = pickle.dumps(clf)
    return ['gaussian_process', accuracy, model]

#still need to add optimization over hyperparamters
def decision_tree_classifier(x_train, y_train, x_test, y_test):
    clf = DecisionTreeClassifier(random_state = 0)
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    model = pickle.dumps(clf)
    return ['decision_tree', accuracy, model]

#still need to add optimization over hyperparameters
def random_forest_classifier(x_train, y_train, x_test, y_test):
    clf = RandomForestClassifier(max_depth = 2, random_state = 0)
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    model = pickle.dumps(clf)
    return ['random_forest', accuracy, model]

#still need to add optimization over hyperparameters
def adaboost_classifier(x_train, y_train, x_test, y_test):
    clf = AdaBoostClassifier(n_estimators = 100, random_state = 0)
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    model = pickle.dumps(clf)
    return ['adaboost', accuracy, model]

def native_bayes(x_train, y_train, x_test, y_test):
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    model = pickle.dumps(clf)
    return ['native_bayes', accuracy, model]

def quadratic_discriminant(x_train, y_train, x_test, y_test):
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    model = pickle.dumps(clf)
    return ['quadratic_discriminant', accuracy, model]

#Do I want to add something like rerunning classifiers with the highest accuracies
#to get a better estimate of mean accuracy?
def easy_classification(x_train, y_train, x_test, y_test):
    results_array = []
    results_array.append(k_nearest(x_train, y_train, x_test, y_test))
    print(k_nearest(x_train, y_train, x_test, y_test)[1])
    results_array.append(support_vector_machine(x_train, y_train, x_test, y_test))
    print(support_vector_machine(x_train, y_train, x_test, y_test)[1])
    results_array.append(decision_tree_classifier(x_train, y_train, x_test, y_test))
    print(decision_tree_classifier(x_train, y_train, x_test, y_test)[1])
    results_array.append(random_forest_classifier(x_train, y_train, x_test, y_test))
    print(random_forest_classifier(x_train, y_train, x_test, y_test)[1])
    results_array.append(adaboost_classifier(x_train, y_train, x_test, y_test))
    print(adaboost_classifier(x_train, y_train, x_test, y_test)[1])
    results_array.append(native_bayes(x_train, y_train, x_test, y_test))
    print(native_bayes(x_train, y_train, x_test, y_test)[1])
    #quadratic discriminant analysis is not currently working
    #results_array.append(quadratic_discriminant(x_train, y_train, x_test, y_test))
    
    accuracies_arr = []
    for row in results_array:
        accuracies_arr.append(row[1])
        
    accuracy_argmax = np.argmax(accuracies_arr)
    classifier = results_array[accuracy_argmax][0]
    model = results_array[accuracy_argmax][2]
    
    with open('easyml_classifier.pickle', 'wb') as f:
        f.write(model)
        f.close()
    
    print(classifier + " model chosen. Yielded max accuracy: " + str(results_array[accuracy_argmax][1]))
    print("Model saved as 'easyml_classifier.pickle'")
    return
    
#main method still work in progress
def main():
    print("Make sure the label variable is the last column in the csv")
    tkinter.Tk().withdraw()
    input_filename = askopenfilename()
    input_array = np.genfromtxt(input_filename, delimiter = ',', skip_header = 1)
    y = input_array[-1]
    x = np.delete(input_array,-1,0)
    print(x[:5])
    sklearn_compatible_array = convert_to_sklearn_shape(x)
    print(sklearn_compatible_array[:5])
    x_train, x_test, y_train, y_test = train_test_split(sklearn_compatible_array,y,test_size=0.2)
    print(x_train[:5])
    analysis_shape_x_train = convert_to_analysis_shape(x_train)
    analysis_shape_normed_x_train = data_norm(analysis_shape_x_train)
    print(analysis_shape_normed_x_train[:5])
    analysis_shape_x_test = convert_to_analysis_shape(x_test)
    analysis_shape_normed_x_test = data_norm(analysis_shape_x_test)
    print(analysis_shape_normed_x_test[:5])
    
    sk_learn_compatible_normed_x_train = convert_to_sklearn_shape(analysis_shape_normed_x_train)
    sk_learn_compatible_normed_x_test = convert_to_sklearn_shape(analysis_shape_normed_x_test)
    
    
    easy_classification(sk_learn_compatible_normed_x_train, y_train, sk_learn_compatible_normed_x_test, y_test)
    
if __name__ == '__main__':
    main()

    

    
        
        