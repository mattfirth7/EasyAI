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
from sklearn import linear_model


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

def __fit_clf_model(name, clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    model = pickle.dumps(clf)
    return [name, accuracy, model]



#still need to add optimization over hyperparameters
def k_nearest(x_train, y_train, x_test, y_test):
    clf = KNeighborsClassifier(n_neighbors = 3)
    return __fit_clf_model('k_nearest', clf, x_train, y_train, x_test, y_test)

# poly kernel is not working so it has been removed
def support_vector_machine(x_train, y_train, x_test, y_test):
    model_arr = []
    accuracy_arr = []
    for kernel in ('linear', 'rbf'):
        clf = svm.SVC(kernel = kernel, gamma = 2)
        model_info = __fit_clf_model(kernel, clf, x_train, y_train, x_test, y_test)
        model_arr.append(model_info)
        accuracy_arr.append(model_info[1])
        
    accuracy_argmax = np.argmax(accuracy_arr)
    return model_arr[accuracy_argmax]

#still need to add optimization over hyperparameters
def gaussian_process_classifier(x_train, y_train, x_test, y_test):
    kernel = 1.0 * RBF(1.0)
    clf = GaussianProcessClassifier(kernel = kernel, random_state = 0)
    return __fit_clf_model('gaussian_process', clf, x_train, y_train, x_test, y_test)

#still need to add optimization over hyperparamters
def decision_tree_classifier(x_train, y_train, x_test, y_test):
    clf = DecisionTreeClassifier(random_state = 0)
    return __fit_clf_model('decision_tree', clf, x_train, y_train, x_test, y_test)

#still need to add optimization over hyperparameters
def random_forest_classifier(x_train, y_train, x_test, y_test):
    clf = RandomForestClassifier(max_depth = 2, random_state = 0)
    return __fit_clf_model('random_forest', clf, x_train, y_train, x_test, y_test)

#still need to add optimization over hyperparameters
def adaboost_classifier(x_train, y_train, x_test, y_test):
    clf = AdaBoostClassifier(n_estimators = 100, random_state = 0)
    return __fit_clf_model('adaboost', clf, x_train, y_train, x_test, y_test)

def native_bayes(x_train, y_train, x_test, y_test):
    clf = GaussianNB()
    return __fit_clf_model('native_bayes', clf, x_train, y_train, x_test, y_test)

def quadratic_discriminant(x_train, y_train, x_test, y_test):
    clf = QuadraticDiscriminantAnalysis()
    return __fit_clf_model('quadratic_discriminant', clf, x_train, y_train, x_test, y_test)

#Do I want to add something like rerunning classifiers with the highest accuracies
#to get a better estimate of mean accuracy?
def easy_classification(x_train, y_train, x_test, y_test):
    functions = [k_nearest, support_vector_machine, decision_tree_classifier, 
                 random_forest_classifier, adaboost_classifier, native_bayes]
    results_array = []
    for f in functions:
        results = f(x_train, y_train, x_test, y_test)
        results_array.append(results)
        print(results[1])
    #quadratic discriminant analysis is not currently working
    #results_array.append(quadratic_discriminant(x_train, y_train, x_test, y_test))
    
    accuracies_arr = [row[1] for row in results_array]
        
    accuracy_argmax = np.argmax(accuracies_arr)
    classifier = results_array[accuracy_argmax][0]
    model = results_array[accuracy_argmax][2]

    OUTPUT = 'easyai_classifier.pickle'
    
    with open(OUTPUT, 'wb') as f:
        f.write(model)
        f.close()
    
    print(classifier + " model chosen. Yielded max accuracy: " + str(results_array[accuracy_argmax][1]))
    print("Model saved as {}".format(OUTPUT))
    return


# need to fix this to use adjusted r2
def __fit_rgr_model(name, rgr, x_train, y_train, x_test, y_test):
    rgr.fit(x_train, y_train)
    r2 = rgr.score(x_test, y_test)
    model = pickle.dumps(rgr)
    return [name, r2, model]

def ordinary_least_squares(x_train, y_train, x_test, y_test):
    rgr = linear_model.LinearRegression()
    return __fit_rgr_model('ols', rgr, x_train, y_train, x_test, y_test)

def ridge_regr(x_train, y_train, x_test, y_test):
    rgr = linear_model.Ridge()
    return __fit_rgr_model('ridge', rgr, x_train, y_train, x_test, y_test)

def lasso_regr(x_train, y_train, x_test, y_test):
    rgr = linear_model.Lasso()
    return __fit_rgr_model('lasso', rgr, x_train, y_train, x_test, y_test)

def least_angle_regr(x_train, y_train, x_test, y_test):
    rgr = linear_model.Lars()
    return __fit_rgr_model('lars', rgr, x_train, y_train, x_test, y_test)

def bayes_ridge_regr(x_train, y_train, x_test, y_test):
    rgr = linear_model.BayesianRidge()
    return __fit_rgr_model('bayesian regr', rgr, x_train, y_train, x_test, y_test)

def support_vector_regr(x_train, y_train, x_test, y_test):
    rgr = svm.SVR()
    return __fit_rgr_model('svr', rgr, x_train, y_train, x_test, y_test)

def huber_regr(x_train, y_train, x_test, y_test):
    rgr = linear_model.HuberRegressor()
    return __fit_rgr_model('huber', rgr, x_train, y_train, x_test, y_test)

def easy_regression(x_train, y_train, x_test, y_test):
    functions = [ordinary_least_squares, ridge_regr, lasso_regr, 
                 least_angle_regr, bayes_ridge_regr, support_vector_regr,
                 huber_regr]
    results_array = []
    for f in functions:
        results = f(x_train, y_train, x_test, y_test)
        results_array.append(results)
        print(results[1])
    
    r2_arr = [row[1] for row in results_array]
        
    r2_argmax = np.argmax(r2_arr)
    regr_model = results_array[r2_argmax][0]
    model = results_array[r2_argmax][2]

    OUTPUT = 'easyai_regression.pickle'
    
    with open(OUTPUT, 'wb') as f:
        f.write(model)
        f.close()
    
    print(regr_model + " model chosen. Yielded max accuracy: " + str(results_array[r2_argmax][1]))
    print("Model saved as {}".format(OUTPUT))
    return




def main():
    print("Make sure the label variable is the last column in the csv")
    tkinter.Tk().withdraw()
    input_filename = askopenfilename()
    input_array = np.genfromtxt(input_filename, delimiter = ',', skip_header = 1)
    x = []
    y = []
    for row in input_array:
        y.append(row[-1])
        x.append(row[:-1])
        
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    analysis_shape_x_train = convert_to_analysis_shape(x_train)
    analysis_shape_normed_x_train = data_norm(analysis_shape_x_train)
    analysis_shape_x_test = convert_to_analysis_shape(x_test)
    analysis_shape_normed_x_test = data_norm(analysis_shape_x_test)
    
    
    
    sk_learn_compatible_normed_x_train = convert_to_sklearn_shape(analysis_shape_normed_x_train)
    sk_learn_compatible_normed_x_test = convert_to_sklearn_shape(analysis_shape_normed_x_test)
    
    print("Is this data regression or classification? ")
    label_type = input()
    
    if label_type.lower() == "classification":
        easy_classification(sk_learn_compatible_normed_x_train, y_train, sk_learn_compatible_normed_x_test, y_test)
    elif label_type.lower() == "regression":
        easy_regression(sk_learn_compatible_normed_x_train, y_train, sk_learn_compatible_normed_x_test, y_test)
    else:
        print("Invalid entry. Restart.")
        
if __name__ == '__main__':
    main()       
        