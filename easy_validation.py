import numpy as np
from sklearn.model_selection import train_test_split
import pickle


def boostrapping_validation(x_test, y_test, sample_size = 0.33):
    mdl = pickle.load(open('temp.pickle', 'rb'))
    accuracy_array = []
    for i in range(5):
        x_test_sample, x_test_leftover, y_test_sample, y_test_leftover = train_test_split(x_test, y_test, test_size = sample_size)
        sample_accuracy = mdl.score(x_test_sample, y_test_sample)
        accuracy_array.append(sample_accuracy)
    
    return np.mean(accuracy_array)