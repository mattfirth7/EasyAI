#EasyAI

EasyAI is a tool designed to lower the barrier to entry to 
employ Machine Learning models. All that is required is a CSV 
file and to tell the program whether the target variable is 
classification or regression.

Note: This tool is currently pre-alpha and while techincally the 
machine learning methods work, the more important facets of the 
process, notably separating training and testing data, and 
normalizing the dependent variables is not operational.


##Dependencies

Numpy
```bash
pip install numpy
```

SciKit Learn
```bash
pip install sklearn
```

MinePy
```bash
pip install minepy
```


##Usage
Select the csv file which contains your data, making sure that the 
label variable is the last column in the dataset. The tool will then 
examine several possible models and determine that which yields the 
strongest out of sample accuracy. The model will then be saved and 
can be imported into your tech stack for future use.

Note: We eventually plan to simplify the process even more by 
automatically detecting regression versus classification data, 
adding more rigorous pre-processing, feature selection, and feature 
extraction, adding interactive dashboards to compare performance 
between models, and add easy importing and implementation of saved 
models.