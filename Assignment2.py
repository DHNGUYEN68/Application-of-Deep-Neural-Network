
# coding: utf-8

# In[ ]:

import os
import sklearn
import pandas as pd
import numpy as np
import tensorflow.contrib.learn as skflow
from sklearn.cross_validation import KFold
from scipy.stats import zscore
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split



path = "./data/"
# These four functions will help you, they were covered in class.
# Encode a text field to dummy variables
def encode_text_dummy(df,name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name,x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
    
    
# Encode a text field to a single index value
def encode_text_index(df,name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_



# Encode a numeric field to Z-Scores
def encode_numeric_zscore(df,name,mean=None,sd=None):
    if mean is None:
        mean = df[name].mean()
    if sd is None:
        sd = df[name].std()
    df[name] = (df[name]-mean)/sd
    
    
    
# Encode a numeric field to fill missing values with the median.
def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)
# Convert a dataframe to x/y suitable for training.
def to_xy(df,target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    return df.as_matrix(result),df[target]





# Encode the toy dataset
def question1():
    print()
    print("***Question 1***")
    path = "./data/"
    
    filename_read = os.path.join(path,"toy1.csv")
    
    df = pd.read_csv(filename_read,na_values=['NA','?'])
    filename_write = os.path.join(path,"submit-hanmingli-prog2q1.csv")

    
    df['height'] = zscore(df['height'])
    df['width'] = zscore(df['width'])
    encode_numeric_zscore(df,'length')

    encode_text_dummy(df,'metal')
    encode_text_dummy(df,'shape')
    
    df.to_csv(filename_write,index=False)
    
    
    print("Wrote {} lines.".format(len(df)))
    
def question2():
    print()
    print("***Question 2***")
    
    
    path = "./data/"

    # Read dataset
    filename_read = os.path.join(path,"submit-hanmingli-prog2q1.csv")
    df = pd.read_csv(filename_read,na_values=['NA','?'])

    weight = encode_text_index(df,"weight")
    # Create x(predictors) and y (expected outcome)
    x,y = to_xy(df,'weight')


    num_classes = len(weight)


    # Split into train/test
    x_train, x_test, y_train, y_test = train_test_split(    
        x, y, test_size=0.25, random_state=45)

    # Create a deep neural network with 3 hidden layers of 10, 20, 10
    regressor = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=num_classes,
        steps=10000)

    # Early stopping
    early_stop = skflow.monitors.ValidationMonitor(x_test, y_test,
        early_stopping_rounds=10000, print_steps=100, n_classes=num_classes)

    # Fit/train neural network
    regressor.fit(x_train, y_train, monitor=early_stop)




    # Measure accuracy
    pred = regressor.predict(x_test)
    score = np.sqrt(metrics.mean_squared_error(pred,y_test))
    print("Final score (RMSE): {}".format(score))


    
    
    
def question3():
    print()
    print("***Question 3***")
    
    
    path = "./data/"
    
    filename_read = os.path.join(path,"toy1.csv")
    
    df = pd.read_csv(filename_read,na_values=['NA','?'])
    filename_write = os.path.join(path,"submit-hanmingli-prog2q3.csv")
    
    length_mean=df['length'].mean()
    width_mean=df['width'].mean()
    height_mean=df['height'].mean()
    
    length_std=df['length'].std()
    width_std=df['width'].std()
    height_std=df['height'].std()
    
    
    
    print("length: ({}, {})".format(length_mean,length_std))
    print("width:({}, {})".format(width_mean,width_std))
    print("height:({}, {})".format(height_mean,height_std))
    
    
    
    
# Z-Score encode these using the mean/sd from the dataset (you got ←  this in question 2)
    testDF = pd.DataFrame([
            {'length':1, 'width':2, 'height': 3},
            {'length':3, 'width':2, 'height': 5},
            {'length':4, 'width':1, 'height': 3}
         ])
    
    encode_numeric_zscore(testDF,'length',mean=length_mean,sd=length_std)
    encode_numeric_zscore(testDF,'width',mean=width_mean,sd=width_std)
    encode_numeric_zscore(testDF,'height',mean=height_mean,sd=height_std)
    
    print(testDF)
    
    
    
    
    
        
    testDF.to_csv(filename_write,index=False)    
def question4():
    print()
    print("***Question 4***")
    
    path = "./data/"

    filename_read = os.path.join(path,"iris.csv")
    filename_write = os.path.join(path,"submit-hanmingli-prog2q4.csv")
    df = pd.read_csv(filename_read,na_values=['NA','?'])
    name = ['species', 'sepal_l', 'sepal_w',  'petal_l','petal_w']
    df = pd.DataFrame(df[name])
    
    encode_numeric_zscore(df,'petal_l')
    encode_numeric_zscore(df,'sepal_w')
    encode_numeric_zscore(df,'sepal_l')
    encode_text_dummy(df,"species")
    np.random.seed(42)
    df = df.reindex(np.random.permutation(df.index))
    df.reset_index(inplace=True, drop=True)

    x, y = to_xy(df,'petal_w')

    # Cross validate
    kf = KFold(len(x), n_folds=5)

    oos_y = []
    oos_pred = []
    oos_x = []
    fold = 1
    for train, test in kf:        
        print("Fold #{}".format(fold))
        fold+=1

        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]

        # Create a deep neural network with 3 hidden layers of 10, 20, 10
        regressor = skflow.TensorFlowDNNRegressor(hidden_units=[10, 20, 10], steps=500)

        # Early stopping
        early_stop = skflow.monitors.ValidationMonitor(x_test, y_test,
            early_stopping_rounds=200, print_steps=50)

        # Fit/train neural network
        regressor.fit(x_train, y_train, monitor=early_stop)

        # Add the predictions to the oos prediction list
        pred = regressor.predict(x_test)

        oos_y.append(y_test)
        oos_pred.append(pred)  
        oos_x.append(x_test)

        # Measure accuracy
        score = np.sqrt(metrics.mean_squared_error(pred,y_test))
        print("Fold score (RMSE): {}".format(score))


    # Build the oos prediction list and calculate the error.
    oos_y = np.concatenate(oos_y)
    oos_pred = np.concatenate(oos_pred)
    oos_x = np.concatenate(oos_x)
   
    score = np.sqrt(metrics.mean_squared_error(oos_pred,oos_y))
    print("Final, out of sample score (RMSE): {}".format(score))    

    # Write the cross-validated prediction
    oos_y = pd.DataFrame(oos_y)
    oos_pred = pd.DataFrame(oos_pred)
    oos_x = pd.DataFrame(oos_x)
    oos_x.insert(3,'petal_w',oos_y[:])
    oosDF = pd.concat([oos_x,oos_y, oos_pred],axis=1 )
    oosDF.columns = ['sepal_l','sepal_w','petal_l','petal_w','species-Iris-setosa','species-Iris-versicolor','species-Iris-virginica',0,0]
    oosDF.to_csv(filename_write,index=False)
    

    
def question5():
    print()
    print("***Question 5***")
    filename_read = os.path.join(path,"auto-mpg.csv")
    filename_write = os.path.join(path,"submit-hanmingli-prog2q5.csv")
    df = pd.read_csv(filename_read,na_values=['NA','?'])

    # create feature vector
    missing_median(df, 'horsepower')
    encode_numeric_zscore(df, 'mpg')
    encode_numeric_zscore(df, 'horsepower')
    encode_numeric_zscore(df, 'weight')
    encode_numeric_zscore(df, 'displacement')
    encode_numeric_zscore(df, 'acceleration')
    encode_numeric_zscore(df, 'origin')
    
    
    tem=df['name']
    df.drop('name',1,inplace=True)
    
    
    # Shuffle
    np.random.seed(42)
    df = df.reindex(np.random.permutation(df.index))
    df.reset_index(inplace=True, drop=True)

    # Encode to a 2D matrix for training
    x,y = to_xy(df,'cylinders')

    # Cross validate
    kf = KFold(len(x), n_folds=5)
    
    oos_y = []
    oos_pred = []
    fold = 1
    for train, test in kf:        
        print("Fold #{}".format(fold))
        fold+=1
        
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        
       
        # Create a deep neural network with 3 hidden layers of 10, 20, 10
        classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=9,
            steps=500)

        # Early stopping
        early_stop = skflow.monitors.ValidationMonitor(x_test, y_test,
            early_stopping_rounds=200, print_steps=50, n_classes=9)
    
        # Fit/train neural network
        classifier.fit(x_train, y_train, monitor=early_stop)
        
        # Add the predictions to the oos prediction list
        pred = classifier.predict(x_test)
    
        oos_y.append(y_test)
        oos_pred.append(pred)        

        # Measure accuracy
        score = np.sqrt(metrics.mean_squared_error(pred,y_test))
        print("Fold score: {}".format(score))


    # Build the oos prediction list and calculate the error.
    oos_y = np.concatenate(oos_y)
    oos_pred = np.concatenate(oos_pred)
    score = np.sqrt(metrics.mean_squared_error(oos_pred,oos_y))
    print("Final, out of sample score: {}".format(score))    
    
    # Write the cross-validated prediction
    oos_y = pd.DataFrame(oos_y)
    oos_pred = pd.DataFrame(oos_pred)
    oos_y.columns = ['ideal']
    oos_pred.columns = ['predict']
    oosDF = pd.concat( [df, tem,oos_y, oos_pred],axis=1 )
    oosDF.to_csv(filename_write,index=False)



    
    
question1()
question2()
question3()
question4()
question5()        
        
        
        
        
        
        
        
        
        
        


# In[ ]:




# In[ ]:




# In[ ]:



