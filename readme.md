## **Background**
The dataset includes over a hundred variables describing the background of insurance applicants. The task is to use these customer data, to help the insurance company to classify applicants based on their similarity in background and accordingly assign different types of insurance quotes. 

Project Goal:
All customers are classified into 8 different classes numbered from 1 to 8. Each class is independent, so the numbers are categorical value. <br>
Challenges:
The size of the database is comparatively large and each entry has high dimensionality. There are many missing data in the database. There are mixed attribute types in the original data, including categorical, continuous and discrete. <br>

The project is based on a kaggle competition: `https://www.kaggle.com/c/prudential-life-insurance-assessment`


## About dataset

The diagram below illustrates the results of PCA performed on the imputed training dataset. It can be observed that as there are over one hundred attributes, there are no attributes significantly contribute to the overall variance of the data. 
 
<p align="center">
  <img src="https://github.com/SelinaWu/insurance_risks_prediction-/blob/master/PCA.png">
</p>


## Data pipeline
There are four Python script files used in this project.<br>
    `utils.py`: contains various class definitions and auxiliary functions<br>
    `data_preprocess.py`: can be executed with command `python3 DataPreprocess.py`. Perfrom data cleaning on both training and testing datasets.<br>
    `data_classification.py`: can be exeucted with command `python3 DataClassification.py`. Perform data classifications based on imputed datasets.<br>
    `predict_dataset.py`: can be exeucted with command `python3 PredictDataset.py`. This script takes 'testing_result.csv' (the complete, or imputed, testing set) as input and uses 'trained_trees.pkl' file (contains the trained model) so that the classification can be performed on this new complete testing set very quickly.<br>

## Result
Diagram below illustrates the relationships between number of trees and performance of the algorithm. 
* This is a multi class classification problem, and there are 8 classes in total.
<p align="center">
  <img src="https://github.com/SelinaWu/insurance_risks_prediction-/blob/master/RF_result.png">
</p>
Below are parameters we used in the random forest tree algorithm:<br>
n_folds = 5 max_depth = 10 min_size = 1 <br>
sample_size = 1.0 n_trees = 10 n_features= 40 <br>
test_ratio = 0.8 
 
## Libraries

External libraries directly used:<br>
    numpy==1.14.2<br>
    pandas==0.22.0<br>
    scipy==1.0.1<br>

Other external library dependencies:<br>
    cycler==0.10.0<br>
    kiwisolver==1.0.1<br>
    matplotlib==2.2.2<br>
    mkl-fft==1.0.0<br>
    mkl-random==1.0.1<br>
    pyparsing==2.2.0<br>
    python-dateutil==2.7.2<br>
    pytz==2018.4<br>
    six==1.11.0<br>
    tornado==5.0.2<br>
