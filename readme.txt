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
There are four Python script files used in this project.
    `utils.py`: contains various class definitions and auxiliary functions
    `DataPreprocess.py`: can be executed with command `python3 DataPreprocess.py`. Perfrom data cleaning on both training and testing datasets.
    `DataClassification.py`: can be exeucted with command `python3 DataClassification.py`. Perform data classifications based on imputed datasets.
    `PredictDataset.py`: can be exeucted with command `python3 PredictDataset.py`. This script takes 'testing_result.csv' (the complete, or imputed, testing set) as input and uses 'trained_trees.pkl' file (contains the trained model) so that the classification can be performed on this new complete testing set very quickly.


## Libraries

External libraries directly used:
    numpy==1.14.2
    pandas==0.22.0
    scipy==1.0.1

Other external library dependencies:
    cycler==0.10.0
    kiwisolver==1.0.1
    matplotlib==2.2.2
    mkl-fft==1.0.0
    mkl-random==1.0.1
    pyparsing==2.2.0
    python-dateutil==2.7.2
    pytz==2018.4
    six==1.11.0
    tornado==5.0.2
