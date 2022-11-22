## Product Categorization
### Multi-Class Text Classification of products based on their description
 
### General info

The goal of the project is product categorization based on their description with Machine Learning algorithms. Additionaly we did EDA analysis (data exploration, data aggregation and cleaning data) for better feature selection.

### Dataset
The dataset comes from http://makeup-api.herokuapp.com/ and has been obtained by an API.

The dataset contains the real descriptions about makeup products where each description has been labeled with a specific product.

### Motivation

The aim of the project is multi-class text classification to make-up products based on their description. Based on given text as an input, we have predicted what would be the category. We have five types of categories corresponding to different makeup products. In our analysis we using various Machine Learning/Deep Learning algorithms to get more accurate predictions and choose the most accurate one for our issue. 

### Project contains:
* Multi-class text classification with ML algorithms- ***Text_analysis.ipynb***
* Text classification with MLP and Convolutional Neural Netwok (CNN) models - ***Text_nn.ipynb***
* EDA analysis - ***Product_Analysis_EDA.ipynb***
* Python script to train ML models - **text_model.py**
* Python script to train ML models with smote method - **text_model_smote.py**
* Python scripts to text clean data - **clean_data.py**
* data, models - data and models used in the project.

### Summary

To resolve problem of the product categorization based on their description we applied multi-class text classification. We started with data analysis and data pre-processing from our dataset.  We have experimented with several Machine Learning algorithms: Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest, Gradient Boosting and Neural Networks: MLP and Convolutional Neural Network (CNN) using different combinations of text representations and embeddings.

From our experiments we can see that the tested models give a overall high accuracy and similar results for our problem. The SVM (BOW +TF-IDF) model give the best accuracy of validation set equal to 96 %. Logistic regression performed very well both with BOW + TF-IDF and Doc2vec and achieved similar accuracy as MLP. CNN with word embeddings also has a very comparable result (93 %) to MLP. 

Model | Embeddings | Accuracy
------------ | ------------- | ------------- 
SVM| BOW +TF-IDF  | 0.96
Random Forest| BOW +TF-IDF | 0.94
CNN | Word embedding | 0.93
Logistic Regression | BOW +TF-IDF  | 0.93
SVM | Doc2vec (DBOW)| 0.92
Naive Bayes | BOW +TF-IDF | 0.92
Gradient Boosting | BOW +TF-IDF | 0.91
Logistic Regression | Doc2vec (DM)  | 0.90


### Technologies
#### The project is created with:

* Python 3.9
* libraries: NLTK, gensim, Keras, TensorFlow, scikit-learn, pandas, numpy, seaborn.
#### Running the project:

use Jupyter Notebook or Google Colab.

You can run the scripts in the terminal:

    clean_data.py
    text_model.py
    text_model_smote.py

