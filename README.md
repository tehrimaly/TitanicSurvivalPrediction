# Titanic Survival Prediction using Machine Learning

## Project Overview
This project uses machine learning to predict whether a passenger survived the Titanic disaster based on features such as age, gender, ticket class, fare, and embarkation point.

The model is trained using historical passenger data and classifies survival as:
- 1 → Survived  
- 0 → Did not survive  
<img width="2688" height="1886" alt="image" src="https://github.com/user-attachments/assets/a4e0bc9d-e26c-4089-bbbf-e51ef15f5d01" />


##  Objective
To build a binary classification model that predicts passenger survival using supervised learning techniques.


##  Dataset
The dataset used is the famous Titanic dataset from Kaggle:
https://www.kaggle.com/c/titanic

Features used:
- Pclass (Ticket class)
- Sex (Gender)
- Age
- Fare
- Embarked

Target:
- Survived


## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib


##  Machine Learning Model
- Random Forest Classifier
- Supervised Learning (Classification)


##  Data Preprocessing
- Handled missing values in Age and Embarked
- Encoded categorical variables (Sex, Embarked)
- Selected relevant features
- Train-test split applied


## Workflow
1. Load dataset
2. Clean missing values
3. Encode categorical data
4. Split data into training and testing sets
5. Train Random Forest model
6. Evaluate performance
7. Predict survival outcomes


##  Model Evaluation
The model is evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report

Expected accuracy: ~80–85%


##  Results
The model successfully predicts passenger survival with good accuracy and demonstrates the importance of gender and passenger class in survival chances.

##  Future Improvements
- Hyperparameter tuning
- Feature engineering
- Using XGBoost or Gradient Boosting
- Deployment using Flask or Streamlit

