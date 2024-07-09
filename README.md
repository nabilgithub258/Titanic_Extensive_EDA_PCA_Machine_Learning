

# Titanic Dataset Analysis and Modeling

Welcome to the Titanic Dataset Analysis and Modeling project! This repository contains a comprehensive analysis and modeling of the Titanic dataset, including extensive Exploratory Data Analysis (EDA), Principal Component Analysis (PCA), Linear Regression, Boosting techniques, and more.
This might have been the most comprehesive I have done yet, because Titanic story is so deeply rooted, hopefully you enjoy.

## Introduction

The sinking of the Titanic is one of the most infamous shipwrecks in history. In this project, we analyze the Titanic dataset to understand the factors that contributed to the survival of passengers. We employ various machine learning techniques to predict the survival of passengers based on different features.

## Dataset

The dataset used in this project is the Titanic dataset, which can be found on [Kaggle](https://www.kaggle.com/c/titanic/data). The dataset contains information about the passengers on the Titanic, including whether they survived or not, their age, gender, class, and other features.


## Exploratory Data Analysis (EDA)

In the EDA section, we perform a detailed analysis of the dataset to understand the distribution of features, identify missing values, and explore relationships between different variables. We visualize the data using various plots and charts to gain insights into the factors affecting passenger survival.

## Feature Engineering

Feature engineering involves creating new features from the existing ones to improve the performance of machine learning models. In this section, we engineer new features such as family or alone. We also handle missing values and perform feature scaling.

## Principal Component Analysis (PCA)

PCA is used to reduce the dimensionality of the dataset while retaining most of the variance. In this section, we apply PCA to the Titanic dataset to identify the most important features and reduce the complexity of our models.

## Modeling

### Linear Regression

We start with a simple Linear Regression model to predict the fare of passengers. The goal is to understand the relationship between the fare and other features in the dataset.
Although this dataset is not ideal for linear regression but I wanted to challenge myself.

### Boosting

Boosting techniques are employed to improve the accuracy of our predictions. We use Gradient Boosting, XGBoost, and other boosting algorithms to enhance the performance of our models.

### Other Models

In addition to Linear Regression and Boosting, we explore other machine learning models such as Decision Trees, Random Forests, and KNN. We compare the performance of these models using various evaluation metrics.

## Conclusion

Our model performed well in predicting the survival of passengers, starting with a Logistic Regression classifier. After further tuning, we achieved our best results using a K-Nearest Neighbors (KNN) model, with an accuracy of 0.81.

We then challenged ourselves by creating a linear model to predict the fare of passengers. Despite the dataset not being ideally suited for this task, we pursued this approach to test our skills. However, the resulting model did not perform well.

Our model's Root Mean Squared Error (RMSE) for fare prediction was 31 pounds, indicating a significant discrepancy. If anyone has suggestions on how to reduce the RMSE, we welcome your input.
