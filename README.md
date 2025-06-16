 💳 Credit Card Fraud Detection using Machine Learning

A machine learning-based system for detecting fraudulent credit card transactions. This project implements **logistic regression** to classify transactions as legitimate or fraudulent, and includes a **Streamlit application** for real-time predictions.



 📌 Project Overview

Credit card fraud poses serious financial risks to consumers and institutions. This project uses machine learning techniques to detect fraudulent transactions based on transaction features.

Key highlights:
- 🏷️ **Classification Model:** Logistic Regression
- ⚖️ **Data balancing:** Undersampling of legitimate transactions to address class imbalance
- 📊 **Evaluation:** Accuracy metric on both training and testing sets
- 🖥️ **Interactive Interface:** Streamlit app for predictions on uploaded datasets and individual transactions


 🗂️ Dataset

- 📁 **Source:** CSV file containing credit card transactions
- 🔢 **Size:** 284,807 rows × 31 columns
- 🎯 **Target variable:** `Class` (0 = legitimate, 1 = fraudulent)



 ⚙️ Preprocessing

- Separated legitimate and fraudulent transactions
- Performed undersampling of legitimate transactions to balance the dataset
- Split data into **training** and **testing** sets using `train_test_split()`

 🤖 Model

- **Algorithm:** Logistic Regression (`LogisticRegression()` from scikit-learn)
- **Training:** Model fitted on balanced training dataset
- **Prediction:** Classified transactions in the testing set


 📝 Evaluation

- Metric: **Accuracy**
- Calculated using `accuracy_score(93)` from scikit-learn
- Achieved high accuracy on both training and testing sets


 🚀 Streamlit Application

The project includes a Streamlit web app that:
- Allows users to **upload a CSV file** of transaction data
- Trains a logistic regression model on the uploaded data
- Lets users input transaction features for real-time fraud prediction

 💻 Tech Stack

- **Python 3**
- **scikit-learn**
- **pandas**
- **Streamlit**

 📌 Note

This repository serves as a portfolio showcase for my machine learning and app development skills.  
➡ *The dataset used is for academic purposes and the project does not contain sensitive code or proprietary data.*


⚡ How to Run

1️⃣ Clone this repo  
