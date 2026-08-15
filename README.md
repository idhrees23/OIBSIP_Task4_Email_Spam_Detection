# Task 4 - Email Spam Detection with Machine Learning

## Project Overview

This project focuses on building a Natural Language Processing (NLP) based machine learning system to classify text messages as either spam or legitimate (ham).

The project uses text preprocessing, TF-IDF feature extraction, and multiple machine learning classification algorithms to detect spam messages.

## Objective

The main objective of this project is to develop a binary text classification model that can distinguish between:

- Spam messages
- Legitimate (ham) messages

The models are evaluated using accuracy, precision, recall, F1-score, and confusion matrix.

## Dataset

The project uses the SMS Spam Collection dataset.

The dataset contains two columns:

- `label` - Indicates whether the message is `ham` or `spam`
- `message` - Contains the text message

### Dataset Statistics

- Total messages: 5,572
- Ham messages: 4,825
- Spam messages: 747

The dataset contains more legitimate messages than spam messages, making it an imbalanced classification dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- WordCloud
- Jupyter Notebook

## Project Workflow

The project follows the following workflow:

1. Dataset Loading
2. Data Inspection
3. Class Distribution Analysis
4. Text Preprocessing
5. Target Label Encoding
6. Train-Test Split
7. TF-IDF Feature Extraction
8. Model Training
9. Model Prediction
10. Model Evaluation
11. Confusion Matrix Visualization
12. Model Comparison
13. Best Model Selection
14. Model Saving
15. WordCloud Visualization
16. Conclusion

## 1. Dataset Loading

The SMS Spam Collection dataset was loaded into a Pandas DataFrame.

The dataset was inspected to understand its shape, columns, data types, and missing values.

No missing values were found in the dataset.

## 2. Class Distribution

The dataset contains two classes:

- Ham: 4,825 messages
- Spam: 747 messages

The class distribution was visualized using a chart to understand the balance between legitimate and spam messages.

## 3. Text Preprocessing

Text preprocessing was performed to clean the message data before feature extraction.

The preprocessing steps include:

- Converting text to lowercase
- Removing punctuation
- Removing stopwords
- Stemming words

These steps reduce unnecessary variations in the text and make the data more suitable for machine learning.

## 4. Target Label Encoding

The target labels were originally represented as text values:

- `ham`
- `spam`

Label encoding was used to convert these categorical labels into numerical values required by machine learning algorithms.

## 5. Train-Test Split

The dataset was divided into training and testing sets.

The training set was used to train the machine learning models, while the testing set was used to evaluate their performance on unseen messages.

A stratified train-test split was used to maintain the class distribution in both datasets.

## 6. TF-IDF Feature Extraction

TF-IDF stands for Term Frequency-Inverse Document Frequency.

It converts text messages into numerical feature vectors that machine learning algorithms can process.

### Term Frequency (TF)

Term Frequency measures how frequently a word appears in a particular message.

### Inverse Document Frequency (IDF)

Inverse Document Frequency reduces the importance of words that occur frequently across many messages and gives more importance to words that are relatively uncommon.

### TF-IDF

TF-IDF combines TF and IDF to assign a numerical importance score to each word.

The TF-IDF vectorizer was fitted only on the training messages and then used to transform both training and testing messages. This prevents information from the test set from leaking into the training process.

## 7. Machine Learning Models

Three classification algorithms were trained:

### Multinomial Naive Bayes

Multinomial Naive Bayes is commonly used for text classification problems because it works well with word-frequency based features such as TF-IDF.

### Logistic Regression

Logistic Regression is a linear classification algorithm that estimates the probability of a message belonging to either the ham or spam class.

### Support Vector Machine

Support Vector Machine (SVM) finds a decision boundary that separates the classes. It is effective for high-dimensional text classification problems.

## 8. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Accuracy

Accuracy measures the overall percentage of correctly classified messages.

### Precision

Precision measures how many messages predicted as spam were actually spam.

### Recall

Recall measures how many actual spam messages were correctly identified by the model.

Recall is particularly important in spam detection because failing to identify an actual spam message means that the spam message may be classified as a legitimate message.

### F1-score

F1-score is the harmonic mean of precision and recall and provides a balanced measure of classification performance.

## 9. Confusion Matrix

A confusion matrix was used to visualize the correct and incorrect predictions made by each model.

It contains:

- True Negatives
- False Positives
- False Negatives
- True Positives

The confusion matrix helps understand the types of classification errors made by the models.

## 10. Model Comparison

The performance of the three models was compared using their evaluation metrics.

SVM achieved the best overall performance among the evaluated models.

The SVM model achieved approximately:

- Accuracy: 98.48%
- Spam Precision: 99%
- Spam Recall: 89%
- Spam F1-score: 94%

The exact values are available in the Jupyter Notebook outputs.

## 11. Best Model

Based on the evaluation results, Support Vector Machine (SVM) was selected as the best-performing model.

The model provided a strong balance between accuracy, precision, recall, and F1-score.

The SVM model was saved for future use.

The TF-IDF vectorizer was also saved because new messages must be transformed using the same feature extraction process before making predictions.

## 12. WordCloud Visualization

WordCloud visualizations were created to identify frequently occurring words in the dataset.

Separate WordCloud visualizations were used for:

- Spam messages
- Ham messages

These visualizations provide a simple way to understand commonly occurring words in each class.

# 13. Conclusion

The project successfully demonstrates the use of Natural Language Processing and machine learning for spam message detection.

Text preprocessing and TF-IDF feature extraction converted raw messages into useful numerical representations. Three machine learning algorithms were trained and evaluated.

Among the tested models, Support Vector Machine achieved the best overall performance with approximately 98.48% accuracy and 89% spam recall.

The project shows that machine learning can effectively classify text messages as spam or legitimate messages when combined with appropriate preprocessing and feature extraction techniques.

# Project Structure

Task4_Email_Spam_Detection/
│
├── dataset/
│   └── SMSSpamCollection
│
├── model/
│   ├── svm_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── MohammedIdhreesAjimulla_Task4.ipynb
├── README.md
└── requirements.txt

# Conclusion

This project developed a machine learning based email spam detection system using Natural Language Processing (NLP) techniques.

The SMS Spam Collection dataset was used to classify messages into two categories: ham (legitimate messages) and spam. The text data was cleaned using lowercase conversion, punctuation removal, stopword removal, and stemming.

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert the cleaned text messages into numerical features suitable for machine learning models.

Three classification algorithms were trained and evaluated:

- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

The models were evaluated using accuracy, precision, recall, F1-score, and confusion matrix.

Among the evaluated models, SVM achieved the best overall performance, with an accuracy of approximately 98.48% and a spam recall of approximately 89%.

Recall is particularly important in spam detection because it measures how many actual spam messages are correctly identified. A higher spam recall helps reduce the number of spam messages that are incorrectly classified as legitimate messages.

WordCloud visualizations were also created to identify frequently occurring words in spam and ham messages.

Overall, the project demonstrates how NLP and machine learning can be combined to build an effective email spam detection system.