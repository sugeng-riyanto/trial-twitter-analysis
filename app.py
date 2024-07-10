import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import joblib
import nltk
from nltk.corpus import stopwords
import string
import seaborn as sns

# Ensure stopwords are available without downloading
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Function to preprocess text data
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Function to plot word frequency
def plot_word_frequency(data, title):
    vectorizer = CountVectorizer(stop_words='english')
    word_counts = vectorizer.fit_transform(data)  # Transform text data into word count vectors
    sum_words = word_counts.sum(axis=0)  # Sum the word counts
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:20]  # Get top 20 most frequent words
    words, counts = zip(*words_freq)
    fig, ax = plt.subplots()
    ax.barh(words, counts)  # Plot horizontal bar chart of word frequencies
    ax.set_title(title)
    ax.set_xlabel('Frequency')
    st.pyplot(fig)

# Function to perform hyperparameter tuning using GridSearchCV
def tune_model(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter
        'solver': ['newton-cg', 'lbfgs', 'liblinear']  # Solvers for optimization
    }
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)  # Fit the model with different combinations of hyperparameters
    return grid_search.best_params_, grid_search.best_estimator_  # Return the best parameters and model

# Function to plot learning curve
def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, val_scores = learning_curve(estimator, X, y, cv=5, scoring='accuracy', n_jobs=-1,
                                                           train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')  # Plot training scores
    ax.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Validation score')  # Plot validation scores
    ax.set_title('Learning Curve')
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Score')
    ax.legend(loc='best')
    st.pyplot(fig)

# Function to plot ROC curve and calculate AUC
def plot_roc_curve(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)  # Calculate AUC

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Plot diagonal line
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    st.pyplot(fig)

# Function to plot precision-recall curve
def plot_precision_recall_curve(y_true, y_pred_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='b', lw=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    st.pyplot(fig)

def main():
    st.title("Twitter Sentiment Analysis")
    st.write("Upload your training and testing CSV files.")

    # File upload widgets
    train_file = st.file_uploader("Upload Train CSV", type="csv")
    test_file = st.file_uploader("Upload Test CSV", type="csv")

    if train_file is not None and test_file is not None:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        st.write("Training Data")
        st.write(train_data.head())
        st.write("Testing Data")
        st.write(test_data.head())

        # Plot the distribution of sentiments in training data
        st.write("Sentiment Distribution in Training Data")
        fig, ax = plt.subplots()
        sns.countplot(x='label', data=train_data, ax=ax)
        ax.set_title('Sentiment Distribution in Training Data')
        st.pyplot(fig)

        # Preprocess text
        train_data['tweet'] = train_data['tweet'].apply(preprocess_text)
        test_data['tweet'] = test_data['tweet'].apply(preprocess_text)

        # Plot the most common words in the training data
        st.write("Most Common Words in Training Data")
        plot_word_frequency(train_data['tweet'], 'Most Common Words in Training Data')

        # Plot the most common words in the testing data
        st.write("Most Common Words in Testing Data")
        plot_word_frequency(test_data['tweet'], 'Most Common Words in Testing Data')

        # Split the training data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(train_data['tweet'], train_data['label'], test_size=0.2, random_state=42)

        # Vectorize text data
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)
        X_test_vec = vectorizer.transform(test_data['tweet'])

        # Tune model
        best_params, best_model = tune_model(X_train_vec, y_train)
        st.write("Best Parameters found: ", best_params)

        # Train the best model
        best_model.fit(X_train_vec, y_train)

        # Predict on validation and test data
        y_val_pred = best_model.predict(X_val_vec)
        y_val_pred_prob = best_model.predict_proba(X_val_vec)[:, 1]
        y_test_pred = best_model.predict(X_test_vec)
        y_test_pred_prob = best_model.predict_proba(X_test_vec)[:, 1]

        st.write("Validation Accuracy: ", accuracy_score(y_val, y_val_pred))
        st.write("Validation Classification Report: ")
        st.text(classification_report(y_val, y_val_pred))

        # Confusion matrix
        st.write("Confusion Matrix")
        cm = confusion_matrix(y_val, y_val_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

        # Learning curve
        st.write("Learning Curve")
        plot_learning_curve(best_model, X_train_vec, y_train)

        # ROC curve and AUC
        st.write("ROC Curve and AUC")
        plot_roc_curve(y_val, y_val_pred_prob)

        # Precision-Recall curve
        st.write("Precision-Recall Curve")
        plot_precision_recall_curve(y_val, y_val_pred_prob)

        # Display test data with predictions
        test_data['predicted_label'] = y_test_pred
        st.write("Test Data with Predictions")
        st.write(test_data.head())

        # Save the model and vectorizer
        model_file = "sentiment_model.pkl"
        vectorizer_file = "vectorizer.pkl"
        joblib.dump(best_model, model_file)
        joblib.dump(vectorizer, vectorizer_file)

        st.write(f"Model saved as {model_file}")
        st.write(f"Vectorizer saved as {vectorizer_file}")

if __name__ == '__main__':
    main()
