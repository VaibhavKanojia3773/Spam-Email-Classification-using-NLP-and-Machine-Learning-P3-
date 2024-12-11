# Spam Email Classification Using Naive Bayes

## Project Problem Statement
Spam emails have become a significant issue for individuals and organizations alike. Unsolicited emails not only clutter inboxes but also pose a security risk, as many contain phishing attempts or malware. An effective solution for automatically classifying emails as **spam** or **not spam** is crucial in maintaining a clean and secure email environment.

The goal of this project is to build a system that can automatically classify emails as spam or not spam using machine learning techniques, specifically **Naive Bayes**. This will help in reducing human intervention and increasing the efficiency of email filtering systems.

## Project Description
In this project, we employ the **Naive Bayes** classifier, a popular and simple machine learning algorithm, to classify emails based on their content. We use a labeled dataset containing various emails, which are preprocessed and fed into the model for training. The system classifies emails as either **spam** or **not spam** by learning patterns in the data, such as word frequency and other characteristics of email content.

Key features of the system include:
- **Data Preprocessing**: Handling missing values and converting text data into a suitable format for the model.
- **Model Training**: Training the Naive Bayes classifier using the features extracted from the dataset.
- **Performance Evaluation**: Assessing the model’s performance using accuracy, confusion matrix, and cross-validation.
- **Real-time Prediction**: Allowing users to input email content for real-time classification as spam or not.
- **Model Saving**: Saving the trained model for future use, so it doesn't need to be retrained each time.

## Solution Approach
1. **Data Collection & Preprocessing**:
   The first step in building the spam classification model is to load and preprocess the dataset. The dataset contains emails labeled as **spam** or **not spam**. We perform the following steps:
   - **Remove irrelevant columns**: For example, removing the 'Email No.' column which does not contribute to classification.
   - **Handle missing values**: If any missing values are found in the dataset, they are either cleaned or imputed.
   - **Text vectorization**: We convert the text content of the emails into numerical features using techniques like **bag of words** or **TF-IDF**.

2. **Model Selection**:
   We choose the **Naive Bayes** classifier, which is well-suited for text classification problems. This classifier works on the assumption that features are independent, making it a simple yet powerful model for tasks like spam detection.

3. **Model Training**:
   The dataset is split into **training** and **testing** sets. The model is trained on the training set using the features derived from the emails and their respective labels (spam or not spam).

4. **Model Evaluation**:
   The performance of the trained model is evaluated using:
   - **Accuracy**: The percentage of correctly classified emails.
   - **Confusion Matrix**: A matrix showing the number of true positives, true negatives, false positives, and false negatives.
   - **Cross-validation**: To ensure the model’s robustness, 10-fold cross-validation is used, providing a reliable measure of its performance.

5. **Real-time Classification**:
   After training, the model allows users to input the content of an email to classify it as spam or not spam. The email content is preprocessed in the same way as the training data, and the model predicts the category (spam or not).

6. **Saving the Model**:
   Once the model is trained and evaluated, it is saved using the **Joblib** library for future use. This way, the model doesn’t need to be retrained every time the system is used.

## Example Output

```bash
Accuracy: 95.45%
Cross Validation Accuracy: 94.78%
Mean Cross-Validation Accuracy: 95.12%
Model saved as 'spam_email_classifier_model.pkl'
