# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# Step 1: Load and preprocess the data
data = pd.read_csv('/kaggle/input/email-spam-classification-dataset-csv/emails.csv')

# Drop the 'Email No.' column as it is not useful for prediction
data.drop(columns=['Email No.'], inplace=True)

# Check for missing values
if data.isna().sum().sum() > 0:
    print("Data contains missing values. Cleaning required.")
else:
    print("No missing values in the dataset.")

# Step 2: Split features (X) and target (y)
X = data.iloc[:, 0:3000]  # Features (all columns except 'Prediction')
y = data.iloc[:, 3000]    # Target variable ('Prediction')

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Step 4: Build and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 5: Predict on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 7: Perform Cross-Validation
cross_validation_score = cross_val_score(model, X_train, y_train, cv=10)
print(f"Cross Validation Accuracy: {cross_validation_score.mean() * 100:.2f}%")

# Step 8: Generate and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Spam', 'Spam'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: Spam Email Classification')
plt.tight_layout()
plt.show()

# Optional: Print confusion matrix values (TP, TN, FP, FN)
tn, fp, fn, tp = cm.ravel()
print(f"True Negative: {tn}, False Positive: {fp}, False Negative: {fn}, True Positive: {tp}")

# Step 9: Perform 10-fold cross-validation and plot the results
cross_validation_score = cross_val_score(model, X_train, y_train, cv=10)

# Plotting cross-validation results
plt.figure(figsize=(6, 4))
plt.bar(range(1, 11), cross_validation_score * 100, color='lightgreen')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy (%)')
plt.title('Cross-Validation Results: 10-Fold Accuracy')
plt.ylim(0, 100)  # Ensure the y-axis is from 0 to 100
plt.tight_layout()
plt.show()

# Print the mean accuracy from cross-validation
print(f"Mean Cross-Validation Accuracy: {cross_validation_score.mean() * 100:.2f}%")

# Step 10: Save the trained model using joblib
joblib.dump(model, 'spam_email_classifier_model.pkl')
print("Model saved as 'spam_email_classifier_model.pkl'")

# Step 11: Function to predict whether user input email is spam or not
def predict_spam(input_text):
    # Process the input text similarly to the dataset features
    input_data = [input_text.split()]
    
    # Convert input data into the same form as the dataset's features (count the frequency of words)
    input_vector = np.zeros((1, 3000))  # Initialize a zero vector of size (1, 3000)
    
    # Mark word occurrence (simple binary representation)
    for word in input_text.split():
        if word in data.columns[:-1]:  # Ignore the 'Prediction' column
            word_index = data.columns.get_loc(word)
            input_vector[0, word_index] = 1  # Mark word occurrence
    
    # Predict using the trained model
    prediction = model.predict(input_vector)
    return "Spam" if prediction == 1 else "Not Spam"

# Step 12: Take user input and classify it
user_input = input("Enter email content to classify as Spam or Not Spam: ")
result = predict_spam(user_input)
print(f"Prediction: {result}")
