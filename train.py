import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("dataset.csv")

X_text = data["message"]
y = data["label"]

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("================================")
print("   AI SPAM DETECTOR")
print("================================")

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["spam", "not spam"]
)

print("\nConfusion Matrix:")
print(cm)

# Test custom message
message = input("\nEnter a message: ")

prediction = model.predict(
    vectorizer.transform([message])
)

print("\nPrediction:", prediction[0])