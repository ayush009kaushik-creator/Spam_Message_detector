import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("dataset.csv")

messages = data["message"]
labels = data["label"]

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Train model
model = MultinomialNB()
model.fit(X, labels)


def check_message():
    message = entry.get()

    if message.strip() == "":
        messagebox.showwarning("Warning", "Please enter a message")
        return

    prediction = model.predict(
        vectorizer.transform([message])
    )

    if prediction[0] == "spam":
        result.config(
            text="⚠️ SPAM MESSAGE",
            fg="red"
        )
    else:
        result.config(
            text="✅ NOT SPAM",
            fg="green"
        )


def clear_message():
    entry.delete(0, tk.END)
    result.config(text="")


# Window
window = tk.Tk()
window.title("AI Spam Message Detector")
window.geometry("550x350")
window.resizable(False, False)

title = tk.Label(
    window,
    text="🤖 AI Spam Message Detector",
    font=("Arial", 22, "bold")
)
title.pack(pady=30)

entry = tk.Entry(
    window,
    width=55,
    font=("Arial", 12)
)
entry.pack(pady=10)

check_button = tk.Button(
    window,
    text="Check Message",
    command=check_message,
    font=("Arial", 12, "bold")
)
check_button.pack(pady=10)

clear_button = tk.Button(
    window,
    text="Clear",
    command=clear_message,
    font=("Arial", 10)
)
clear_button.pack()

result = tk.Label(
    window,
    text="",
    font=("Arial", 20, "bold")
)
result.pack(pady=30)

window.mainloop()