import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    names=["label", "message"]
)

# 2. Features and target
X = data["message"]
y = data["label"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# 5. Train model
model.fit(X_train, y_train)

# 6. Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")

# 7. Save model
with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")