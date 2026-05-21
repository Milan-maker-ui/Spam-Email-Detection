import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv(
    r"C:\Users\arote\OneDrive\Desktop\spam.csv",
    encoding='latin-1'
)

# Keep required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels
data['label'] = data['label'].map({
    'ham': 0,
    'spam': 1
})

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(data['message'])

y = data['label']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(x_train, y_train)

# Save model
joblib.dump(model, 'models/spam_model.pkl')

# Save vectorizer
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("Model and vectorizer saved successfully!")