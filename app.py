from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('models/spam_model.pkl')

vectorizer = joblib.load('models/vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():

    message = request.form['message']
    data = vectorizer.transform([message])
    prediction = model.predict(data)
    result = "spam" if prediction[0] == 1 else "Not Spam"

    return render_template( 'index.html', prediction = result)

if __name__ == "__mian__":
    app.run(debug = True)