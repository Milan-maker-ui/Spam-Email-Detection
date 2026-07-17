from flask import Flask, render_template, request
import joblib
import sqlite3

app = Flask(__name__)

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


def init_db():
    conn = sqlite3.connect("spam.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT,
            prediction TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


init_db()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    message = request.form["message"]

    data = vectorizer.transform([message])

    prediction = model.predict(data)

    result = "Spam" if prediction[0] == 1 else "Not Spam"

    conn = sqlite3.connect("spam.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO predictions(message,prediction) VALUES(?,?)",
        (message, result),
    )

    conn.commit()
    conn.close()

    return render_template(
        "index.html",
        prediction=result,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)
