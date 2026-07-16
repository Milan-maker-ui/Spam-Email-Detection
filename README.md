# 📧 Spam Email Detection using Machine Learning

A Machine Learning-based web application that classifies emails or SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP). The project uses TF-IDF vectorization and a Multinomial Naive Bayes classifier, with a simple Flask web interface for real-time predictions.

---

## 🚀 Features

- Detects Spam and Ham messages instantly
- Text preprocessing using NLP techniques
- TF-IDF feature extraction
- Trained Multinomial Naive Bayes model
- Flask-based web application
- Pre-trained model saved with Pickle
- Easy to retrain using your own dataset

---

## 🛠️ Technologies Used

- Python 3.x
- Flask
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Pickle
- HTML

---

## 📊 Dataset

The project uses the **SMS Spam Collection Dataset**.

Dataset contains:

- **label** → spam / ham
- **message** → SMS or email text

---

## ▶️ Run the Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 🧠 Machine Learning Workflow

1. Load Dataset
2. Clean Text
3. Tokenization & Preprocessing
4. TF-IDF Vectorization
5. Train Naive Bayes Classifier
6. Save Model
7. Predict New Messages

---

## 📈 Future Improvements

- Email attachment analysis
- Deep Learning (LSTM/BERT)
- Multi-language spam detection
- Responsive UI
- Docker support
- Cloud deployment (Render, Railway, Azure)
  
---

## ⭐ Show Your Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub.

---

