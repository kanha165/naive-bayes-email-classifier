# 📧 Naive Bayes Email Classifier

A Machine Learning project that classifies emails as **Spam** or **Not Spam** using the **Naive Bayes Algorithm**.

It analyzes the text content of emails and predicts whether a message is legitimate or spam based on trained data.

---

## 🚀 Project Overview

This project demonstrates how **Natural Language Processing (NLP)** and **Machine Learning** can be used to build an intelligent spam detection system.

The model is trained on labeled email data using the **Multinomial Naive Bayes classifier** along with **TF-IDF Vectorization**.

---

## ✨ Key Features

* 📩 Classifies emails into **Spam** or **Not Spam**
* 🧠 Uses **Multinomial Naive Bayes Algorithm**
* 🔤 Text processing with **TF-IDF Vectorization**
* 📊 Visualization using **Matplotlib & Seaborn**
* 📒 Easy experimentation via **Jupyter Notebook**
* 💾 Pre-trained model and vectorizer included

---

## 📁 Project Structure

```
NaiveBayes-Spam-Classifier/
│
├── train_model.ipynb        # Notebook for training the model
├── test_model.ipynb         # Notebook for testing and visualization
├── email_data.csv           # Dataset used for training
├── naive_bayes_model.pkl    # Saved trained model
├── vectorizer.pkl           # Saved TF-IDF vectorizer
├── requirements.txt         # Dependencies
├── .gitignore               # Ignored files
└── README.md                # Project documentation
```

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/kanha165/naive-bayes-email-classifier.git
cd naive-bayes-email-classifier
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the Project

Open Jupyter Notebook:

```bash
jupyter notebook
```

Then run:

* `train_model.ipynb` → Train the model
* `test_model.ipynb` → Test and visualize results

---

## 📊 Model & Evaluation

* **Algorithm:** Multinomial Naive Bayes
* **Vectorizer:** TF-IDF
* **Metrics Used:**

  * Accuracy
  * Precision
  * Recall
  * Confusion Matrix

---

## 📈 Visualization

* Confusion Matrix (Heatmap)
* Spam vs Not Spam distribution
* Prediction analysis charts

---

## 🧪 Example Output

| Input Text               | Prediction |
| ------------------------ | ---------- |
| Get free money now!      | Spam       |
| Meeting at 5 PM tomorrow | Not Spam   |

---

## 🧰 Technologies Used

| Tool / Library   | Purpose                |
| ---------------- | ---------------------- |
| Python           | Programming language   |
| Scikit-learn     | Machine Learning model |
| Pandas           | Data manipulation      |
| NumPy            | Numerical computations |
| Matplotlib       | Data visualization     |
| Seaborn          | Advanced visualization |
| Jupyter Notebook | Development & testing  |

---

## 👨‍💻 Author

**Kanha Patidar**
🎓 B.Tech CSIT (5th Semester)
🏫 Chameli Devi Group of Institutions, Indore

---

## 🔗 Connect with Me

* 💼 LinkedIn: https://www.linkedin.com/in/kanha-patidar-837421290/
* 🐙 GitHub: https://github.com/kanha165
* 📧 Email: [kanhapatidar7251@gmail.com](mailto:kanhapatidar7251@gmail.com)

---

## ⭐ Support

If you like this project, please ⭐ star the repository and share it!

---

## 📌 Future Improvements

* Add Deep Learning models (LSTM / BERT)
* Deploy as a web app (Streamlit / Flask)
* Improve dataset size and accuracy
* Real-time email filtering system

---

## 📜 License

This project is open-source and available under the **MIT License**.
