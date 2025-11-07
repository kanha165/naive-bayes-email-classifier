#  Naive Bayes Email Classifier

A machine learning project that classifies emails as **Spam** or **Not Spam** using the **Naive Bayes Algorithm**.  
It analyzes the text of an email and predicts whether the message is legitimate or spam based on training data.

---

##  Project Overview

This project demonstrates how **Natural Language Processing (NLP)** and **Machine Learning** can be applied to build an intelligent email spam detection system.  
The model is trained on a dataset of labeled emails using the **Multinomial Naive Bayes classifier**.

**Key Features:**
- Classifies email text as **Spam** or **Not Spam**
- Uses **TF-IDF Vectorization** for text processing
- Built with **Scikit-learn**, **Pandas**, and **NumPy**
- Easy to train, test, and visualize results in Jupyter Notebook

---

##  Project Structure
NaiveBayes-Spam-Classifier/
│
├── train_model.ipynb # Notebook for model training
├── test_model.ipynb # Notebook for testing and visualization
├── email_data.csv # Email dataset used for training
├── naive_bayes_model.pkl # Saved trained model
├── vectorizer.pkl # Saved vectorizer
├── requirements.txt # List of dependencies
├── .gitignore # Ignored files
└── README.md # Project documentation




---

##  Installation and Usage

### 1. Clone the repository
```bash
git clone https://github.com/kanha165/naive-bayes-email-classifier.git
cd naive-bayes-email-classifier

2.Install dependencies
pip install -r requirements.txt


3️⃣ Run the notebooks
Open the Jupyter notebooks:

train_model.ipynb → for training the model

test_model.ipynb → for testing or visualizing predictions



📊 Model & Visualization

Algorithm: Multinomial Naive Bayes

Vectorizer: TF-IDF

Evaluation Metrics: Accuracy, Precision, Recall, Confusion Matrix

Visualization: Matplotlib & Seaborn (for confusion matrix and bar charts)

Example Output:

Input Text	Prediction
"Get free money now!"	Spam
"Meeting at 5 PM tomorrow"	Not Spam







🧰 Technologies Used
Tool / Library	Purpose
Python	Programming language
Scikit-learn	Naive Bayes algorithm
Pandas	Data handling
NumPy	Numerical computation
Matplotlib / Seaborn	Visualization
Jupyter Notebook	Model training & testing




🧑‍💻 Author

Kanha Patidar
B.Tech CSIT Student | Chamelidevi Group of Institutions, Indore
📍 India

GitHub: kanha165

LinkedIn: (https://www.linkedin.com/in/kanha-patidar-837421290/)
Email: (kanhapatidar7251@gmail.com)



📜 License
This project is licensed under the MIT License — you are free to use and modify it for learning and development purposes.

