# Spam Detection using Machine Learning

This project implements a simple **Spam Detection System** using **Machine Learning** and **Natural Language Processing (NLP)**.  
It classifies text messages as **Spam** or **Not Spam** using a Naive Bayes classifier.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit

## Dataset
- SMS spam dataset containing message text and labels (spam / ham)
- Duplicate records removed and labels converted for clarity

## Model
- Text features extracted using CountVectorizer
- Classification performed using Multinomial Naive Bayes
- Model evaluated using a train-test split

## How to Run
```bash
pip install streamlit
streamlit run spamdetection.py
