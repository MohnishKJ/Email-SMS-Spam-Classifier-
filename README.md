# IsItSpamOrNot? 📩

ML-based app that classifies messages as spam or not spam in real-time.

## 🚀 What it does
- Takes any email/SMS text as input and predicts spam or not spam instantly
- Shows example spam messages for reference
- Gives quick tips to avoid spam

## ⚙️ How it works
- Cleans and preprocesses text using NLTK (tokenization, stopword removal, stemming)
- Converts text into numerical features using vectorization
- Classifies messages using a trained Scikit-learn model
- Model loaded via Pickle for fast inference
- Streamlit handles the UI

## 🛠️ Tech Stack
- Python
- Streamlit
- NLTK
- Scikit-learn
- Pickle

## 💡 Use case
Built to quickly check if a message is spam without relying on built-in email/SMS filters.

## 📩 Demo 
- Watch the demo video of the project here: https://drive.google.com/file/d/1L01aBq2vblGfCZVgRCZq_S4yqciuetbc/view?usp=sharing
