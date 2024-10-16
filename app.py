import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to preprocess the text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    y = [i for i in text if i.isalnum()]

    # Remove stop words and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Stem the words
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Set up the app layout
st.set_page_config(page_title="Email/SMS Spam Classifier", page_icon="📩")
st.title("Email/SMS Spam Classifier 🕵️‍♂️")
st.markdown(
    """
    <style>
    .highlight { color: red; font-weight: bold; }
    .not-highlight { color: green; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True
)

# Input text area
input_sms = st.text_area("Enter the message", placeholder="Type your message here...")

if st.button('Predict'):
    # 1. Preprocess the input
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize the input
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict the result
    result = model.predict(vector_input)[0]

    # 4. Display the result
    if result == 1:
        st.markdown(f"<h2 class='highlight'>Spam 🚫</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 class='not-highlight'>Not Spam ✅</h2>", unsafe_allow_html=True)

# Tips to avoid spam messages
st.subheader("Tips to Avoid Spam Messages:")
st.markdown("""
- **Be cautious with links**: Don't click on links from unknown senders.
- **Use spam filters**: Ensure your email provider has strong spam filters enabled.
- **Don't share personal information**: Avoid giving out personal information to unsolicited messages.
- **Report spam messages**: Help improve spam detection by reporting suspicious messages.
""")

# Footer with additional information and rights reserved notice
st.markdown("---")
st.markdown("© 2024 Mohnish. All rights reserved.")
