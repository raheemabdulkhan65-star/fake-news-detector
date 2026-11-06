import streamlit as st
import joblib

# Model aur vectorizer load kar rahe hain
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("📰 Fake News Detection System")
st.write("Enter a news article below to check whether it is Real or Fake using Machine Learning.")

# User input text area
news_text = st.text_area("Enter News Article Text:", height=150)

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        transform_text = vectorizer.transform([news_text])
        prediction = model.predict(transform_text)[0]
        
        if prediction == 1:
            st.success("🟢 This News is **REAL**.")
        else:
            st.error("🔴 This News is **FAKE**.")
