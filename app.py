import streamlit as st
import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.stem import PorterStemmer
import re
nltk.download('stopwords')
stopwords=set(nltk.corpus.stopwords.words("english"))


# load models
model = pickle.load(open("RandomForestClassifier.pkl","rb"))
lb = pickle.load(open("LabelEncoder.pkl","rb"))
tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl","rb"))

# ===============Custome fucntions-====

# defining a function for predicting emotions
def prediction(input_text):
  # Cleaning the input text
  cleaned_text = clean_text(input_text)

  # Vectorizing the input usign TFIDF
  input_vectorized = tfidf_vectorizer.transform([cleaned_text])

  # making prediction using Random Forest Classifier
  predicted_Code = model.predict(input_vectorized)[0]  # it will give number code of the emotion

  # getting named emotion corresponding to number code of emotion
  predicted_emotion = lb.inverse_transform([predicted_Code])[0]
  label=np.max(model.predict(input_vectorized)[0])

  return predicted_emotion,label


def clean_text(text):
  stemmer = PorterStemmer()
  # to remove all the numbers special characters except the small and capital case words.
  text = re.sub("[^a-zA-Z]"," ",text)
  # converting upper case to Lower case
  text = text.lower()
  # splitting the sentences into words
  text = text.split()
  # stemming anf removing the stop words.
  text = [stemmer.stem(word) for word in text if word not in stopwords]
  # again joining the single words into sentences
  return " ".join(text)

# app---

st.title("Human Emotions Recognition")
#st.write(["Joy","sad","fear","Anger","Love","Sadness","Surprise"])
input_text = st.text_input("Enter your emotion here")

if st.button("Predict"):
    predicted_emotion,label = prediction(input_text)
    st.write("Predicted Emotion : ",predicted_emotion)
