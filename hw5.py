import streamlit as st
from transformers import pipeline

st.header("Определение тональности текстов")
st.subheader("Введите текст для анализа")

text = st.text_area(" ",height=100)

classifier = pipeline("sentiment-analysis",   
                      "blanchefort/rubert-base-cased-sentiment")

result = st.button('Распознать текст')

def classif(txt):
  return classifier(txt)[0]["label"]

st.write ("Тональность текста:")
st.write(classif(text))
