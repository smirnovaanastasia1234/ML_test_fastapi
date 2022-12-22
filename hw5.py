import streamlit as st
from transformers import pipeline

st.header("Определение тональности текстов")
st.subheader("Введите текст для анализа")

text = st.text_area(" ",height=100)

classifier = pipeline("sentiment-analysis",   
                      "blanchefort/rubert-base-cased-sentiment")

result = st.button('Распознать текст')

def test_predict_positive():
    response = client.post("/predict/",
        json={"text": "I like machine learning!"}
    )
    json_data = response.json() 

    assert json_data['label'] == 'POSITIVE'


def test_predict_negative():
    response = client.post("/predict/",
        json={"text": "I hate machine learning!"}
    )
    json_data = response.json() 

    assert json_data['label'] == 'NEGATIVE' 
    
 def test_predict_neutral():
    response = client.post("/predict/",
        json={"text": "Sun"}
    )
    json_data = response.json() 

    assert json_data['label'] == 'NEUTRAL' 
st.write ("Тональность текста:")
st.write(classifier(text)[0]["label"])
