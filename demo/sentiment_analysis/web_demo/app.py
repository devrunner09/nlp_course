import streamlit as st 
import requests
import json

st.title("Sentiment analysis")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload dataset")

if uploaded_file:
    print("Uploading dataset ...")

    upload_url = "http://127.0.0.1:8000/upload_file"

    data = uploaded_file.getvalue()
    files = {"file": data}
    
    response = requests.post(url=upload_url, files=files)
    print(response)


sentence = st.text_input("Enter the sentence")

if sentence:
    print("Current sentence: {}".format(sentence))
    predict_url = "http://127.0.0.1:8000/predict_sentiment?sentence={}".format(sentence)
    response = requests.post(url=predict_url)
    res_json = json.loads(response.text)
    sentiment_score = res_json['sentiment_result']
    print("Current score: {}".format(sentiment_score))
    if sentiment_score > 0:
        message = "Positive Sentence"
    else:
        message = "Negative Sentence"
    print(message)
    st.write(message)
    st.write(sentiment_score)



