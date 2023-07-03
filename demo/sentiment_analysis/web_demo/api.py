from fastapi import FastAPI 
from fastapi import File, UploadFile
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import io 

# utils functions
from utils import count_sentence, train_naive_bayes, naive_bayes_predict

app = FastAPI()

@app.get("/")
async def index():
    return {"message" : "Welcome to HomePage"}

@app.post("/upload_file")
async def get_file(file: UploadFile = File(...)):
    data = file.file.read()
    df = pd.read_csv(io.BytesIO(data), index_col="Unnamed: 0")

    # TODO: update naive bayes parameters
    # Step 1: count sentences
    x_train, y_train = df['sentence'], df['sentiment']
    result = {}
    freqs = count_sentence(result, x_train.tolist(), y_train.tolist())
    global logprior, loglikelihood
    logprior, loglikelihood = train_naive_bayes(freqs, x_train, y_train)
    message = "Training Naive Bayes Successfully!"
    return {"result" : message}

@app.post("/predict_sentiment")
async def predict_sentiment(sentence: str):
    global logprior, loglikelihood
    if logprior is not None and loglikelihood is not None:
        sentiment = naive_bayes_predict(sentence, logprior, loglikelihood)
        return {"sentiment_result" : sentiment, "code" : 200}
    else:
        return {"sentiment_result" : "Please provide dataset", "code" : 400}