
import streamlit as st
import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification

MODEL_PATH = "model"

tokenizer = AlbertTokenizer.from_pretrained(MODEL_PATH)
model = AlbertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

id2label = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sadness",
    6: "surprise"
}

st.title("Emotion Detection (MELD + ALBERT)")

text = st.text_area("Enter text")

if st.button("Predict"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    pred = torch.argmax(probs).item()

    st.write("Prediction:", id2label[pred])
