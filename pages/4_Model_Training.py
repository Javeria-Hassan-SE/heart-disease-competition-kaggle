import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.header("🤖 Model Training")

train = pd.read_csv("data/train.csv")
train.rename(columns={"Heart Disease": "target"}, inplace=True)

train["target"] = train["target"].map({
    "Absence": 0,
    "Presence": 1
})

target = "target"

X = train.drop(columns=[target])
y = train[target]

model = LogisticRegression()
model.fit(X, y)

st.success("Model trained successfully")

st.write("Model coefficients:")
st.write(model.coef_)
