import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.header("📤 Prediction & Download")

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
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

predictions = model.predict(test)

output = pd.DataFrame({
    "id": test["id"],
    "target": predictions
})

st.dataframe(output.head())

st.download_button(
    "⬇️ Download Predictions",
    output.to_csv(index=False),
    "submission.csv",
    "text/csv"
)
