import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

st.header("🧹 Data Preprocessing")

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train.rename(columns={"Heart Disease": "target"}, inplace=True)

train["target"] = train["target"].map({
    "Absence": 0,
    "Presence": 1
})

target = "target"

X_train = train.drop(columns=[target])
y_train = train[target]

numeric_cols = X_train.select_dtypes(include=["int64","float64"]).columns

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
test[numeric_cols] = scaler.transform(test[numeric_cols])

st.success("Scaling applied correctly (fit on train, transform on test)")

st.dataframe(X_train.head())
