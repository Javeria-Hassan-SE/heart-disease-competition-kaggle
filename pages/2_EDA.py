import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.header("📊 Exploratory Data Analysis")

train = pd.read_csv("data/train.csv")
train.rename(columns={"Heart Disease": "target"}, inplace=True)

train["target"] = train["target"].map({
    "Absence": 0,
    "Presence": 1
})


numeric_cols = train.select_dtypes(include=["int64","float64"]).columns

for col in numeric_cols:
    st.subheader(col)

    fig, ax = plt.subplots(1,2, figsize=(10,4))

    ax[0].hist(train[col].dropna(), bins=20)
    ax[0].set_title("Histogram")

    ax[1].boxplot(train[col].dropna(), vert=False)
    ax[1].set_title("Boxplot")

    st.pyplot(fig)

st.subheader("Missing Values")
st.dataframe(train.isnull().sum())
