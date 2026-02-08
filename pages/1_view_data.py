import streamlit as st
import pandas as pd

st.header("📁 View Dataset Files")

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
submission = pd.read_csv("data/sample_submission.csv")

st.subheader("Train Data")
st.dataframe(train.head())

st.subheader("Test Data")
st.dataframe(test.head())

st.subheader("Sample Submission")
st.dataframe(submission.head())
