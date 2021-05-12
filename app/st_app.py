from fsdl.covidtweets import predict
import streamlit as st
import pandas as pd


add_select_box = st.sidebar.selectbox("Demo Options", ("Noisy Tweets in Test Set", "Tweet Prediction"))
st.title("Predicting informativeness in tweet")
st.header('Dataset : WNUT-2020 Task 2 - Identification of informative COVID-19 English Tweets.')

if add_select_box == "Noisy Tweets in Test Set":
    st.subheader("Incorrectly Labeled Tweets from Test Set provided")
    df = pd.read_csv('./data/covid/noisy/noisy_text.csv', names=["Id", "Text", "Label", "HashTag", "predicted", "confidence"], header=None)
    df.style.set_properties(subset=['Text'], **{'width': '300px'})

    st.table(df)

elif add_select_box == "Tweet Prediction":
    st.subheader('''This app automatically identifes whether an English Tweet related to the novel coronavirus (COVID-19) is informative or not. Informatives is determined by text mentioning recovered, suspected, confirmed and death cases as well as location or travel history of the cases.''')
    text1 = st.text_area('Enter tweet text', value='Eg. Public health officials said there were two cases of #coronavirus in Santa Monica Thurs after confirming that three individuals in the city had tested positive Officials announced 40 new cases across Los Angeles County for a total of 231 HTTPURL')
    if st.button('Predict'):
        output = predict(text1)
        st.success(f"The tweet is {output}")