# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime

startTime = datetime.now()
# import of libraries known to us

filename = "model.sv"
model = pickle.load(open(filename, 'rb'))
# we open a previously trained model

sex_d = {0: "Female", 1: "Male"}
chestPainATA_d = {0: "No", 1: "Yes"}
chestPainNAP_d = {0: "No", 1: "Yes"}
chestPainTA_d = {0: "No", 1: "Yes"}

fastingBS_d = {0: "0", 1: "1"}
restingECG_Normal_d = {0: "No", 1: "Yes"}
restingECG_ST_d = {0: "No", 1: "Yes"}
exerciseAngina_d = {0: "No", 1: "Yes"}
stSlopeUp_d = {0: "No", 1: "Yes"}
stSlopeFlat_d = {0: "No", 1: "Yes"}


def main():
    st.set_page_config(page_title="Heart Disease App")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://cdn.pixabay.com/photo/2016/01/27/15/19/heart-1164567_960_720.jpg")

    with overview:
        st.title("Heart Disease App")


    with left:
        sex_radio = st.radio("Gender", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        chestPainATA_radio = st.radio("Chest pain type: ATA (Select No if you have ASY)", list(chestPainATA_d.keys()),
                                      format_func = lambda x: chestPainATA_d[x])
        chestPainNAP_radio = st.radio("Chest pain type: NAP (Select No if you have ASY)", list(chestPainNAP_d.keys()),
                                      format_func = lambda x: chestPainNAP_d[x])
        chestPainTA_radio = st.radio("Chest pain type: TA (Select No if you have ASY )", list(chestPainTA_d.keys()),
                                     format_func = lambda x: chestPainTA_d[x])
        fastingBS_radio = st.radio("Fasting BS", list(fastingBS_d.keys()), format_func=lambda x: fastingBS_d[x])
        restingECG_Normal_radio = st.radio("Resting ECG: Normal (Select No if you have LVH)", list(restingECG_Normal_d.keys()),
                                           format_func = lambda x: restingECG_Normal_d[x])
        restingECG_ST_radio = st.radio("Resting ECG: ST (Select No if you have LVH)", list(restingECG_ST_d.keys()),
                                       format_func = lambda x: restingECG_ST_d[x])
        exerciseAngina_radio = st.radio("Exercise Angina", list(exerciseAngina_d.keys()),
                                        format_func = lambda x: exerciseAngina_d[x])
        stSlopeUp_radio = st.radio("ST Slope: Up (Select No if you have Down)", list(stSlopeUp_d.keys()), format_func=lambda x: stSlopeUp_d[x])
        stSlopeFlat_radio = st.radio("ST Slope: Flat (Select No if you have Down)", list(stSlopeFlat_d.keys()), format_func=lambda x: stSlopeFlat_d[x])
       
    with right:
        age_slider = st.slider("Age", value = 1, min_value = 0, max_value = 100)
        restingBP_slider = st.slider("Resting BP", min_value=0, max_value = 200)
        cholesterol_slider = st.slider("Cholesterol", min_value=0, max_value = 1000)
        maxHR_slider = st.slider("Max HR", min_value = 0, max_value = 200, step = 1)
        oldPeak_slider = st.slider("Old Peak", min_value = -5.0, max_value = 5.0, step=0.1)
    # slider

    data = [[age_slider, restingBP_slider, cholesterol_slider, fastingBS_radio, maxHR_slider, oldPeak_slider, sex_radio,
             chestPainATA_radio, chestPainNAP_radio, chestPainTA_radio, restingECG_Normal_radio, restingECG_ST_radio,
             exerciseAngina_radio, stSlopeUp_radio, stSlopeFlat_radio]]
    heartDisease = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Would I get heart disease?")
        st.subheader(("Yes" if heartDisease[0] == 1 else "No"))
        st.write("Confidence {0:.2f} %".format(s_confidence[0][heartDisease][0] * 100))


if __name__ == "__main__":
    main()