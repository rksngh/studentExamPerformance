import streamlit as st
import pickle


# Load model
model = pickle.load(
    open("model.pkl","rb")
)


st.title("🎓 Student Exam Score Prediction")


hours = st.number_input(
    "Hours Studied",
    min_value=0,
    max_value=15
)


previous = st.number_input(
    "Previous Score",
    min_value=0,
    max_value=100
)


sleep = st.number_input(
    "Sleep Hours",
    min_value=0,
    max_value=12
)


if st.button("Predict Score"):

    input_data = [[
        hours,
        previous,
        sleep
    ]]


    result = model.predict(
        input_data
    )


    st.success(
        f"Predicted Exam Score: {round(result[0],2)}"
    )
