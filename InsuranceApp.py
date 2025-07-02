import streamlit as st
import pandas as pd
import joblib

models = {
    'Best Model': joblib.load('best_model.pkl')
} 

input_features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

st.title('Insurance Price App')

model_name = st.sidebar.selectbox('Select Model', list(models.keys()))

user_input = {}
for feature in input_features:
    user_input[feature] = st.sidebar.text_input(feature)

st.write("User Input:", user_input)

input_data = pd.DataFrame([user_input])
input_data['age'] = pd.to_numeric(input_data['age'], errors='coerce')
input_data['bmi'] = pd.to_numeric(input_data['bmi'], errors='coerce')
input_data['children'] = pd.to_numeric(input_data['children'], errors='coerce')

st.write("Converted Input Data:", input_data)


if st.sidebar.button('Predict'):
    model = models[model_name]
    prediction = model.predict(input_data)
    st.write(f'The prediction for GradeClass using {model_name} is: {prediction}')
