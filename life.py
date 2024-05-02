import streamlit as st
import numpy as np
import pickle

# Load the model
et = pickle.load(open('life2.pkl', 'rb'))
df = pickle.load(open('5_best.pkl','rb'))

st.title('Life Expectancy Predictor')

# User input fields
infant_deaths = st.selectbox('Infant deaths',df['infant deaths'].unique())
under_five_deaths = st.selectbox('Under-five deaths',df['under-five deaths '].unique())
HIV_AIDS = st.selectbox('HIV/AIDS',df[' HIV/AIDS'].unique())
Polio = st.selectbox('Polio',df['Polio_random'].unique())
BMI_random = st.selectbox('BMI',df[' BMI _random'].unique())
Adult_Mortality_random = st.selectbox('Adult Mortality',df['Adult Mortality_random'].unique())
Income_composition_of_resources_random = st.selectbox('Income composition of resources',df['Income composition of resources_random'].unique())
Schooling_random = st.selectbox('Schooling',df['Schooling_random'].unique())
Diphtheria_random = st.selectbox('Diphtheria',df['Diphtheria _random'].unique())
thinness_59_years_random = st.selectbox('Thinness 5-9 years',df[' thinness 5-9 years_random'].unique())
thinness_119_years_random = st.selectbox('Thinness 1-19 years',df[' thinness  1-19 years_random'].unique())
Status_targuided = st.selectbox('Status',df['Status_targuided'].unique())

if st.button('Predict Life Expectancy'):
    # Gather user input into an array
    query = np.array([infant_deaths, under_five_deaths, HIV_AIDS, Polio, BMI_random, Adult_Mortality_random,
                      Income_composition_of_resources_random, Schooling_random, Diphtheria_random,
                      thinness_59_years_random, thinness_119_years_random, Status_targuided])
    # Make prediction using the model
    prediction = et.predict([query])[0]
    st.title("The Predicted Life Expectancy: {:.2f}".format(prediction))
