import streamlit as st
import numpy as np
import pickle

# Load the model
et = pickle.load(open('life2.pkl', 'rb'))
df = pickle.load(open('5_best.pkl','rb'))

st.sidebar.title('Life Expectancy Predictor')
image = "sidebar.png"
st.sidebar.image(image, use_column_width=True)

# User input fields
infant_deaths = st.sidebar.selectbox('Infant deaths',df['infant deaths'].unique())
under_five_deaths = st.sidebar.selectbox('Under-five deaths',df['under-five deaths '].unique())
HIV_AIDS = st.sidebar.selectbox('HIV/AIDS',df[' HIV/AIDS'].unique())
Polio = st.sidebar.selectbox('Polio',df['Polio_random'].unique())
BMI_random = st.sidebar.selectbox('BMI',df[' BMI _random'].unique())
Adult_Mortality_random = st.sidebar.selectbox('Adult Mortality',df['Adult Mortality_random'].unique())
Income_composition_of_resources_random = st.sidebar.selectbox('Income composition of resources',df['Income composition of resources_random'].unique())
Schooling_random = st.sidebar.selectbox('Schooling',df['Schooling_random'].unique())
Diphtheria_random = st.sidebar.selectbox('Diphtheria',df['Diphtheria _random'].unique())
thinness_59_years_random = st.sidebar.selectbox('Thinness 5-9 years',df[' thinness 5-9 years_random'].unique())
thinness_119_years_random = st.sidebar.selectbox('Thinness 1-19 years',df[' thinness  1-19 years_random'].unique())
Status_targuided = st.sidebar.selectbox('Status',df['Status_targuided'].unique())



# Load the logo image
logo_image = "logo.jpeg"
st.image(logo_image, use_column_width=True)

if st.sidebar.button('Predict Life Expectancy'):
    # Gather user input into an array
    query = np.array([infant_deaths, under_five_deaths, HIV_AIDS, Polio, BMI_random, Adult_Mortality_random,
                      Income_composition_of_resources_random, Schooling_random, Diphtheria_random,
                      thinness_59_years_random, thinness_119_years_random, Status_targuided])
    # Make prediction using the model
    prediction = et.predict([query])[0]

# About Our Model
st.markdown("---")
st.markdown("## About Our Model")
st.markdown("Our model is based on a machine learning algorithm(Extra-Tree Regressor) trained on a dataset containing various health and demographic indicators. It takes into account factors such as infant deaths, under-five deaths, HIV/AIDS prevalence, Polio, BMI, adult mortality, income composition of resources, schooling, and other indicators to predict life expectancy. The model has been trained and fine-tuned to provide accurate predictions based on the input provided by the user.")

if 'prediction' in locals():
    st.markdown("---")
    
    years = int(prediction * 100) 
    st.info(f"### Predicted Life Expectancy is {years} years.")
    
