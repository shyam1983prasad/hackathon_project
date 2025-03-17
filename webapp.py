import streamlit as st
import pandas as pd
import joblib

st.title('Car Insurance Prediction')

#id	
#Gender	
#Age	
#Driving_License	
#Region_Code	
#Previously_Insured	
#Vehicle_Age	
#Vehicle_Damage	
#Annual_Premium	
#Policy_Sales_Channel	
#Vintage

#read the dataset to fill the values in the drop down list
df = pd.read_csv('train.csv')

#create input fields
id = st.number_input('Enter id', min_value=0, max_value=100000, value=0)
gender = st.selectbox('Select the Gender', pd.unique(df['Gender']))
age = st.number_input('Enter your Age', min_value=18, max_value=100, value=18)
driving_license = st.selectbox('Do you have Driving License (Yes:1, No:0)', pd.unique(df['Driving_License']))
region_code = st.number_input('Enter your Region Code', min_value=0, max_value=99999, value=0)
previously_insured = st.selectbox('Was your vehicle Previously Insured', pd.unique(df['Previously_Insured']))
vehicle_age = st.selectbox('Enter the Vehicle Age', pd.unique(df['Vehicle_Age']))
vehicle_damage = st.selectbox('Does your Vehicle has Damage', pd.unique(df['Vehicle_Damage']))
annual_premium = st.number_input('Whats your Annual Premium', min_value=1000, max_value=100000, value=1000)
policy_sales_channel = st.number_input('Whats your Policy Sales Channel', min_value=0, max_value=100000, value=0)
vintage = st.number_input('Whats your vehicle Vintage', min_value=0, max_value=365, value=0)

#convert the input data into a dictionary  
input_data = {
    'id': id,
    'Gender': gender,
    'Age': age,
    'Driving_License': driving_license,
    'Region_Code': region_code,
    'Previously_Insured': previously_insured,
    'Vehicle_Age': vehicle_age,
    'Vehicle_Damage': vehicle_damage,
    'Annual_Premium': annual_premium,
    'Policy_Sales_Channel': policy_sales_channel,
    'Vintage': vintage
 }

#click on the predict button
if st.button('Predict'):
    #load the model
    model = joblib.load('car_insurance_predict_model.pkl')

    #convert the input data into a dataframe
    input_df = pd.DataFrame([input_data])
    
    #predict the result
    prediction = model.predict(input_df)
    
    #display the result
    if prediction == 0:
        st.write('The person will not buy the insurance')
    else:
        st.write('The person will buy the insurance')