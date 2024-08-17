!pip install joblib

import pandas as pd
import streamlit as st
import joblib

# Load the model and encoders outside the main function to avoid reloading
model = joblib.load('airline_fare_predictor.joblib')
label_encoders = joblib.load('airline_encoders.joblib')

def main():
    st.title(':red[Airline Fare Prediction]')
    st.image('images/plane.jpg')
    st.header('About This Model')
    st.write("This model is designed to predict flight fares based on a comprehensive dataset of flight bookings from six major metro cities in India.Dataset contains information about flight booking options from the website Easemytrip for flight travel between India's top 6 metro cities. There are 300261 datapoints and 11 features in the cleaned dataset. By analyzing key factors such as airline, departure and arrival cities, flight class, and the number of stops, the model provides accurate fare predictions, helping users plan their travels more effectively. Whether you're booking a flight from Delhi to Bangalore or Mumbai to Chennai, this model leverages historical data and advanced machine learning techniques to offer reliable fare estimates, ensuring that you get the best possible insights for your travel planning.")
    st.header('Fare Prediction')
    airline = st.selectbox('Select your Airline', ['Vistara', 'Air_India', 'Indigo', 'GO_FIRST', 'AirAsia', 'SpiceJet'], index=None)
    source_city = st.selectbox('Select departure city', ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'], index=None)
    departure_time = st.selectbox('Select departure time', ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'], index=None)
    stops = st.selectbox('No of stops', ['zero', 'one', 'two_or_more'], index=None)
    destination_city = st.selectbox('Select your destination city', ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'], index=None)
    arrival_time = st.selectbox('Select your arrival time', ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'], index=None)
    flight_class = st.selectbox('Select your flight class', ['Economy', 'Business'], index=None)
    duration = st.number_input('Select your duration in hours', min_value=0.0, step=1.0)
    days_left = st.slider('Days left for flight', 1, 60)

    if st.button("PREDICT"):
        user_input = {
            'airline': airline,
            'source_city': source_city,
            'departure_time': departure_time,
            'stops': stops,
            'arrival_time': arrival_time,
            'destination_city': destination_city,
            'flight_class': flight_class,
            'duration': duration,
            'days_left': days_left
        }

        user_input_df = pd.DataFrame([user_input])

        for column in label_encoders:
            le = label_encoders[column]
            user_input_df[column] = le.transform(user_input_df[column])

        prediction = model.predict(user_input_df)
        st.write(f"The predicted fare amount is {prediction[0]}")

if __name__ == "__main__":
    main()
