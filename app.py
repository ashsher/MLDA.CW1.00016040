import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

# loading model, scaler and encoder
pipeline = joblib.load("final_model_pipeline.pkl")

# retrieving neighbourhoods and room types
df = pd.read_csv("sevilla.listings.csv")
neighbourhoods = sorted(
    df["neighbourhood"].dropna().unique()
)
room_types = sorted(
    df["room_type"].dropna().unique()
)

# setting app configurations
st.set_page_config(
    page_title="Airbnb Price Classifier", 
    layout="centered"
)

st.title("Airbnb Price Classification in Sevilla")
st.markdown("---")

st.image(
    "https://www.theboutiquevibe.com/wp-content/uploads/2024/01/iStock-1910223055-scaled.jpg"
)

st.write("""
This app predicts whether an Airbnb listing in Sevilla is **High Price** or **Low Price** based on its characteristics.  

Just fill in the details below to generate a prediction.
""")

# user input form
st.header("Listing Details")

# map to get latitude and longitude
st.subheader("Select the Property Location on the Map")

# Sevilla bounding box
SEVILLA_BOUNDS = {
    "min_lat": 37.33,
    "max_lat": 37.43,
    "min_lon": -6.06,
    "max_lon": -5.92
}

center_coords = [37.39, -5.99]   # Sevilla city centre

# create map
m = folium.Map(location=center_coords, zoom_start=13)

# add instruction marker
folium.Marker(
    center_coords,
    popup="Click anywhere on the map to select latitude/longitude",
    icon=folium.Icon(color="blue")
).add_to(m)

# enable click-to-get-coordinates
clicked_map = st_folium(
    m,
    height=400,
    width=700,
    key="sevilla_map"
)

# extract coordinates if user clicks
lat = None
lon = None

if clicked_map and clicked_map.get("last_clicked"):
    lat = clicked_map["last_clicked"]["lat"]
    lon = clicked_map["last_clicked"]["lng"]

    # restrict to Sevilla bounding box
    if not (SEVILLA_BOUNDS["min_lat"] <= lat <= SEVILLA_BOUNDS["max_lat"] and
            SEVILLA_BOUNDS["min_lon"] <= lon <= SEVILLA_BOUNDS["max_lon"]):
        st.error("Selected location is outside Sevilla. Please choose inside the city area.")
        lat, lon = None, None
    else:
        st.success(f"Location selected: Latitude {lat:.5f}, Longitude {lon:.5f}")

# columns
col1, col2 = st.columns(2)

with col1:
    neighbourhood = st.selectbox("What is the neibourhood?", neighbourhoods)
    room_type = st.selectbox("What kind of room are you staying?", room_types)
    minimum_nights = st.number_input("How many nights should you stay at minimum?", min_value=1, max_value=365, value=3)
    availability_365 = st.number_input("How many days per year should it be available?", min_value=0, max_value=365, value=120)
    latitude = st.number_input("Latitude", value=float(df["latitude"].mean()))
    longitude = st.number_input("Longitude", value=float(df["longitude"].mean()))

with col2:
    number_of_reviews = st.number_input("How many reviews in total does listing have?", min_value=0, value=30)
    reviews_per_month = st.number_input("How many reviews per month does listing recieve?", min_value=0.0, value=1.2, step=0.1)
    number_of_reviews_ltm = st.number_input("How many reviews did the listing have during the whole last year?", min_value=0, value=10)
    calculated_host_listings_count = st.number_input("How many listings does the host have in total?", min_value=1, value=1)

# converting inputs into DataFrame
input_data = pd.DataFrame({
    "room_type": [room_type],
    "neighbourhood": [neighbourhood],
    "minimum_nights": [minimum_nights],
    "availability_365": [availability_365],
    "number_of_reviews": [number_of_reviews],
    "reviews_per_month": [reviews_per_month],
    "number_of_reviews_ltm": [number_of_reviews_ltm],
    "calculated_host_listings_count": [calculated_host_listings_count],
    "latitude": [latitude],
    "longitude": [longitude]
})

# predict button
if st.button("Predict Price Category"):
    # getting prediction and probability from a pipeline
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0]

    price_label = "High Price" if prediction == 1 else "Low Price"

    st.subheader("Prediction Result:")
    st.markdown(f"### {price_label}")
    
    percent = round(max(probability) * 100, 1)

    if prediction == 1:
        st.warning(f"There is a **{percent}%** probability that this listing will be **High Price**.")
    else:
        st.success(f"There is a **{percent}%** probability that this listing will be **Low Price**.")

