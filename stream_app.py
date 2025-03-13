import streamlit as st
import pandas as pd
import pickle

# Load trained model
@st.cache_resource
def load_model():
    with open(r"C:\Users\ether\OneDrive\Documents\assignment_imago\xgb_regressor.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Preprocessing function
def preprocess_data(df):
    if "hsi_id" in df.columns:
        df = df.drop(columns=["hsi_id"])
    
    # Add any other preprocessing steps if needed
    
    return df

# Streamlit UI
st.title("Mycotoxin Prediction App")
st.write("Upload your dataset, and the model will predict **vomitoxin_ppb**.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Sample")
    st.dataframe(df.head())

    # Preprocess data
    processed_df = preprocess_data(df)

    # Load model and make predictions
    model = load_model()
    predictions = model.predict(processed_df)

    # Show results
    df["Predicted_vomitoxin_ppb"] = predictions
    st.write("### Predictions")
    st.dataframe(df)

    # Download results
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )
