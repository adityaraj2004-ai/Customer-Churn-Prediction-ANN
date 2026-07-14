import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# ==========================
# Load Model
# ==========================
model = tf.keras.models.load_model("model.h5")

# ==========================
# Load Encoders and Scaler
# ==========================
with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("scaler.pickle", "rb") as file:
    scaler = pickle.load(file)

# ==========================
# Streamlit App
# ==========================
st.title("Customer Churn Prediction")

# ==========================
# User Inputs
# ==========================

Geography = st.selectbox(
    "Select Geography",
    onehot_encoder_geo.categories_[0]
)

Gender = st.selectbox(
    "Select Gender",
    label_encoder_gender.classes_
)

CreditScore = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

Tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

Balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=0.0
)

NumOfProducts = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

HasCrCard = st.selectbox(
    "Has Credit Card?",
    ["Yes", "No"]
)

IsActiveMember = st.selectbox(
    "Is Active Member?",
    ["Yes", "No"]
)

EstimatedSalary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

# ==========================
# Prediction Button
# ==========================

if st.button("Predict"):

    # Create dictionary
    input_data = {
        "CreditScore": [CreditScore],
        "Gender": [label_encoder_gender.transform([Gender])[0]],
        "Age": [Age],
        "Tenure": [Tenure],
        "Balance": [Balance],
        "NumOfProducts": [NumOfProducts],
        "HasCrCard": [1 if HasCrCard == "Yes" else 0],
        "IsActiveMember": [1 if IsActiveMember == "Yes" else 0],
        "EstimatedSalary": [EstimatedSalary],
    }

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame(input_data)

    # One-Hot Encode Geography
    geo_encoded = onehot_encoder_geo.transform([[Geography]]).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    # Combine both DataFrames
    input_df = pd.concat(
        [input_df.reset_index(drop=True),
         geo_encoded_df.reset_index(drop=True)],
        axis=1
    )

    # Reorder columns exactly like training data
    input_df = input_df[scaler.feature_names_in_]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)

    probability = prediction[0][0]

    st.subheader("Prediction")

    st.write(f"### Churn Probability: {probability:.2%}")

    if probability > 0.5:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")