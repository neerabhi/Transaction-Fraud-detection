import streamlit as st
import pandas as pd
import joblib

from preprocess import Fraud


# Load model
model = joblib.load('models/dummy/dummy_model.joblib')

# Create Fraud pipeline instance
# self.minmaxscaler = joblib.load('models\minmax_scaler.joblib')
# self.onehotencoder = joblib.load('models\one_hot_encoder.joblib')

pipeline = Fraud()

st.title("Transaction-Fraud Detection System")

# Collect inputs
step = st.number_input("Step (hour)", min_value=0)
tx_type = st.selectbox("Transaction Type", ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT', 'CASH_IN'])
amount = st.number_input("Transaction Amount")
name_orig = st.text_input("Name Orig (e.g., C12345)")
oldbalance_org = st.number_input("Old Balance Orig")
newbalance_orig = st.number_input("New Balance Orig")
name_dest = st.text_input("Name Dest (e.g., C67890)")
oldbalance_dest = st.number_input("Old Balance Dest")
newbalance_dest = st.number_input("New Balance Dest")
# is_fraud = st.selectbox("Is Fraud (for logging/testing)", [0, 1])
# is_flagged_fraud = st.selectbox("Is Flagged Fraud", [0, 1])

# When user clicks 'Predict'
if st.button("Predict Fraud"):
    # Prepare input
    input_dict = {
        'step': step,
        'type': tx_type,
        'amount': amount,
        'nameOrig': name_orig,
        'oldbalanceOrg': oldbalance_org,
        'newbalanceOrig': newbalance_orig,
        'nameDest': name_dest,
        'oldbalanceDest': oldbalance_dest,
        'newbalanceDest': newbalance_dest,
        # 'isFraud': is_fraud,
        # 'isFlaggedFraud': is_flagged_fraud
    }
    
    df_input = pd.DataFrame([input_dict])
    
    try:
        # Pass through pipeline
        df_clean = pipeline.data_cleaning(df_input.copy())
        df_feat = pipeline.feature_engineering(df_clean.copy())
        df_prepped = pipeline.data_preparation(df_feat.copy())

        # Get prediction
        result = pipeline.get_prediction(model, df_input, df_prepped)

        st.success("✅ Prediction completed!")
        st.json(result)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
