import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
API_URL = "http://housing-api-alb-945997111.eu-west-2.elb.amazonaws.com/predict"  # your FastAPI endpoint
HOLDOUT_PATH = "data/processed/feature_engineered_holdout.csv"

# ----------------------------
# Load holdout dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(HOLDOUT_PATH, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df

data = load_data()

# ----------------------------
# UI
# ----------------------------
st.title("🏠 Housing Price Prediction Dashboard")

# Filters
years = sorted(data["year"].unique())
months = list(range(1, 13))
regions = ["All"] + sorted(data["region"].unique())

year = st.selectbox("Select Year", years)
month = st.selectbox("Select Month", months)
region = st.selectbox("Select Region", regions)

if st.button("Run Predictions 🚀"):
    # ----------------------------
    # Filter data
    # ----------------------------
    subset = data[(data["year"] == year) & (data["month"] == month)]
    if region != "All":
        subset = subset[subset["region"] == region]

    if subset.empty:
        st.warning("No data found for this selection.")
    else:
        st.write(f"📅 Running predictions for {year}-{month}, Region: {region}")

        # ----------------------------
        # Call API
        # ----------------------------
        try:
            response = requests.post(API_URL, json=subset.to_dict(orient="records"))
            response.raise_for_status()
            preds = response.json()

            # Add predictions to dataframe
            subset["prediction"] = preds["predictions"]

            # ----------------------------
            # Results
            # ----------------------------
            st.subheader("Predictions vs Actuals")
            st.dataframe(subset[["date", "region", "actual_price", "prediction"]])

            # Error metrics
            mae = (subset["prediction"] - subset["actual_price"]).abs().mean()
            rmse = ((subset["prediction"] - subset["actual_price"]) ** 2).mean() ** 0.5
            st.metric("Mean Absolute Error (MAE)", f"{mae:,.2f}")
            st.metric("Root Mean Squared Error (RMSE)", f"{rmse:,.2f}")

            # Plot
            fig, ax = plt.subplots()
            ax.plot(subset["date"], subset["actual_price"], label="Actual", marker="o")
            ax.plot(subset["date"], subset["prediction"], label="Prediction", marker="x")
            ax.set_title("Predicted vs Actual Prices")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"API call failed: {e}")
