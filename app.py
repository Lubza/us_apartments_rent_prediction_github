from pathlib import Path

import pandas as pd
import streamlit as st


# ------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------

st.set_page_config(
    page_title="US Apartment Rent Prediction",
    page_icon="🏠",
    layout="wide"
)


# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------

DATA_PATH = Path(__file__).parent / "data" / "apartments_for_rent_10K.csv"


@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_PATH,
        sep=";",
        encoding="cp1252"
    )


df = load_data()


# ------------------------------------------------------
# HEADER
# ------------------------------------------------------

st.title("🏠 US Apartment Rent Prediction")

st.markdown(
    """
    Interactive data science project exploring apartment rental prices
    across the United States and developing machine learning models
    to predict monthly rent.
    """
)

st.divider()


# ------------------------------------------------------
# KEY METRICS
# ------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Listings",
        value=f"{len(df):,}"
    )

with col2:
    st.metric(
        label="Median Monthly Rent",
        value=f"${df['price'].median():,.0f}"
    )

with col3:
    st.metric(
        label="States",
        value=df["state"].nunique()
    )

with col4:
    st.metric(
        label="Cities",
        value=df["cityname"].nunique()
    )


# ------------------------------------------------------
# DATASET PREVIEW
# ------------------------------------------------------

st.divider()

st.subheader("Dataset Preview")

st.dataframe(
    df.head(100),
    use_container_width=True
)