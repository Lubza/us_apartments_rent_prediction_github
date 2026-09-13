from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------

DATA_PATH = Path(__file__).parent / "data" / "apartments_for_rent_10K.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        encoding="cp1252"
    )

    # Ensure important variables are numeric
    numeric_columns = [
        "price",
        "square_feet",
        "bedrooms",
        "bathrooms",
        "latitude",
        "longitude",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


df = load_data()


# ------------------------------------------------------
# HEADER
# ------------------------------------------------------

st.title("🏠 US Apartment Rent Prediction")

st.markdown(
    """
    An interactive data science project exploring the US apartment rental
    market and developing machine learning models to predict monthly rent.

    The project covers exploratory data analysis, preprocessing,
    feature engineering, model comparison, cross-validation,
    hyperparameter tuning and model deployment.
    """
)

st.divider()


# ------------------------------------------------------
# KPI METRICS
# ------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Listings",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Median Rent",
        f"${df['price'].median():,.0f}"
    )

with col3:
    st.metric(
        "Median Size",
        f"{df['square_feet'].median():,.0f} ft²"
    )

with col4:
    st.metric(
        "States",
        f"{df['state'].nunique():,}"
    )

with col5:
    st.metric(
        "Cities",
        f"{df['cityname'].nunique():,}"
    )


# ------------------------------------------------------
# RENT DISTRIBUTION
# ------------------------------------------------------

st.divider()

st.header("Rental Market Overview")

chart_col, stats_col = st.columns([3, 1])

with stats_col:

    st.subheader("Rent Statistics")

    st.metric(
        "Average Rent",
        f"${df['price'].mean():,.0f}"
    )

    st.metric(
        "Median Rent",
        f"${df['price'].median():,.0f}"
    )

    st.metric(
        "Minimum Rent",
        f"${df['price'].min():,.0f}"
    )

    st.metric(
        "Maximum Rent",
        f"${df['price'].max():,.0f}"
    )


with chart_col:

    st.subheader("Monthly Rent Distribution")

    exclude_outliers = st.toggle(
        "Exclude extreme rents",
        value=True,
        help="Removes listings above $5,000 to make the main distribution easier to interpret."
    )

    if exclude_outliers:
        price_data = df[df["price"] <= 5000]
    else:
        price_data = df.copy()

    fig_price = px.histogram(
        price_data,
        x="price",
        nbins=50,
        labels={
            "price": "Monthly Rent ($)"
        },
    )

    fig_price.update_layout(
        xaxis_title="Monthly Rent ($)",
        yaxis_title="Number of Listings",
        bargap=0.05,
        height=420,
    )

    st.plotly_chart(
        fig_price,
        use_container_width=True
    )


# ------------------------------------------------------
# PROPERTY SIZE VS RENT
# ------------------------------------------------------

st.divider()

st.header("Rent vs Apartment Size")

scatter_data = df[
    (df["price"] <= 5000)
    & (df["square_feet"] <= 3000)
].dropna(
    subset=["price", "square_feet", "bedrooms"]
)

fig_scatter = px.scatter(
    scatter_data,
    x="square_feet",
    y="price",
    color="bedrooms",
    opacity=0.55,
    labels={
        "square_feet": "Apartment Size (ft²)",
        "price": "Monthly Rent ($)",
        "bedrooms": "Bedrooms",
    },
    hover_data=[
        "cityname",
        "state",
        "bathrooms",
    ],
)

fig_scatter.update_layout(
    height=500
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)

st.caption(
    "The visualization excludes extremely large apartments and rents above "
    "$5,000 to make the main relationship easier to interpret."
)


# ------------------------------------------------------
# GEOGRAPHIC SNAPSHOT
# ------------------------------------------------------

st.divider()

st.header("Geographic Snapshot")

geo_col1, geo_col2 = st.columns(2)

with geo_col1:

    state_summary = (
        df.groupby("state", dropna=True)
        .agg(
            listings=("price", "size"),
            median_rent=("price", "median")
        )
        .reset_index()
        .sort_values("listings", ascending=False)
        .head(15)
    )

    fig_states = px.bar(
        state_summary,
        x="state",
        y="listings",
        labels={
            "state": "State",
            "listings": "Number of Listings"
        },
        title="Top States by Number of Listings"
    )

    st.plotly_chart(
        fig_states,
        use_container_width=True
    )


with geo_col2:

    median_state_rent = (
        df.groupby("state", dropna=True)["price"]
        .median()
        .reset_index()
        .sort_values("price", ascending=False)
        .head(15)
    )

    fig_state_rent = px.bar(
        median_state_rent,
        x="state",
        y="price",
        labels={
            "state": "State",
            "price": "Median Monthly Rent ($)"
        },
        title="Highest Median Rents by State"
    )

    st.plotly_chart(
        fig_state_rent,
        use_container_width=True
    )


# ------------------------------------------------------
# PROJECT WORKFLOW
# ------------------------------------------------------

st.divider()

st.header("Project Workflow")

st.markdown(
    """
    **1. Exploratory Data Analysis**  
    Distribution analysis, multivariate relationships and geographic patterns.

    **2. Data Preparation**  
    Missing value imputation, encoding, scaling and feature engineering.

    **3. Model Development**  
    Baseline regression model and ensemble machine learning models.

    **4. Model Evaluation**  
    Cross-validation, RMSE, MAE and R² comparison.

    **5. Hyperparameter Optimization**  
    GridSearchCV and model tuning.

    **6. Final Application**  
    Interactive apartment rent prediction using the selected model.
    """
)


# ------------------------------------------------------
# DATASET PREVIEW
# ------------------------------------------------------

with st.expander("View raw dataset sample"):

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )