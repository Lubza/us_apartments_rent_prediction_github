from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "apartments_for_rent_10K.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        encoding="cp1252",
    )

    numeric_columns = [
        "price",
        "square_feet",
        "bedrooms",
        "bathrooms",
        "latitude",
        "longitude",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df


df = load_data()


# ------------------------------------------------------
# HEADER
# ------------------------------------------------------

st.title("📊 Interactive Data Explorer")

st.markdown(
    """
    Explore the apartment rental dataset interactively.
    Use the filters to investigate rental prices, apartment characteristics
    and relationships between variables.
    """
)

st.divider()


# ------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------

st.sidebar.header("Filters")


# STATE
state_options = sorted(
    df["state"]
    .dropna()
    .unique()
)

selected_states = st.sidebar.multiselect(
    "State",
    options=state_options,
    placeholder="All states",
)


# Apply state filter first
filtered_df = df.copy()

if selected_states:
    filtered_df = filtered_df[
        filtered_df["state"].isin(selected_states)
    ]


# CITY
city_options = sorted(
    filtered_df["cityname"]
    .dropna()
    .unique()
)

selected_cities = st.sidebar.multiselect(
    "City",
    options=city_options,
    placeholder="All cities",
)

if selected_cities:
    filtered_df = filtered_df[
        filtered_df["cityname"].isin(selected_cities)
    ]


# BEDROOMS
bedroom_options = sorted(
    filtered_df["bedrooms"]
    .dropna()
    .unique()
)

selected_bedrooms = st.sidebar.multiselect(
    "Bedrooms",
    options=bedroom_options,
)

if selected_bedrooms:
    filtered_df = filtered_df[
        filtered_df["bedrooms"].isin(selected_bedrooms)
    ]


# BATHROOMS
bathroom_options = sorted(
    filtered_df["bathrooms"]
    .dropna()
    .unique()
)

selected_bathrooms = st.sidebar.multiselect(
    "Bathrooms",
    options=bathroom_options,
)

if selected_bathrooms:
    filtered_df = filtered_df[
        filtered_df["bathrooms"].isin(selected_bathrooms)
    ]


# PRICE RANGE
price_min = int(df["price"].dropna().min())
price_max = int(df["price"].dropna().max())

price_range = st.sidebar.slider(
    "Monthly Rent ($)",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, min(5000, price_max)),
)

filtered_df = filtered_df[
    filtered_df["price"].between(
        price_range[0],
        price_range[1]
    )
]


# ------------------------------------------------------
# FILTER SUMMARY
# ------------------------------------------------------

st.subheader("Filtered Dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Listings",
        f"{len(filtered_df):,}"
    )

with col2:
    median_rent = filtered_df["price"].median()

    st.metric(
        "Median Rent",
        f"${median_rent:,.0f}"
        if not np.isnan(median_rent)
        else "N/A"
    )

with col3:
    median_size = filtered_df["square_feet"].median()

    st.metric(
        "Median Size",
        f"{median_size:,.0f} ft²"
        if not np.isnan(median_size)
        else "N/A"
    )

with col4:
    st.metric(
        "Cities",
        filtered_df["cityname"].nunique()
    )


if filtered_df.empty:
    st.warning(
        "No listings match the selected filters."
    )
    st.stop()


# ------------------------------------------------------
# TABS
# ------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Distribution",
        "Multivariate Analysis",
        "Category Analysis",
        "Data Table",
    ]
)


# ======================================================
# TAB 1 - DISTRIBUTION
# ======================================================

with tab1:

    st.header("Distribution Explorer")

    col1, col2 = st.columns([1, 3])

    with col1:

        variable = st.selectbox(
            "Variable",
            options=[
                "price",
                "square_feet",
                "bedrooms",
                "bathrooms",
            ],
            format_func=lambda x: {
                "price": "Monthly Rent",
                "square_feet": "Square Feet",
                "bedrooms": "Bedrooms",
                "bathrooms": "Bathrooms",
            }[x]
        )

        bins = st.slider(
            "Number of bins",
            min_value=10,
            max_value=100,
            value=40,
            step=5,
        )

        log_scale = st.checkbox(
            "Logarithmic X-axis",
            value=False,
        )


    with col2:

        distribution_data = filtered_df[
            filtered_df[variable].notna()
        ]

        fig_distribution = px.histogram(
            distribution_data,
            x=variable,
            nbins=bins,
            marginal="box",
            labels={
                "price": "Monthly Rent ($)",
                "square_feet": "Square Feet",
                "bedrooms": "Bedrooms",
                "bathrooms": "Bathrooms",
            },
        )

        if log_scale:
            fig_distribution.update_xaxes(
                type="log"
            )

        fig_distribution.update_layout(
            height=500,
            yaxis_title="Number of Listings",
        )

        st.plotly_chart(
            fig_distribution,
            width="stretch",
        )


    # Descriptive statistics
    st.subheader("Descriptive Statistics")

    stats = (
        filtered_df[variable]
        .describe()
        .to_frame("Value")
    )

    st.dataframe(
        stats,
        width="stretch",
    )


# ======================================================
# TAB 2 - MULTIVARIATE
# ======================================================

with tab2:

    st.header("Multivariate Analysis")

    numeric_variables = {
        "Monthly Rent": "price",
        "Square Feet": "square_feet",
        "Bedrooms": "bedrooms",
        "Bathrooms": "bathrooms",
        "Latitude": "latitude",
        "Longitude": "longitude",
    }

    control1, control2, control3 = st.columns(3)

    with control1:
        x_label = st.selectbox(
            "X-axis",
            options=list(numeric_variables.keys()),
            index=1,
        )

    with control2:
        y_label = st.selectbox(
            "Y-axis",
            options=list(numeric_variables.keys()),
            index=0,
        )

    with control3:
        color_variable = st.selectbox(
            "Color by",
            options=[
                "None",
                "Bedrooms",
                "Bathrooms",
                "State",
            ],
        )


    x_variable = numeric_variables[x_label]
    y_variable = numeric_variables[y_label]

    color_mapping = {
        "Bedrooms": "bedrooms",
        "Bathrooms": "bathrooms",
        "State": "state",
    }

    color_column = color_mapping.get(
        color_variable
    )


    scatter_columns = [
        x_variable,
        y_variable,
    ]

    if color_column:
        scatter_columns.append(color_column)

    scatter_data = filtered_df.dropna(
        subset=[
            x_variable,
            y_variable,
        ]
    )


    fig_scatter = px.scatter(
        scatter_data,
        x=x_variable,
        y=y_variable,
        color=color_column,
        opacity=0.55,
        hover_data=[
            "cityname",
            "state",
            "bedrooms",
            "bathrooms",
            "square_feet",
            "price",
        ],
        labels={
            "price": "Monthly Rent ($)",
            "square_feet": "Square Feet",
            "bedrooms": "Bedrooms",
            "bathrooms": "Bathrooms",
            "latitude": "Latitude",
            "longitude": "Longitude",
            "state": "State",
        },
    )

    fig_scatter.update_layout(
        height=600,
    )

    st.plotly_chart(
        fig_scatter,
        width="stretch",
    )


    # Correlations
    st.subheader("Correlation Matrix")

    correlation_columns = [
        "price",
        "square_feet",
        "bedrooms",
        "bathrooms",
        "latitude",
        "longitude",
    ]

    correlation_matrix = (
        filtered_df[correlation_columns]
        .corr()
        .round(2)
    )

    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        labels=dict(
            color="Correlation"
        ),
    )

    fig_corr.update_layout(
        height=550,
    )

    st.plotly_chart(
        fig_corr,
        width="stretch",
    )


# ======================================================
# TAB 3 - CATEGORY ANALYSIS
# ======================================================

with tab3:

    st.header("Categorical Analysis")

    category = st.selectbox(
        "Analyze rent by",
        options=[
            "bedrooms",
            "bathrooms",
            "state",
            "pets_allowed",
            "has_photo",
        ],
        format_func=lambda x: {
            "bedrooms": "Bedrooms",
            "bathrooms": "Bathrooms",
            "state": "State",
            "pets_allowed": "Pets Allowed",
            "has_photo": "Has Photo",
        }[x]
    )


    category_data = filtered_df[
        filtered_df[category].notna()
    ].copy()


    # Prevent unreadable chart for state
    if category == "state":

        top_states = (
            category_data["state"]
            .value_counts()
            .head(15)
            .index
        )

        category_data = category_data[
            category_data["state"].isin(
                top_states
            )
        ]


    st.subheader("Rent Distribution by Category")

    fig_box = px.box(
        category_data,
        x=category,
        y="price",
        points=False,
        labels={
            "price": "Monthly Rent ($)",
            "bedrooms": "Bedrooms",
            "bathrooms": "Bathrooms",
            "state": "State",
            "pets_allowed": "Pets Allowed",
            "has_photo": "Has Photo",
        },
    )

    fig_box.update_layout(
        height=550,
    )

    st.plotly_chart(
        fig_box,
        width="stretch",
    )


    st.subheader("Median Rent")

    category_summary = (
        category_data
        .groupby(category)["price"]
        .agg(
            listings="count",
            median_rent="median",
            average_rent="mean",
        )
        .reset_index()
        .sort_values(
            "median_rent",
            ascending=False,
        )
    )

    st.dataframe(
        category_summary,
        width="stretch",
        hide_index=True,
    )


# ======================================================
# TAB 4 - DATA TABLE
# ======================================================

with tab4:

    st.header("Filtered Dataset")

    display_columns = [
        "price",
        "square_feet",
        "bedrooms",
        "bathrooms",
        "cityname",
        "state",
        "pets_allowed",
        "has_photo",
        "latitude",
        "longitude",
    ]

    st.dataframe(
        filtered_df[display_columns],
        width="stretch",
        hide_index=True,
    )

    st.caption(
        f"Showing {len(filtered_df):,} listings."
    )