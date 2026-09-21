from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------

DATA_PATH = (
    Path(__file__).parent.parent
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

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # New feature
    df["rent_per_sqft"] = np.where(
        df["square_feet"] > 0,
        df["price"] / df["square_feet"],
        np.nan,
    )

    return df


df = load_data()


# ------------------------------------------------------
# HEADER
# ------------------------------------------------------

st.title("🗺️ Geographic Rent Analysis")

st.markdown(
    """
    Explore how apartment rental prices vary geographically across the
    United States. Filter listings by location and property characteristics
    and compare both monthly rent and rent per square foot.
    """
)

st.divider()


# ------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------

st.sidebar.header("Geographic Filters")


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


# PRICE
min_price = int(df["price"].min())
max_price = int(df["price"].max())

price_range = st.sidebar.slider(
    "Monthly Rent ($)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, min(5000, max_price)),
)

filtered_df = filtered_df[
    filtered_df["price"].between(
        price_range[0],
        price_range[1],
    )
]


if filtered_df.empty:
    st.warning(
        "No listings match the selected filters."
    )
    st.stop()


# ------------------------------------------------------
# KPI
# ------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Listings",
        f"{len(filtered_df):,}",
    )

with col2:
    st.metric(
        "Median Rent",
        f"${filtered_df['price'].median():,.0f}",
    )

with col3:
    median_sqft_rent = filtered_df["rent_per_sqft"].median()

    st.metric(
        "Median Rent / ft²",
        (
            f"${median_sqft_rent:,.2f}"
            if pd.notna(median_sqft_rent)
            else "N/A"
        ),
    )

with col4:
    st.metric(
        "Cities",
        filtered_df["cityname"].nunique(),
    )


st.divider()


# ------------------------------------------------------
# TABS
# ------------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "Listing Map",
        "State Analysis",
        "City Rankings",
    ]
)


# ======================================================
# TAB 1 - LISTING MAP
# ======================================================

with tab1:

    st.header("Apartment Listing Map")

    st.markdown(
        """
        Explore the geographic distribution of apartment listings.
        Use the metric selector to compare monthly rent or rent per square foot.
        """
    )

    # --------------------------------------------------
    # MAP CONTROLS
    # --------------------------------------------------

    control_col1, control_col2 = st.columns([2, 1])

    with control_col1:

        metric = st.radio(
            "Map metric",
            options=[
                "Monthly Rent",
                "Rent per Square Foot",
            ],
            horizontal=True,
        )

    with control_col2:

        marker_size = st.slider(
            "Point size",
            min_value=5,
            max_value=15,
            value=9,
            step=1,
        )


    # --------------------------------------------------
    # METRIC CONFIGURATION
    # --------------------------------------------------

    if metric == "Monthly Rent":

        map_variable = "price"
        map_label = "Monthly Rent ($)"

    else:

        map_variable = "rent_per_sqft"
        map_label = "Rent per ft² ($)"


    # --------------------------------------------------
    # PREPARE MAP DATA
    # --------------------------------------------------

    map_data = filtered_df.dropna(
        subset=[
            "latitude",
            "longitude",
            map_variable,
        ]
    ).copy()


    if map_data.empty:

        st.warning(
            "No valid geographic data is available "
            "for the selected filters."
        )

        st.stop()


    # --------------------------------------------------
    # PERFORMANCE LIMIT
    # --------------------------------------------------

    if len(map_data) > 3000:

        map_data = map_data.sample(
            3000,
            random_state=42,
        )

        st.info(
            "The map displays a random sample of 3,000 listings "
            "to keep the visualization responsive."
        )


    # --------------------------------------------------
    # OUTLIER-AWARE COLOR SCALE
    # --------------------------------------------------

    color_min = map_data[
        map_variable
    ].quantile(0.05)

    color_max = map_data[
        map_variable
    ].quantile(0.95)


    # Fallback if filtered data have almost no variation
    if (
        pd.isna(color_min)
        or pd.isna(color_max)
        or color_min == color_max
    ):

        color_min = map_data[
            map_variable
        ].min()

        color_max = map_data[
            map_variable
        ].max()


    # --------------------------------------------------
    # DYNAMIC MAP CENTER
    # --------------------------------------------------

    map_center = {
        "lat": map_data["latitude"].mean(),
        "lon": map_data["longitude"].mean(),
    }


    # Dynamic zoom based on selected filters
    if selected_cities:

        zoom_level = 8

    elif selected_states:

        zoom_level = 5

    else:

        zoom_level = 3


    # --------------------------------------------------
    # MAP
    # --------------------------------------------------

    fig_map = px.scatter_mapbox(
        map_data,
        lat="latitude",
        lon="longitude",
        color=map_variable,
        color_continuous_scale="Viridis",
        range_color=[
            color_min,
            color_max,
        ],
        hover_name="cityname",
        hover_data={
            "state": True,
            "price": ":$,.0f",
            "square_feet": ":,.0f",
            "bedrooms": ":.0f",
            "bathrooms": ":.1f",
            "rent_per_sqft": ":.2f",
            "latitude": False,
            "longitude": False,
        },
        labels={
            "price": "Monthly Rent",
            "square_feet": "Square Feet",
            "bedrooms": "Bedrooms",
            "bathrooms": "Bathrooms",
            "rent_per_sqft": "Rent / ft²",
        },
        zoom=zoom_level,
        center=map_center,
        height=680,
    )


    # --------------------------------------------------
    # MAP STYLE
    # --------------------------------------------------

    fig_map.update_traces(
        marker={
            "size": marker_size,
            "opacity": 0.82,
        }
    )


    fig_map.update_layout(
        mapbox_style="open-street-map",

        margin={
            "r": 0,
            "t": 0,
            "l": 0,
            "b": 0,
        },

        coloraxis_colorbar={
            "title": map_label,
            "thickness": 15,
            "len": 0.75,
        },
    )


    st.plotly_chart(
        fig_map,
        use_container_width=True,
    )


    # --------------------------------------------------
    # MAP NOTES
    # --------------------------------------------------

    note_col1, note_col2 = st.columns(2)

    with note_col1:

        st.caption(
            "Each point represents one apartment listing. "
            "Hover over a point to see property details."
        )

    with note_col2:

        st.caption(
            "The color scale uses the 5th–95th percentile range "
            "so extreme listings do not dominate the visualization."
        )


    # --------------------------------------------------
    # CURRENT MAP SUMMARY
    # --------------------------------------------------

    st.subheader("Current Map Summary")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)


    with summary_col1:

        st.metric(
            "Mapped Listings",
            f"{len(map_data):,}",
        )


    with summary_col2:

        st.metric(
            "Median Rent",
            f"${map_data['price'].median():,.0f}",
        )


    with summary_col3:

        median_rent_sqft = (
            map_data["rent_per_sqft"].median()
        )

        st.metric(
            "Median Rent / ft²",
            (
                f"${median_rent_sqft:,.2f}"
                if pd.notna(median_rent_sqft)
                else "N/A"
            ),
        )


    with summary_col4:

        st.metric(
            "Cities on Map",
            map_data["cityname"].nunique(),
        )


# ======================================================
# TAB 2 - STATE ANALYSIS
# ======================================================

with tab2:

    st.header("State-Level Analysis")

    state_summary = (
        filtered_df
        .dropna(subset=["state"])
        .groupby("state")
        .agg(
            listings=("price", "size"),
            median_rent=("price", "median"),
            average_rent=("price", "mean"),
            median_size=("square_feet", "median"),
            median_rent_per_sqft=(
                "rent_per_sqft",
                "median",
            ),
        )
        .reset_index()
    )


    state_metric = st.selectbox(
        "Metric",
        options=[
            "Median Rent",
            "Average Rent",
            "Median Rent per ft²",
            "Number of Listings",
        ],
    )


    metric_mapping = {
        "Median Rent": "median_rent",
        "Average Rent": "average_rent",
        "Median Rent per ft²":
            "median_rent_per_sqft",
        "Number of Listings": "listings",
    }

    selected_metric = metric_mapping[
        state_metric
    ]


    fig_choropleth = px.choropleth(
        state_summary,
        locations="state",
        locationmode="USA-states",
        color=selected_metric,
        scope="usa",
        hover_data={
            "listings": ":,",
            "median_rent": ":$,.0f",
            "average_rent": ":$,.0f",
            "median_size": ":,.0f",
            "median_rent_per_sqft": ":$.2f",
        },
        labels={
            "median_rent": "Median Rent",
            "average_rent": "Average Rent",
            "median_size": "Median Size",
            "median_rent_per_sqft":
                "Median Rent / ft²",
            "listings": "Listings",
        },
    )

    fig_choropleth.update_layout(
        height=600,
        margin={
            "r": 0,
            "t": 20,
            "l": 0,
            "b": 0,
        },
    )

    st.plotly_chart(
        fig_choropleth,
        use_container_width=True,
    )


    # State ranking
    st.subheader("State Ranking")

    ranking = state_summary.sort_values(
        selected_metric,
        ascending=False,
    )

    st.dataframe(
        ranking,
        use_container_width=True,
        hide_index=True,
        column_config={
            "state": "State",
            "listings":
                st.column_config.NumberColumn(
                    "Listings",
                    format="%d",
                ),
            "median_rent":
                st.column_config.NumberColumn(
                    "Median Rent",
                    format="$%.0f",
                ),
            "average_rent":
                st.column_config.NumberColumn(
                    "Average Rent",
                    format="$%.0f",
                ),
            "median_size":
                st.column_config.NumberColumn(
                    "Median Size",
                    format="%.0f ft²",
                ),
            "median_rent_per_sqft":
                st.column_config.NumberColumn(
                    "Median Rent / ft²",
                    format="$%.2f",
                ),
        },
    )


# ======================================================
# TAB 3 - CITY RANKINGS
# ======================================================

with tab3:

    st.header("City-Level Comparison")

    minimum_listings = st.slider(
        "Minimum number of listings per city",
        min_value=1,
        max_value=50,
        value=10,
    )


    city_summary = (
        filtered_df
        .dropna(
            subset=[
                "cityname",
                "state",
            ]
        )
        .groupby(
            [
                "cityname",
                "state",
            ]
        )
        .agg(
            listings=("price", "size"),
            median_rent=("price", "median"),
            average_rent=("price", "mean"),
            median_size=("square_feet", "median"),
            median_rent_per_sqft=(
                "rent_per_sqft",
                "median",
            ),
        )
        .reset_index()
    )


    city_summary = city_summary[
        city_summary["listings"]
        >= minimum_listings
    ]


    city_metric = st.selectbox(
        "Rank cities by",
        options=[
            "Median Rent",
            "Median Rent per ft²",
            "Average Rent",
        ],
    )


    city_metric_mapping = {
        "Median Rent": "median_rent",
        "Median Rent per ft²":
            "median_rent_per_sqft",
        "Average Rent": "average_rent",
    }

    city_metric_column = city_metric_mapping[
        city_metric
    ]


    top_cities = (
        city_summary
        .sort_values(
            city_metric_column,
            ascending=False,
        )
        .head(20)
        .copy()
    )


    top_cities["city_state"] = (
        top_cities["cityname"]
        + ", "
        + top_cities["state"]
    )


    fig_city = px.bar(
        top_cities.sort_values(
            city_metric_column,
            ascending=True,
        ),
        x=city_metric_column,
        y="city_state",
        orientation="h",
        hover_data=[
            "listings",
            "median_size",
        ],
        labels={
            "city_state": "City",
            "median_rent": "Median Rent ($)",
            "average_rent": "Average Rent ($)",
            "median_rent_per_sqft":
                "Median Rent / ft² ($)",
        },
        title=(
            f"Top 20 Cities by {city_metric}"
        ),
    )

    fig_city.update_layout(
        height=650,
        yaxis_title="",
    )

    st.plotly_chart(
        fig_city,
        use_container_width=True,
    )


    st.subheader("City Summary")

    st.dataframe(
        city_summary.sort_values(
            city_metric_column,
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )