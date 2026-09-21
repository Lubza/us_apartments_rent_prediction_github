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

st.title("🗺️ Geografická analýza nájomného")

st.markdown(
    """
    Preskúmajte, ako sa ceny prenájmu bytov líšia v jednotlivých častiach
    Spojených štátov. Filtrujte ponuky podľa lokality a charakteristík bytu
    a porovnávajte mesačné nájomné aj nájomné na štvorcovú stopu.
    """
)

st.divider()


# ------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------

st.sidebar.header("Geografické filtre")


# STATE
state_options = sorted(
    df["state"]
    .dropna()
    .unique()
)

selected_states = st.sidebar.multiselect(
    "Štát",
    options=state_options,
    placeholder="Všetky štáty",
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
    "Mesto",
    options=city_options,
    placeholder="Všetky mestá",
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
    "Spálne",
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
    "Kúpeľne",
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
    "Mesačné nájomné ($)",
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
        "Vybraným filtrom nezodpovedajú žiadne ponuky."
    )
    st.stop()


# ------------------------------------------------------
# KPI
# ------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Ponuky",
        f"{len(filtered_df):,}",
    )

with col2:
    st.metric(
        "Medián nájomného",
        f"${filtered_df['price'].median():,.0f}",
    )

with col3:
    median_sqft_rent = filtered_df["rent_per_sqft"].median()

    st.metric(
        "Medián nájomného / ft²",
        (
            f"${median_sqft_rent:,.2f}"
            if pd.notna(median_sqft_rent)
            else "N/A"
        ),
    )

with col4:
    st.metric(
        "Mestá",
        filtered_df["cityname"].nunique(),
    )


st.divider()


# ------------------------------------------------------
# TABS
# ------------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "Mapa ponúk",
        "Analýza štátov",
        "Rebríček miest",
    ]
)


# ======================================================
# TAB 1 - LISTING MAP
# ======================================================

with tab1:

    st.header("Mapa ponúk bytov")

    st.markdown(
        """
        Preskúmajte geografické rozloženie ponúk bytov.
        Pomocou výberu metriky porovnajte mesačné nájomné alebo nájomné na ft².
        """
    )

    # --------------------------------------------------
    # MAP CONTROLS
    # --------------------------------------------------

    control_col1, control_col2 = st.columns([2, 1])

    with control_col1:

        metric = st.radio(
            "Metrika mapy",
            options=[
                "Mesačné nájomné",
                "Nájomné na ft²",
            ],
            horizontal=True,
        )

    with control_col2:

        marker_size = st.slider(
            "Veľkosť bodov",
            min_value=5,
            max_value=15,
            value=9,
            step=1,
        )


    # --------------------------------------------------
    # METRIC CONFIGURATION
    # --------------------------------------------------

    if metric == "Mesačné nájomné":

        map_variable = "price"
        map_label = "Mesačné nájomné ($)"

    else:

        map_variable = "rent_per_sqft"
        map_label = "Nájomné na ft² ($)"


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
            "Pre zvolené filtre nie sú dostupné platné geografické údaje."
        )

        st.stop()


    # --------------------------------------------------
    # POČET ZOBRAZENÝCH PONÚK
    # --------------------------------------------------

    available_listings = len(map_data)

    if available_listings > 100:

        listings_to_display = st.slider(
            "Počet ponúk zobrazených na mape",
            min_value=100,
            max_value=available_listings,
            value=min(3000, available_listings),
            step=1,
            help=(
                "Menší počet bodov udržiava interaktívnu mapu rýchlejšiu. "
                "Zvýšte hodnotu, ak chcete zobraziť väčšiu vzorku alebo "
                "všetky ponuky dostupné po použití aktuálnych filtrov."
            ),
        )

        st.caption(
            "Menej bodov zlepšuje plynulosť mapy. "
            "Posuňte slider na maximum, ak chcete zobraziť všetky dostupné ponuky."
        )

    else:

        listings_to_display = available_listings


    if listings_to_display < available_listings:

        map_data = map_data.sample(
            listings_to_display,
            random_state=42,
        )

        st.info(
            f"Zobrazuje sa náhodná vzorka {listings_to_display:,} z "
            f"{available_listings:,} dostupných ponúk. "
            "Zvýšte hodnotu slidera, ak chcete zobraziť viac bodov."
        )

    else:

        st.info(
            f"Zobrazujú sa všetky dostupné ponuky: {available_listings:,}."
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

    fig_map = px.scatter_map(
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
            "price": "Mesačné nájomné",
            "square_feet": "Plocha (ft²)",
            "bedrooms": "Spálne",
            "bathrooms": "Kúpeľne",
            "rent_per_sqft": "Nájomné / ft²",
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
        map_style="open-street-map",

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
        width="stretch",
    )


    # --------------------------------------------------
    # MAP NOTES
    # --------------------------------------------------

    note_col1, note_col2 = st.columns(2)

    with note_col1:

        st.caption(
            "Každý bod predstavuje jednu ponuku bytu. "
            "Po prejdení kurzorom zobrazíte detaily nehnuteľnosti."
        )

    with note_col2:

        st.caption(
            "Farebná škála používa rozsah 5.–95. percentilu, "
            "aby extrémne ponuky nedomínovali vizualizácii."
        )


    # --------------------------------------------------
    # CURRENT MAP SUMMARY
    # --------------------------------------------------

    st.subheader("Súhrn aktuálnej mapy")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)


    with summary_col1:

        st.metric(
            "Zobrazené ponuky",
            f"{len(map_data):,}",
        )


    with summary_col2:

        st.metric(
            "Medián nájomného",
            f"${map_data['price'].median():,.0f}",
        )


    with summary_col3:

        median_rent_sqft = (
            map_data["rent_per_sqft"].median()
        )

        st.metric(
            "Medián nájomného / ft²",
            (
                f"${median_rent_sqft:,.2f}"
                if pd.notna(median_rent_sqft)
                else "N/A"
            ),
        )


    with summary_col4:

        st.metric(
            "Mestá na mape",
            map_data["cityname"].nunique(),
        )


# ======================================================
# TAB 2 - STATE ANALYSIS
# ======================================================

with tab2:

    st.header("Analýza na úrovni štátov")

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
        "Metrika",
        options=[
            "Medián nájomného",
            "Priemerné nájomné",
            "Medián nájomného na ft²",
            "Počet ponúk",
        ],
    )


    metric_mapping = {
        "Medián nájomného": "median_rent",
        "Priemerné nájomné": "average_rent",
        "Medián nájomného na ft²":
            "median_rent_per_sqft",
        "Počet ponúk": "listings",
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
            "median_rent": "Medián nájomného",
            "average_rent": "Priemerné nájomné",
            "median_size": "Medián veľkosti",
            "median_rent_per_sqft":
                "Medián nájomného / ft²",
            "listings": "Ponuky",
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
        width="stretch",
    )


    # State ranking
    st.subheader("Rebríček štátov")

    ranking = state_summary.sort_values(
        selected_metric,
        ascending=False,
    )

    st.dataframe(
        ranking,
        width="stretch",
        hide_index=True,
        column_config={
            "state": "Štát",
            "listings":
                st.column_config.NumberColumn(
                    "Ponuky",
                    format="%d",
                ),
            "median_rent":
                st.column_config.NumberColumn(
                    "Medián nájomného",
                    format="$%.0f",
                ),
            "average_rent":
                st.column_config.NumberColumn(
                    "Priemerné nájomné",
                    format="$%.0f",
                ),
            "median_size":
                st.column_config.NumberColumn(
                    "Medián veľkosti",
                    format="%.0f ft²",
                ),
            "median_rent_per_sqft":
                st.column_config.NumberColumn(
                    "Medián nájomného / ft²",
                    format="$%.2f",
                ),
        },
    )


# ======================================================
# TAB 3 - CITY RANKINGS
# ======================================================

with tab3:

    st.header("Porovnanie miest")

    minimum_listings = st.slider(
        "Minimálny počet ponúk na mesto",
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
        "Zoradiť mestá podľa",
        options=[
            "Medián nájomného",
            "Medián nájomného na ft²",
            "Priemerné nájomné",
        ],
    )


    city_metric_mapping = {
        "Medián nájomného": "median_rent",
        "Medián nájomného na ft²":
            "median_rent_per_sqft",
        "Priemerné nájomné": "average_rent",
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
            "city_state": "Mesto",
            "median_rent": "Median Rent ($)",
            "average_rent": "Average Rent ($)",
            "median_rent_per_sqft":
                "Median Rent / ft² ($)",
        },
        title=(
            f"Top 20 miest podľa: {city_metric}"
        ),
    )

    fig_city.update_layout(
        height=650,
        yaxis_title="",
    )

    st.plotly_chart(
        fig_city,
        width="stretch",
    )


    st.subheader("Súhrn miest")

    st.dataframe(
        city_summary.sort_values(
            city_metric_column,
            ascending=False,
        ),
        width="stretch",
        hide_index=True,
    )