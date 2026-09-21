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

st.title("📊 Interaktívny prieskum dát")

st.markdown(
    """
    Interaktívne preskúmajte dataset prenájmov bytov.
    Pomocou filtrov môžete analyzovať ceny nájomného, charakteristiky bytov
    a vzťahy medzi premennými.
    """
)

st.divider()


# ------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------

st.sidebar.header("Filtre")


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


# PRICE RANGE
price_min = int(df["price"].dropna().min())
price_max = int(df["price"].dropna().max())

price_range = st.sidebar.slider(
    "Mesačné nájomné ($)",
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

st.subheader("Filtrovaný dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Ponuky",
        f"{len(filtered_df):,}"
    )

with col2:
    median_rent = filtered_df["price"].median()

    st.metric(
        "Medián nájomného",
        f"${median_rent:,.0f}"
        if not np.isnan(median_rent)
        else "N/A"
    )

with col3:
    median_size = filtered_df["square_feet"].median()

    st.metric(
        "Medián veľkosti",
        f"{median_size:,.0f} ft²"
        if not np.isnan(median_size)
        else "N/A"
    )

with col4:
    st.metric(
        "Mestá",
        filtered_df["cityname"].nunique()
    )


if filtered_df.empty:
    st.warning(
        "Vybraným filtrom nezodpovedajú žiadne ponuky."
    )
    st.stop()


# ------------------------------------------------------
# TABS
# ------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Rozdelenie",
        "Viacrozmerná analýza",
        "Analýza kategórií",
        "Tabuľka dát",
    ]
)


# ======================================================
# TAB 1 - DISTRIBUTION
# ======================================================

with tab1:

    st.header("Prieskum rozdelenia")

    col1, col2 = st.columns([1, 3])

    with col1:

        variable = st.selectbox(
            "Premenná",
            options=[
                "price",
                "square_feet",
                "bedrooms",
                "bathrooms",
            ],
            format_func=lambda x: {
                "price": "Mesačné nájomné",
                "square_feet": "Plocha (ft²)",
                "bedrooms": "Spálne",
                "bathrooms": "Kúpeľne",
            }[x]
        )

        bins = st.slider(
            "Počet intervalov",
            min_value=10,
            max_value=100,
            value=40,
            step=5,
        )

        log_scale = st.checkbox(
            "Logaritmická os X",
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
                "price": "Mesačné nájomné ($)",
                "square_feet": "Plocha (ft²)",
                "bedrooms": "Spálne",
                "bathrooms": "Kúpeľne",
            },
        )

        if log_scale:
            fig_distribution.update_xaxes(
                type="log"
            )

        fig_distribution.update_layout(
            height=500,
            yaxis_title="Počet ponúk",
        )

        st.plotly_chart(
            fig_distribution,
            use_container_width=True,
        )


    # Descriptive statistics
    st.subheader("Deskriptívne štatistiky")

    stats = (
        filtered_df[variable]
        .describe()
        .to_frame("Value")
    )

    st.dataframe(
        stats,
        use_container_width=True,
    )


# ======================================================
# TAB 2 - MULTIVARIATE
# ======================================================

with tab2:

    st.header("Viacrozmerná analýza")

    numeric_variables = {
        "Mesačné nájomné": "price",
        "Plocha (ft²)": "square_feet",
        "Spálne": "bedrooms",
        "Kúpeľne": "bathrooms",
        "Latitude": "latitude",
        "Longitude": "longitude",
    }

    control1, control2, control3 = st.columns(3)

    with control1:
        x_label = st.selectbox(
            "Os X",
            options=list(numeric_variables.keys()),
            index=1,
        )

    with control2:
        y_label = st.selectbox(
            "Os Y",
            options=list(numeric_variables.keys()),
            index=0,
        )

    with control3:
        color_variable = st.selectbox(
            "Farba podľa",
            options=[
                "Bez rozlíšenia",
                "Spálne",
                "Kúpeľne",
                "Štát",
            ],
        )


    x_variable = numeric_variables[x_label]
    y_variable = numeric_variables[y_label]

    color_mapping = {
        "Spálne": "bedrooms",
        "Kúpeľne": "bathrooms",
        "Štát": "state",
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
            "price": "Mesačné nájomné ($)",
            "square_feet": "Plocha (ft²)",
            "bedrooms": "Spálne",
            "bathrooms": "Kúpeľne",
            "latitude": "Latitude",
            "longitude": "Longitude",
            "state": "Štát",
        },
    )

    fig_scatter.update_layout(
        height=600,
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True,
    )


    # Correlations
    st.subheader("Korelačná matica")

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
            color="Korelácia"
        ),
    )

    fig_corr.update_layout(
        height=550,
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True,
    )


# ======================================================
# TAB 3 - CATEGORY ANALYSIS
# ======================================================

with tab3:

    st.header("Analýza kategórií")

    category = st.selectbox(
        "Analyzovať nájomné podľa",
        options=[
            "bedrooms",
            "bathrooms",
            "state",
            "pets_allowed",
            "has_photo",
        ],
        format_func=lambda x: {
            "bedrooms": "Spálne",
            "bathrooms": "Kúpeľne",
            "state": "Štát",
            "pets_allowed": "Povolené zvieratá",
            "has_photo": "Má fotografiu",
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


    st.subheader("Rozdelenie nájomného podľa kategórie")

    fig_box = px.box(
        category_data,
        x=category,
        y="price",
        points=False,
        labels={
            "price": "Mesačné nájomné ($)",
            "bedrooms": "Spálne",
            "bathrooms": "Kúpeľne",
            "state": "Štát",
            "pets_allowed": "Povolené zvieratá",
            "has_photo": "Má fotografiu",
        },
    )

    fig_box.update_layout(
        height=550,
    )

    st.plotly_chart(
        fig_box,
        use_container_width=True,
    )


    st.subheader("Medián nájomného")

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
        use_container_width=True,
        hide_index=True,
    )


# ======================================================
# TAB 4 - DATA TABLE
# ======================================================

with tab4:

    st.header("Filtrovaný dataset")

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
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Zobrazených {len(filtered_df):,} ponúk."
    )