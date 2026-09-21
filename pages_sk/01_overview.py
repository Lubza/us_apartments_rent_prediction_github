from pathlib import Path

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

st.title("🏠 Predikcia nájomného bytov v USA")

st.markdown(
    """
    Interaktívny data-science projekt skúmajúci trh s prenájmom bytov v USA
    a vyvíjajúci modely strojového učenia na predikciu mesačného nájomného.

    Projekt zahŕňa exploratívnu analýzu dát, preprocessing, tvorbu premenných,
    porovnanie modelov, krížovú validáciu, ladenie hyperparametrov a nasadenie
    modelu.
    """
)

st.divider()


# ------------------------------------------------------
# KPI METRICS
# ------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Ponuky",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Medián nájomného",
        f"${df['price'].median():,.0f}"
    )

with col3:
    st.metric(
        "Medián veľkosti",
        f"{df['square_feet'].median():,.0f} ft²"
    )

with col4:
    st.metric(
        "Štáty",
        f"{df['state'].nunique():,}"
    )

with col5:
    st.metric(
        "Mestá",
        f"{df['cityname'].nunique():,}"
    )


# ------------------------------------------------------
# RENT DISTRIBUTION
# ------------------------------------------------------

st.divider()

st.header("Prehľad trhu s prenájmom")

chart_col, stats_col = st.columns([3, 1])

with stats_col:

    st.subheader("Štatistiky nájomného")

    st.metric(
        "Priemerné nájomné",
        f"${df['price'].mean():,.0f}"
    )

    st.metric(
        "Medián nájomného",
        f"${df['price'].median():,.0f}"
    )

    st.metric(
        "Minimálne nájomné",
        f"${df['price'].min():,.0f}"
    )

    st.metric(
        "Maximálne nájomné",
        f"${df['price'].max():,.0f}"
    )


with chart_col:

    st.subheader("Rozdelenie mesačného nájomného")

    exclude_outliers = st.toggle(
        "Vylúčiť extrémne nájomné",
        value=True,
        help="Odstráni ponuky nad 5 000 USD, aby bolo hlavné rozdelenie prehľadnejšie."
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
            "price": "Mesačné nájomné ($)"
        },
    )

    fig_price.update_layout(
        xaxis_title="Mesačné nájomné ($)",
        yaxis_title="Počet ponúk",
        bargap=0.05,
        height=420,
    )

    st.plotly_chart(
        fig_price,
        width="stretch"
    )


# ------------------------------------------------------
# PROPERTY SIZE VS RENT
# ------------------------------------------------------

st.divider()

st.header("Nájomné vs. veľkosť bytu")

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
        "square_feet": "Veľkosť bytu (ft²)",
        "price": "Mesačné nájomné ($)",
        "bedrooms": "Spálne",
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
    width="stretch"
)

st.caption(
    "Vizualizácia vylučuje extrémne veľké byty a nájomné nad 5 000 USD, "
    "aby bol hlavný vzťah jednoduchšie interpretovateľný."
)


# ------------------------------------------------------
# GEOGRAPHIC SNAPSHOT
# ------------------------------------------------------

st.divider()

st.header("Geografický prehľad")

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
            "state": "Štát",
            "listings": "Počet ponúk"
        },
        title="Štáty s najvyšším počtom ponúk"
    )

    st.plotly_chart(
        fig_states,
        width="stretch"
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
            "state": "Štát",
            "price": "Medián mesačného nájomného ($)"
        },
        title="Najvyšší medián nájomného podľa štátu"
    )

    st.plotly_chart(
        fig_state_rent,
        width="stretch"
    )


# ------------------------------------------------------
# PROJECT WORKFLOW
# ------------------------------------------------------

st.divider()

st.header("Priebeh projektu")

st.markdown(
    """
    **1. Exploratívna analýza dát**  
    Analýza rozdelení, viacrozmerných vzťahov a geografických vzorcov.

    **2. Príprava dát**  
    Imputácia chýbajúcich hodnôt, kódovanie, škálovanie a tvorba premenných.

    **3. Vývoj modelov**  
    Regresné modely Random Forest, XGBoost, LASSO a Ridge.

    **4. Vyhodnotenie modelov**  
    Krížová validácia a porovnanie R² na odloženej testovacej vzorke.

    **5. Optimalizácia hyperparametrov**  
    RandomizedSearchCV so spoločným ladením preprocessingu a modelu.

    **6. Finálna aplikácia**  
    Interaktívna predikcia nájomného pomocou zvoleného modelu.
    """
)


# ------------------------------------------------------
# DATASET PREVIEW
# ------------------------------------------------------

with st.expander("Zobraziť ukážku surového datasetu"):

    st.dataframe(
        df.head(100),
        width="stretch",
        hide_index=True
    )