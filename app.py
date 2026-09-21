import streamlit as st


# ------------------------------------------------------
# GLOBAL APP CONFIGURATION
# ------------------------------------------------------

st.set_page_config(
    page_title="US Apartment Rent Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------
# LANGUAGE
# ------------------------------------------------------

language = st.sidebar.radio(
    "Language / Jazyk",
    options=["English", "Slovenčina"],
    horizontal=True,
    key="dashboard_language",
)


# ------------------------------------------------------
# PAGE DEFINITIONS
# ------------------------------------------------------

if language == "English":

    pages = [
        st.Page(
            "pages_en/00_about_project.py",
            title="About the Project",
            icon="ℹ️",
            default=True,
        ),
        st.Page(
            "pages_en/01_overview.py",
            title="Overview",
            icon="🏠",
        ),
        st.Page(
            "pages_en/02_data_explorer.py",
            title="Data Explorer",
            icon="📊",
        ),
        st.Page(
            "pages_en/03_geographic_analysis.py",
            title="Geographic Analysis",
            icon="🗺️",
        ),
        st.Page(
            "pages_en/04_data_preparation.py",
            title="Data Preparation",
            icon="🧹",
        ),
        st.Page(
            "pages_en/05_model_development.py",
            title="Model Development",
            icon="🤖",
        ),
        st.Page(
            "pages_en/06_rent_predictor.py",
            title="Rent Predictor",
            icon="💵",
        ),
    ]

else:

    pages = [
        st.Page(
            "pages_sk/00_about_project.py",
            title="O projekte",
            icon="ℹ️",
            default=True,
        ),
        st.Page(
            "pages_sk/01_overview.py",
            title="Prehľad",
            icon="🏠",
        ),
        st.Page(
            "pages_sk/02_data_explorer.py",
            title="Prieskum dát",
            icon="📊",
        ),
        st.Page(
            "pages_sk/03_geographic_analysis.py",
            title="Geografická analýza",
            icon="🗺️",
        ),
        st.Page(
            "pages_sk/04_data_preparation.py",
            title="Príprava dát",
            icon="🧹",
        ),
        st.Page(
            "pages_sk/05_model_development.py",
            title="Vývoj modelov",
            icon="🤖",
        ),
        st.Page(
            "pages_sk/06_rent_predictor.py",
            title="Predikcia nájomného",
            icon="💵",
        ),
    ]


# ------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------

pg = st.navigation(
    pages,
    position="sidebar",
)

pg.run()
