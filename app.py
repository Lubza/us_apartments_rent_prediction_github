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
# PAGE DEFINITIONS
# ------------------------------------------------------

overview_page = st.Page(
    "pages/00_overview.py",
    title="Overview",
    icon="🏠",
    default=True,
)

data_explorer_page = st.Page(
    "pages/01_data_explorer.py",
    title="Data Explorer",
    icon="📊",
)

geographic_page = st.Page(
    "pages/02_geographic_analysis.py",
    title="Geographic Analysis",
    icon="🗺️",
)

data_preparation_page = st.Page(
    "pages/03_data_preparation.py",
    title="Data Preparation",
    icon="🧹",
)


# ------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------

pg = st.navigation(
    [
        overview_page,
        data_explorer_page,
        geographic_page,
        data_preparation_page,
    ],
    position="sidebar",
)


# ------------------------------------------------------
# RUN SELECTED PAGE
# ------------------------------------------------------

pg.run()