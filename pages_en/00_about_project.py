import pandas as pd
import streamlit as st


# ------------------------------------------------------
# HEADER
# ------------------------------------------------------

st.title("About the Project")

st.markdown(
    """
    This page provides a non-technical introduction to the project and
    describes the variables contained in the apartment rental dataset.
    """
)

st.divider()


# ======================================================
# 1. NON-TECHNICAL SUMMARY
# ======================================================

st.header("Non-technical Summary")

st.markdown(
    """
    The goal of this project is to model apartment rental prices in the United
    States. The dataset contains approximately **10,000 observations from 2019**
    and, based on the structure of the `source` variable, was most likely
    created by web scraping several real-estate listing websites.

    Using the applied approach, the model achieved **R² = 76.8%** on the test
    set for the **logarithm of price**.

    For the non-log-transformed price, **R² = 37.0%**. However, after filtering
    out the top 2% of prices, the score increases to **R² = 74.0%**.
    """
)

st.info(
    """
    **Note from the solution explanation:** the specific business case is not
    defined, so this summary focuses mainly on the dataset and the achieved
    modeling results. In a real project, this section should clearly state who
    will use the model, how the predictions will be used, and what level of
    prediction error is acceptable from a business perspective. It is also
    worth considering whether a technical metric such as R² belongs directly
    in a management summary.
    """
)

st.divider()


# ======================================================
# 2. DATASET DESCRIPTION
# ======================================================

st.header("Dataset Description")

st.markdown(
    """
    The dataset contains apartment rental listings with property
    characteristics, geographic information, listing metadata and rental
    prices.
    """
)

dataset_description = pd.DataFrame(
    {
        "#": list(range(1, 23)),
        "Variable": [
            "id",
            "category",
            "title",
            "body",
            "amenities",
            "bathrooms",
            "bedrooms",
            "currency",
            "fee",
            "has_photo",
            "pets_allowed",
            "price",
            "price_display",
            "price_type",
            "square_feet",
            "address",
            "cityname",
            "state",
            "latitude",
            "longitude",
            "source",
            "time",
        ],
        "Description": [
            "Unique apartment listing identifier.",
            "Listing category.",
            "Listing title text.",
            "Listing description text.",
            (
                "Apartment amenities such as air conditioning, basketball "
                "court, cable TV, fitness center, internet access, pool, "
                "refrigerator, etc."
            ),
            "Number of bathrooms.",
            "Number of bedrooms.",
            "Listing currency.",
            "Possible additional fee.",
            "Indicates whether the listing contains apartment photos.",
            "Which pets are allowed (e.g. dogs, cats).",
            "Apartment rental price.",
            "Price formatted for display.",
            "Price frequency/type in USD.",
            "Apartment size in square feet.",
            "Apartment location/address.",
            "City where the apartment is located.",
            "US state where the apartment is located.",
            "Latitude of the apartment location.",
            "Longitude of the apartment location.",
            "Source of the listing.",
            "Time when the listing was created.",
        ],
    }
)

st.dataframe(
    dataset_description,
    width="stretch",
    hide_index=True,
    column_config={
        "#": st.column_config.NumberColumn(
            "#",
            width="small",
            format="%d",
        ),
        "Variable": st.column_config.TextColumn(
            "Variable",
            width="medium",
        ),
        "Description": st.column_config.TextColumn(
            "Description",
            width="large",
        ),
    },
)

st.caption(
    "The summary and dataset descriptions on this page are based on the "
    "introductory sections of the project Jupyter notebook."
)
