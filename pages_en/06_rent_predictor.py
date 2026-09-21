from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------------------------------
# PATHS
# ------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "apartments_for_rent_10K.csv"
)


# ------------------------------------------------------
# MODEL CONFIGURATION
# ------------------------------------------------------

MODEL_CONFIG = {
    "XGBoost": {
        "filename": "model_xgb.pkl",
        "description": (
            "Gradient-boosting model selected as the strongest model "
            "in the notebook comparison."
        ),
        "cv_r2": 0.766,
        "test_r2": 0.768,
    },
    "Random Forest": {
        "filename": "model_rf.pkl",
        "description": (
            "Bagging-based ensemble of decision trees. It achieved a "
            "slightly lower validation score than XGBoost."
        ),
        "cv_r2": 0.751,
        "test_r2": 0.752,
    },
    "LASSO": {
        "filename": "model_lasso.pkl",
        "description": (
            "L1-regularized linear regression used as a simpler reference model."
        ),
        "cv_r2": 0.679,
        "test_r2": 0.680,
    },
    "Ridge": {
        "filename": "model_ridge.pkl",
        "description": (
            "L2-regularized linear regression. It retains correlated features "
            "while shrinking their coefficients toward zero."
        ),
        "cv_r2": 0.696,
        "test_r2": 0.693,
    },
}


# ------------------------------------------------------
# REQUIRED MODEL INPUTS
# ------------------------------------------------------

MODEL_FEATURES = [
    "bathrooms",
    "bedrooms",
    "log_square_feet",
    "cityname",
    "state",
    "latitude",
    "longitude",
    "source",
    "n_amenities",
    "pets_allowed_int",
    "has_photo_int",
    "latitude_sq",
    "longitude_sq",
]


# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(
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
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    # Convert weekly prices to an approximate monthly basis,
    # following the notebook preprocessing.
    if {"price_type", "price"}.issubset(data.columns):
        weekly_mask = data["price_type"].isin(
            ["Monthly|Weekly", "Weekly"]
        )

        data.loc[weekly_mask, "price"] = (
            data.loc[weekly_mask, "price"] * 4
        )

    # Number of amenities, following the notebook logic.
    if "amenities" in data.columns:
        amenities_list = data["amenities"].str.split(",")
        amenities_list = amenities_list.where(
            amenities_list.notna(),
            "",
        )
        data["n_amenities"] = amenities_list.apply(len)

    return data


# ------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------

def find_model_path(filename):
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "models" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as file:
        saved_object = pickle.load(file)

    # The notebook stores RandomizedSearchCV objects.
    # For deployment, only the fitted best estimator is required.
    if hasattr(saved_object, "best_estimator_"):
        return saved_object.best_estimator_

    return saved_object


# ------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------

def city_coordinates(data, state, city):
    location_data = data[
        (data["state"] == state)
        & (data["cityname"] == city)
    ].dropna(
        subset=[
            "latitude",
            "longitude",
        ]
    )

    if location_data.empty:
        return np.nan, np.nan

    return (
        float(location_data["latitude"].median()),
        float(location_data["longitude"].median()),
    )


def source_options_for_city(data, state, city):
    city_data = data[
        (data["state"] == state)
        & (data["cityname"] == city)
    ]

    if "source" not in city_data.columns:
        return []

    counts = (
        city_data["source"]
        .dropna()
        .astype(str)
        .value_counts()
    )

    return counts.index.tolist()


def city_market_summary(data, state, city):
    city_data = data[
        (data["state"] == state)
        & (data["cityname"] == city)
    ].dropna(
        subset=["price"]
    )

    if city_data.empty:
        return {
            "listings": 0,
            "median_rent": np.nan,
            "median_size": np.nan,
        }

    return {
        "listings": int(len(city_data)),
        "median_rent": float(
            city_data["price"].median()
        ),
        "median_size": float(
            city_data["square_feet"].median()
        ),
    }


# ------------------------------------------------------
# INITIALIZE
# ------------------------------------------------------

if not DATA_PATH.exists():
    st.error(
        f"Dataset not found: {DATA_PATH}"
    )
    st.stop()


data = load_data()


# ------------------------------------------------------
# HEADER
# ------------------------------------------------------

st.title("Rent Predictor")

st.markdown(
    """
    Estimate the monthly rent of a US apartment using one of the four
    trained models from the project.

    All models use the **same fitted preprocessing workflow**, so the same
    apartment can be evaluated with XGBoost, Random Forest, LASSO or Ridge.
    """
)

st.divider()


# ======================================================
# 1. MODEL SELECTION
# ======================================================

st.header("1. Select Prediction Model")

model_col1, model_col2 = st.columns([2, 3])


with model_col1:

    selected_model_name = st.selectbox(
        "Model",
        options=list(MODEL_CONFIG.keys()),
        index=0,
        help=(
            "Each option loads the corresponding trained pipeline "
            "saved from the project notebook."
        ),
    )


selected_model_config = MODEL_CONFIG[
    selected_model_name
]

model_path = find_model_path(
    selected_model_config["filename"]
)


if model_path is None:

    st.error(
        f"""
        `{selected_model_config["filename"]}` was not found.

        Place the trained model either in the project root or in
        the `models/` folder.
        """
    )

    st.stop()


try:

    model = load_model(model_path)


except Exception as error:

    st.error(
        f"{selected_model_name} could not be loaded."
    )

    st.exception(error)

    st.info(
        """
        Pickle files should be loaded with package versions compatible
        with the environment in which the model was trained. In particular,
        check the installed versions of scikit-learn and XGBoost.
        """
    )

    st.stop()


with model_col2:

    st.markdown(
        f"""
        **{selected_model_name}**

        {selected_model_config["description"]}
        """
    )

    score_col1, score_col2 = st.columns(2)

    with score_col1:

        st.metric(
            "Notebook CV R²",
            f"{selected_model_config['cv_r2']:.3f}",
        )

    with score_col2:

        st.metric(
            "Notebook Test R²",
            f"{selected_model_config['test_r2']:.3f}",
        )


st.caption(
    """
    The scores above are the model-comparison results on `log_price`
    from the project notebook. They are shown only as context and are not
    recalculated for the apartment entered below.
    """
)

st.divider()


# ======================================================
# 2. LOCATION
# ======================================================

st.header("2. Location")

location_col1, location_col2 = st.columns(2)


state_options = sorted(
    data["state"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


with location_col1:

    selected_state = st.selectbox(
        "State",
        options=state_options,
    )


city_options = sorted(
    data.loc[
        data["state"] == selected_state,
        "cityname",
    ]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


with location_col2:

    selected_city = st.selectbox(
        "City",
        options=city_options,
    )


latitude, longitude = city_coordinates(
    data,
    selected_state,
    selected_city,
)


if pd.isna(latitude) or pd.isna(longitude):

    st.warning(
        """
        No valid latitude/longitude information is available for the
        selected city. Choose another city.
        """
    )

    st.stop()


market_summary = city_market_summary(
    data,
    selected_state,
    selected_city,
)


location_info1, location_info2, location_info3 = st.columns(3)

with location_info1:

    st.metric(
        "Listings in dataset",
        f"{market_summary['listings']:,}",
    )


with location_info2:

    if pd.notna(market_summary["median_rent"]):

        st.metric(
            "City median rent",
            f"${market_summary['median_rent']:,.0f}",
        )


with location_info3:

    if pd.notna(market_summary["median_size"]):

        st.metric(
            "City median size",
            f"{market_summary['median_size']:,.0f} ft²",
        )


st.caption(
    """
    Latitude and longitude are inferred from the median coordinates of
    listings in the selected city and state.
    """
)

st.divider()


# ======================================================
# 3. APARTMENT CHARACTERISTICS
# ======================================================

st.header("3. Apartment Characteristics")

property_col1, property_col2 = st.columns(2)


with property_col1:

    default_size = (
        int(round(market_summary["median_size"]))
        if pd.notna(market_summary["median_size"])
        else 900
    )

    square_feet = st.number_input(
        "Apartment size (ft²)",
        min_value=100,
        max_value=10000,
        value=max(
            100,
            min(default_size, 10000),
        ),
        step=50,
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=10,
        value=2,
        step=1,
        help="0 can represent a studio apartment.",
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.5,
    )


with property_col2:

    max_amenities = (
        int(data["n_amenities"].max())
        if "n_amenities" in data.columns
        else 20
    )

    n_amenities = st.number_input(
        "Number of amenities",
        min_value=0,
        max_value=max(
            20,
            max_amenities,
        ),
        value=5,
        step=1,
        help=(
            "The model uses only the number of listed amenities, "
            "not their individual types."
        ),
    )

    pets_label = st.selectbox(
        "Pet policy",
        options=[
            "No pet type listed",
            "One pet type allowed",
            "Two pet types allowed",
        ],
        help=(
            "The reference model converts pets_allowed into the number "
            "of allowed pet types."
        ),
    )

    pets_mapping = {
        "No pet type listed": 0,
        "One pet type allowed": 1,
        "Two pet types allowed": 2,
    }

    pets_allowed_int = pets_mapping[
        pets_label
    ]

    has_photo = st.checkbox(
        "Listing has a photo or thumbnail",
        value=True,
    )

    has_photo_int = int(has_photo)


st.divider()


# ======================================================
# 4. LISTING SOURCE
# ======================================================

st.header("4. Listing Metadata")

st.markdown(
    """
    `source` was retained as a predictor in the reference models because
    different listing platforms can contain different mixes of apartments
    and prices.
    """
)


source_options = source_options_for_city(
    data,
    selected_state,
    selected_city,
)


if not source_options:

    source_options = sorted(
        data["source"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


selected_source = st.selectbox(
    "Listing source",
    options=source_options,
    help=(
        "Sources are ordered by frequency for the selected city when "
        "city-level observations are available."
    ),
)


st.divider()


# ======================================================
# 5. PREDICTION
# ======================================================

st.header("5. Predicted Monthly Rent")

st.markdown(
    f"""
    Click the button below to send the engineered apartment features through
    the complete fitted preprocessing pipeline and **{selected_model_name}**
    model.
    """
)


predict_button = st.button(
    f"Predict rent with {selected_model_name}",
    type="primary",
    use_container_width=True,
)


if predict_button:

    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------

    log_square_feet = float(
        np.log(square_feet)
    )

    latitude_sq = float(
        latitude ** 2
    )

    longitude_sq = float(
        longitude ** 2
    )


    model_input = pd.DataFrame(
        [
            {
                "bathrooms": float(bathrooms),
                "bedrooms": float(bedrooms),
                "log_square_feet": log_square_feet,
                "cityname": selected_city,
                "state": selected_state,
                "latitude": float(latitude),
                "longitude": float(longitude),
                "source": selected_source,
                "n_amenities": float(n_amenities),
                "pets_allowed_int": float(
                    pets_allowed_int
                ),
                "has_photo_int": float(
                    has_photo_int
                ),
                "latitude_sq": latitude_sq,
                "longitude_sq": longitude_sq,
            }
        ],
        columns=MODEL_FEATURES,
    )


    try:

        predicted_log_price = float(
            model.predict(model_input)[0]
        )

        predicted_rent = float(
            np.exp(predicted_log_price)
        )


    except Exception as error:

        st.error(
            f"Prediction with {selected_model_name} failed."
        )

        st.exception(error)

        st.info(
            """
            If the error is related to loading or transforming the fitted
            sklearn pipeline, verify that the deployment uses package
            versions compatible with the versions used to create the
            model pickle files.
            """
        )

        st.stop()


    # --------------------------------------------------
    # MAIN RESULT
    # --------------------------------------------------

    result_left, result_center, result_right = st.columns(
        [1, 2, 1]
    )

    with result_center:

        st.metric(
            f"Estimated monthly rent – {selected_model_name}",
            f"${predicted_rent:,.0f}",
        )


    # --------------------------------------------------
    # LOCAL COMPARISON
    # --------------------------------------------------

    city_median = market_summary["median_rent"]

    if pd.notna(city_median) and city_median > 0:

        difference = (
            predicted_rent - city_median
        )

        difference_pct = (
            difference / city_median * 100
        )


        comparison_col1, comparison_col2 = st.columns(2)

        with comparison_col1:

            st.metric(
                f"Dataset median – {selected_city}",
                f"${city_median:,.0f}",
            )


        with comparison_col2:

            st.metric(
                "Prediction vs city median",
                f"{difference_pct:+.1f}%",
                delta=f"${difference:+,.0f}",
            )


    st.caption(
        """
        The city median is shown only as descriptive context from the dataset.
        It is not a prediction interval or a measure of model confidence.
        """
    )


    # --------------------------------------------------
    # TECHNICAL DETAILS
    # --------------------------------------------------

    with st.expander(
        "View engineered model input"
    ):

        display_input = model_input.copy()

        display_input[
            "selected_model"
        ] = selected_model_name

        display_input[
            "predicted_log_price"
        ] = predicted_log_price

        display_input[
            "predicted_monthly_rent"
        ] = predicted_rent

        st.dataframe(
            display_input,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            f"""
            **Model**

            `{selected_model_name}`

            **Inferred geographic coordinates**

            - Latitude: `{latitude:.5f}`
            - Longitude: `{longitude:.5f}`

            **Target transformation**

            All four models predict `log_price`.

            `exp({predicted_log_price:.4f}) = ${predicted_rent:,.0f}`
            """
        )


st.divider()


# ======================================================
# MODEL NOTE
# ======================================================

st.subheader("About the Prediction")

st.markdown(
    f"""
    The predictor currently uses **{selected_model_name}**.

    All four deployed models use the same modeling target, `log_price`.
    The final result is therefore converted back to the original dollar scale
    using the exponential function.

    XGBoost achieved the strongest CV/test performance in the project
    notebook. Random Forest, Ridge and LASSO are included as alternative
    models so their predictions can also be explored interactively.

    These predictions should be interpreted as part of an educational
    data-science project rather than as a production rental appraisal.
    """
)

st.caption(
    f"Loaded model file: {model_path.name}"
)
