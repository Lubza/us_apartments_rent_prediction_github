from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    return data


@st.cache_data
def prepare_data(raw_data):
    """
    Reproduce the deterministic preprocessing steps from the reference notebook.
    Transformations that learn from data are intentionally left for the sklearn
    pipeline after the train/test split.
    """
    data = raw_data.copy()

    # amenities -> number of comma-separated items
    data["amenities_list"] = data["amenities"].str.split(",")
    data.loc[
        data["amenities_list"].isnull(),
        "amenities_list",
    ] = ""
    data["n_amenities"] = data["amenities_list"].apply(len)

    # pets_allowed -> number of allowed pet types
    data["pets_allowed_list"] = data["pets_allowed"].str.split(",")
    data.loc[
        data["pets_allowed_list"].isnull(),
        "pets_allowed_list",
    ] = ""
    data["pets_allowed_int"] = data["pets_allowed_list"].apply(len)

    # weekly prices -> approximate monthly basis
    weekly_mask = data["price_type"].isin(
        ["Monthly|Weekly", "Weekly"]
    )
    data.loc[weekly_mask, "price"] = (
        data.loc[weekly_mask, "price"] * 4
    )

    # has_photo -> binary
    data["has_photo_int"] = 0
    data.loc[
        data["has_photo"].isin(["Yes", "Thumbnail"]),
        "has_photo_int",
    ] = 1

    # logarithmic transformations
    data["log_square_feet"] = np.log(data["square_feet"])
    data["log_price"] = np.log(data["price"])

    # narrow modeling dataset
    data_narrow = data[
        [
            "log_price",
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
        ]
    ].copy()

    # geographic polynomial inputs
    data_narrow["latitude_sq"] = data_narrow["latitude"] ** 2
    data_narrow["longitude_sq"] = data_narrow["longitude"] ** 2

    return data, data_narrow


def histogram_figure(series, title, xlabel, bins=40):
    """Notebook-like single histogram using Matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    clean = series.dropna()
    ax.hist(clean, bins=bins, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def compact_dataframe(frame, height=None):
    dataframe_kwargs = {
        "width": "stretch",
        "hide_index": True,
    }

    if height is not None:
        dataframe_kwargs["height"] = height

    st.dataframe(
        frame,
        **dataframe_kwargs,
    )


raw_data = load_data()
data, data_narrow = prepare_data(raw_data)


# ------------------------------------------------------
# HEADER
# ------------------------------------------------------

st.title("Data Preparation")

st.markdown(
    """
    After exploratory data analysis, the raw variables are transformed into
    a form suitable for modeling.

    **Important principle:** transformations that *learn from the data* —
    such as imputation, Target Encoding or scaling — must not be fitted on
    the complete dataset before the train/test split. These operations are
    therefore placed later inside the machine-learning pipeline and are
    learned only from the training data.
    """
)

st.divider()


# ======================================================
# 1. COLUMN-LEVEL PREPROCESSING
# ======================================================

st.header("1. Preprocessing Individual Variables")

st.markdown(
    """
    The first steps are deterministic feature-engineering operations. They
    convert raw text-like or differently scaled variables into compact
    numerical representations without learning statistics from the complete
    dataset.
    """
)


# ------------------------------------------------------
# AMENITIES
# ------------------------------------------------------

st.subheader("`amenities`")

st.markdown(
    """
    `amenities` contains a comma-separated list of apartment features.

    In the reference solution, a deliberately simple representation is used:
    **the number of listed amenities** (`n_amenities`).

    **Advantage:** it is fast and adds almost no dimensionality.

    **Limitation:** the model no longer knows *which* amenities are present.
    A refrigerator, swimming pool, tennis court or hot tub may have very
    different effects on rent, but they all contribute only one count here.

    A more detailed alternative would be to create a separate 0/1
    multi-hot feature for every amenity type.
    """
)

st.code(
    """data['amenities_list'] = data['amenities'].str.split(',')
data.loc[data['amenities_list'].isnull(), 'amenities_list'] = ''
data['n_amenities'] = data['amenities_list'].apply(len)""",
    language="python",
)

amenities_preview = data[
    ["amenities", "n_amenities"]
].head(10)

compact_dataframe(amenities_preview)

fig = histogram_figure(
    data["n_amenities"],
    "Distribution of n_amenities",
    "Number of amenities",
    bins=max(10, int(data["n_amenities"].max()) + 1),
)
st.pyplot(fig, width="content")
plt.close(fig)

st.divider()


# ------------------------------------------------------
# PETS ALLOWED
# ------------------------------------------------------

st.subheader("`pets_allowed`")

st.markdown(
    """
    As with `amenities`, the text list is converted into a **count of allowed
    pet types**:

    - nothing listed → `0`
    - dog or cat → `1`
    - both → `2`

    This is a simple numerical representation. However, the difference
    between values 1 and 2 does not necessarily have a linear economic
    interpretation for rent.
    """
)

st.code(
    """data['pets_allowed_list'] = data['pets_allowed'].str.split(',')
data.loc[data['pets_allowed_list'].isnull(), 'pets_allowed_list'] = ''
data['pets_allowed_int'] = data['pets_allowed_list'].apply(len)""",
    language="python",
)

pets_col1, pets_col2 = st.columns(2)

with pets_col1:
    st.markdown("**Original values**")
    pets_original = (
        raw_data["pets_allowed"]
        .fillna("Missing")
        .value_counts()
        .rename_axis("pets_allowed")
        .reset_index(name="count")
    )
    compact_dataframe(pets_original)

with pets_col2:
    st.markdown("**Engineered values**")
    pets_engineered = (
        data["pets_allowed_int"]
        .value_counts()
        .sort_index()
        .rename_axis("pets_allowed_int")
        .reset_index(name="count")
    )
    compact_dataframe(pets_engineered)

st.divider()


# ------------------------------------------------------
# PRICE TYPE / PRICE
# ------------------------------------------------------

st.subheader("`price_type` / `price`")

st.markdown(
    """
    Before modeling, we check which price frequencies occur in the data.

    Weekly and monthly rents must be expressed on the **same time basis**.
    Otherwise, a weekly listing could appear to be an extremely cheap monthly
    apartment and distort both model error and $R^2$.

    The reference solution converts `Weekly` and `Monthly|Weekly` observations
    to an approximate monthly price using **price × 4**. There are only a few
    such records in this dataset, but the preprocessing should also remain
    sensible if the dataset is expanded later.
    """
)

price_type_col1, price_type_col2 = st.columns([1, 2])

with price_type_col1:
    st.markdown("**Price types**")
    price_type_counts = (
        raw_data["price_type"]
        .value_counts()
        .rename_axis("price_type")
        .reset_index(name="count")
    )
    compact_dataframe(price_type_counts)

with price_type_col2:
    st.markdown("**Non-monthly observations before conversion**")
    non_monthly = raw_data.loc[
        raw_data["price_type"].isin(
            ["Weekly", "Monthly|Weekly"]
        ),
        ["price_type", "price"],
    ].copy()

    non_monthly["monthly_price_used"] = (
        non_monthly["price"] * 4
    )

    compact_dataframe(non_monthly)

st.code(
    """data.loc[
    data['price_type'].isin(['Monthly|Weekly', 'Weekly']),
    'price'
] = data['price'] * 4""",
    language="python",
)

st.divider()


# ------------------------------------------------------
# HAS PHOTO
# ------------------------------------------------------

st.subheader("`has_photo`")

st.markdown(
    """
    The original variable contains `Yes`, `No` and `Thumbnail`.

    The reference solution converts it to binary information:

    - `Yes` or `Thumbnail` → `1`
    - otherwise → `0`

    The idea is that a photograph — or at least a thumbnail — may be related
    to the quality of presentation or marketing of the listing and therefore
    indirectly to rent.
    """
)

photo_col1, photo_col2 = st.columns(2)

with photo_col1:
    photo_original = (
        raw_data["has_photo"]
        .value_counts()
        .rename_axis("has_photo")
        .reset_index(name="count")
    )
    compact_dataframe(photo_original)

with photo_col2:
    photo_binary = (
        data["has_photo_int"]
        .value_counts()
        .sort_index()
        .rename_axis("has_photo_int")
        .reset_index(name="count")
    )
    compact_dataframe(photo_binary)

st.code(
    """data['has_photo_int'] = 0
data.loc[
    data.has_photo.isin(['Yes', 'Thumbnail']),
    'has_photo_int'
] = 1""",
    language="python",
)

st.divider()


# ======================================================
# 2. LOG TRANSFORMATION
# ======================================================

st.header("2. Log Transformation")

st.markdown(
    """
    ### Why log-transform `square_feet`?

    Apartment size has a typically right-skewed distribution: most apartments
    are within a common size range, while a small number of properties are
    extremely large. Taking the logarithm compresses these extreme values and
    usually produces a more symmetric distribution.

    ### Why is `price` more complicated?

    `price` is the target variable. Modeling `log_price` can be useful, but the
    result must be interpreted carefully:

    - the model is trained on `log_price`,
    - to evaluate errors in actual dollars, predictions must be converted back
      using the exponential function,
    - $R^2$ on `log_price` is **not the same metric** as $R^2$ on the original
      rent scale.

    This distinction is revisited on the model-evaluation page.
    """
)

st.markdown("#### Before log transformation")

before_col1, before_col2 = st.columns(2)

with before_col1:
    fig = histogram_figure(
        raw_data["square_feet"],
        "square_feet",
        "Square feet",
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with before_col2:
    fig = histogram_figure(
        data["price"],
        "price",
        "Monthly rent ($)",
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)

st.code(
    """data['log_square_feet'] = np.log(data['square_feet'])
data['log_price'] = np.log(data['price'])""",
    language="python",
)

st.markdown("#### After log transformation")

after_col1, after_col2 = st.columns(2)

with after_col1:
    fig = histogram_figure(
        data["log_square_feet"],
        "log_square_feet",
        "log(Square feet)",
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with after_col2:
    fig = histogram_figure(
        data["log_price"],
        "log_price",
        "log(Monthly rent)",
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)

st.divider()


# ======================================================
# 3. CHECKING REMAINING COLUMNS
# ======================================================

st.header("3. Checking the Remaining Variables")

st.markdown(
    """
    `value_counts()` is used as a quick diagnostic for categorical variables.
    It helps identify:

    - constant or almost constant columns,
    - very rare categories,
    - duplicated information,
    - variables that may or may not be useful for modeling.

    In this solution:

    - `currency` is constant → excluded,
    - `fee` is constant → excluded,
    - `category` is almost constant → excluded,
    - `price_display` duplicates the target price information → excluded,
    - `source` is retained because different listing websites may contain
      systematically different mixes of properties and prices.
    """
)

check_col1, check_col2, check_col3 = st.columns(3)

with check_col1:
    st.markdown("**category**")
    category_counts = (
        raw_data["category"]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="count")
    )
    compact_dataframe(category_counts)

with check_col2:
    st.markdown("**currency**")
    currency_counts = (
        raw_data["currency"]
        .value_counts()
        .rename_axis("currency")
        .reset_index(name="count")
    )
    compact_dataframe(currency_counts)

with check_col3:
    st.markdown("**fee**")
    fee_counts = (
        raw_data["fee"]
        .value_counts()
        .rename_axis("fee")
        .reset_index(name="count")
    )
    compact_dataframe(fee_counts)

st.markdown("**price vs price_display**")
compact_dataframe(
    raw_data[["price", "price_display"]].head()
)

st.markdown("**source**")
source_counts = (
    raw_data["source"]
    .value_counts()
    .rename_axis("source")
    .reset_index(name="count")
)
compact_dataframe(source_counts)

st.divider()


# ======================================================
# 4. NARROWING THE DATASET
# ======================================================

st.header("4. Narrowing the Dataset to Relevant Columns")

st.markdown(
    """
    We now create `data_narrow`, which contains only the variables that will
    continue into modeling.

    The purpose is to:

    - remove constant or nearly constant information,
    - remove identifiers and duplicated information,
    - exclude long text fields that are not processed with NLP in this
      reference solution,
    - retain `source`, because the listing website may contain useful pricing
      information.

    This also makes the following preprocessing pipeline easier to understand.
    """
)

selection_table = pd.DataFrame(
    {
        "Variable": [
            "fee",
            "currency",
            "category",
            "title",
            "body",
            "id",
            "price_display",
            "time",
            "source",
        ],
        "Decision": [
            "Exclude",
            "Exclude",
            "Exclude",
            "Exclude",
            "Exclude",
            "Exclude",
            "Exclude",
            "Exclude",
            "Keep",
        ],
        "Reason": [
            "Practically no variability",
            "All prices are in USD",
            "Almost all observations belong to the same category",
            "Long text; NLP is not used in this solution",
            "Long text; NLP is not used in this solution",
            "Identifier, not an apartment characteristic",
            "Duplicates target price information",
            "Not used as a time feature in this solution",
            "Listing source may contain predictive price information",
        ],
    }
)

compact_dataframe(selection_table)

st.code(
    """data_narrow = data[[
    'log_price', 'bathrooms', 'bedrooms', 'log_square_feet',
    'cityname', 'state', 'latitude', 'longitude', 'source',
    'n_amenities', 'pets_allowed_int', 'has_photo_int'
]].copy()""",
    language="python",
)

st.markdown("**Modeling dataset preview**")
compact_dataframe(data_narrow.head(10))

st.divider()


# ======================================================
# 5. MISSING VALUES
# ======================================================

st.header("5. Missing Values")

st.markdown(
    """
    At this stage we only **identify** missing values and zeros.

    > **Key project rule:** imputation is not performed on the complete
    > dataset before the train/test split.

    The imputer is fitted later inside the pipeline using only the training
    data. Otherwise, information from the test set could indirectly influence
    the imputed values — a form of **data leakage**.

    A zero is not automatically an error. For example, `bedrooms = 0` can
    represent a studio apartment. In a production project, the meaning of
    suspicious zero values should ideally be checked against the source
    websites.
    """
)

missing_counts = (
    data_narrow
    .isna()
    .sum()
    .rename("missing")
    .to_frame()
)

zero_counts = (
    (data_narrow == 0)
    .sum()
    .rename("zeros")
    .to_frame()
)

quality_check = (
    missing_counts
    .join(zero_counts)
    .reset_index()
    .rename(columns={"index": "variable"})
)

compact_dataframe(quality_check)

st.divider()


# ======================================================
# 6. GEOGRAPHIC POLYNOMIAL INPUTS
# ======================================================

st.header("6. Squared Latitude and Longitude")

st.markdown(
    """
    Geographic EDA suggests that rent is not a linear function of location:
    some regions — for example parts of the coasts — have systematically
    higher prices.

    We therefore add:

    - `latitude_sq = latitude²`
    - `longitude_sq = longitude²`

    Later, `PolynomialFeatures` creates additional interactions. Because the
    squared geographic variables already exist, these interactions can also
    represent higher-order relationships (including combinations corresponding
    to third- and fourth-order terms), giving the model more flexibility to
    capture curved geographic price patterns.

    These features could also be created inside the pipeline; they are created
    beforehand here mainly to keep column handling simpler.
    """
)

st.code(
    """data_narrow['latitude_sq'] = data_narrow['latitude'] ** 2
data_narrow['longitude_sq'] = data_narrow['longitude'] ** 2""",
    language="python",
)

geo_preview = data_narrow[
    [
        "latitude",
        "latitude_sq",
        "longitude",
        "longitude_sq",
    ]
].head(10)

compact_dataframe(geo_preview)

st.divider()


# ======================================================
# 7. TRAIN / TEST SPLIT
# ======================================================

st.header("7. Train / Test Split")

st.markdown(
    """
    The data is split into **75% training data and 25% test data**
    (`test_size=0.25`, `random_state=42`).

    The test set represents future, unseen observations. Imputation, encoding
    and scaling must therefore **not** be fitted on the test data. These steps
    remain inside the pipeline and learn only from the training portion.
    """
)

n_total = len(data_narrow)
n_test = int(np.ceil(n_total * 0.25))
n_train = n_total - n_test

split_table = pd.DataFrame(
    {
        "Dataset": ["Training set", "Test set"],
        "Share": ["75%", "25%"],
        "Rows": [n_train, n_test],
    }
)

compact_dataframe(split_table)

st.code(
    """X = data_narrow.drop(columns='log_price')
y = data_narrow.log_price

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)""",
    language="python",
)

st.divider()


# ======================================================
# 8. TRANSFORMATION PIPELINE
# ======================================================

st.header("8. Transformation Pipeline")

st.markdown(
    """
    All learned transformations are wrapped in a pipeline so that each step is
    fitted only on training observations — and, during cross-validation, only
    on the training fold of that particular split.

    **Order of transformations**

    1. **Categorical encoding**
       - `cityname`, `state` and `source` receive both Target Encoding and a
         limited One-Hot Encoding.
       - `max_categories` limits the number of OHE columns; less frequent
         categories are grouped into `infrequent`.
       - `max_categories` is itself treated as a tunable hyperparameter.

    2. **KNN imputation**
       - takes place after encoding because `KNNImputer` works with numerical
         values,
       - both the number of neighbors and the weighting method are later tuned.

    3. **Polynomial / interaction features**
       - created only for selected variables to avoid uncontrolled growth in
         dimensionality.

    4. **RobustScaler**
       - reduces the influence of extreme values,
       - scaling is especially important for regularized linear models such as
         LASSO and Ridge.

    `verbose_feature_names_out=False` keeps transformed feature names shorter
    and easier to use later.
    """
)

st.markdown("#### Columns used by the pipeline")

pipeline_columns = pd.DataFrame(
    {
        "Object": [
            "cols_ohe",
            "cols_nan",
            "cols_interactions",
        ],
        "Columns": [
            "cityname, state",
            "bathrooms, bedrooms, latitude, latitude_sq, longitude, longitude_sq, cityname, state",
            (
                "n_amenities, bathrooms, bedrooms, pets_allowed_int, "
                "log_square_feet, cityname, state, source, latitude, "
                "longitude, latitude_sq, longitude_sq"
            ),
        ],
    }
)

compact_dataframe(pipeline_columns)

st.markdown("#### 8.1 Encoding")

st.code(
    """encoder = make_column_transformer(
    (TargetEncoder(target_type='continuous'), ['cityname', 'state']),
    (OneHotEncoder(
        max_categories=5,
        handle_unknown='infrequent_if_exist',
        sparse_output=False
    ), ['cityname', 'state']),

    (TargetEncoder(target_type='continuous'), ['source']),
    (OneHotEncoder(
        max_categories=3,
        handle_unknown='infrequent_if_exist',
        sparse_output=False
    ), ['source']),

    remainder='passthrough',
    verbose_feature_names_out=False
).set_output(transform='pandas')""",
    language="python",
)

st.markdown(
    """
    Target Encoding captures the relationship between a category and the target,
    while limited One-Hot Encoding preserves explicit information about the most
    frequent categories. Using both gives the model two different views of the
    same categorical information.
    """
)

st.markdown("#### 8.2 KNN Imputation")

st.code(
    """imputer = make_column_transformer(
    (KNNImputer(
        n_neighbors=5,
        weights='distance',
        add_indicator=False
    ), cols_nan),

    remainder='passthrough',
    verbose_feature_names_out=False
).set_output(transform='pandas')""",
    language="python",
)

st.markdown(
    """
    KNN imputation is performed **after the train/test split and inside the
    pipeline**, so the test data does not influence the missing-value estimates.
    `n_neighbors` and `weights` are later included in hyperparameter tuning.
    """
)

st.markdown("#### 8.3 Interaction Features")

st.code(
    """interactions = make_column_transformer(
    (PolynomialFeatures(
        degree=2,
        interaction_only=True,
        include_bias=False
    ), cols_interactions),

    remainder='passthrough',
    verbose_feature_names_out=False
).set_output(transform='pandas')""",
    language="python",
)

st.markdown(
    """
    `interaction_only=True` does not directly create every squared term.
    However, because `latitude_sq` and `longitude_sq` are already present,
    their interactions can represent more complex higher-order geographic
    relationships.
    """
)

st.markdown("#### 8.4 Complete Transformation Pipeline")

st.code(
    """pipe_transform = make_pipeline(
    encoder,
    imputer,
    interactions,
    RobustScaler().set_output(transform='pandas')
)""",
    language="python",
)

st.markdown(
    """
    During cross-validation, fitting of **every step** is repeated separately
    on the training fold. This is the main reason for keeping preprocessing
    and the model in one sklearn pipeline: it prevents leakage and allows
    preprocessing hyperparameters and model hyperparameters to be optimized
    together.
    """
)

st.divider()

st.caption(
    "This page follows the structure, transformations and explanatory comments "
    "of the reference notebook `projekt_reseni_EDA_komentare.ipynb`, translated "
    "and adapted for the Streamlit application."
)
