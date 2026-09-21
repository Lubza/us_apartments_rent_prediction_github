from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------------------------------
# PATHS
# ------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "figures"

KDE_IMAGE_PATH = (FIGURES_DIR / "log_price_distribution_EN.png")

SCATTER_IMAGE_PATH = (FIGURES_DIR / "predicted_vs_actual_EN.png")


# ------------------------------------------------------
# NOTEBOOK RESULTS
# ------------------------------------------------------
# This page intentionally reproduces results stored in the reference notebook.
# It does NOT rerun RandomizedSearchCV every time the Streamlit app is opened.

MODEL_RESULTS = pd.DataFrame(
    {
        "Model": [
            "Random Forest",
            "XGBoost",
            "LASSO",
            "Ridge",
        ],
        "CV R²": [
            0.751,
            0.766,
            0.679,
            0.696,
        ],
        "Train R²": [
            0.895,
            0.861,
            0.735,
            0.747,
        ],
        "Test R²": [
            0.752,
            0.768,
            0.680,
            0.693,
        ],
    }
)

SHUFFLE_SPLIT_SCORES = np.array(
    [
        0.74534479,
        0.76303870,
        0.74662850,
        0.74514441,
        0.74178695,
    ]
)

XGB_LOG_PRICE_R2 = 0.7677780039966096
XGB_PRICE_R2 = 0.3693013183635344
XGB_PRICE_TRIMMED_R2 = 0.7391256722125452
PCT98_LOG_PRICE = 8.292798858200374
PCT98_PRICE = float(np.exp(PCT98_LOG_PRICE))


FEATURE_IMPORTANCE = pd.DataFrame(
    {
        "Feature": [
            "log_square_feet",
            "state",
            "cityname",
            "bathrooms",
            "longitude",
            "longitude_sq",
            "bedrooms",
            "latitude",
            "latitude_sq",
            "n_amenities",
            "source",
            "pets_allowed_int",
            "has_photo_int",
        ],
        "Importance": [
            0.330337,
            0.245732,
            0.152923,
            0.042295,
            0.036078,
            0.032815,
            0.029325,
            0.025808,
            0.023116,
            0.008316,
            0.006910,
            0.005427,
            0.000071,
        ],
        "Std": [
            0.011967,
            0.006413,
            0.004132,
            0.002810,
            0.002684,
            0.002801,
            0.001260,
            0.002241,
            0.001650,
            0.000913,
            0.001227,
            0.001479,
            0.000046,
        ],
    }
)




# ------------------------------------------------------
# HELPERS
# ------------------------------------------------------

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


def centered_pyplot(fig):
    """Display a Matplotlib figure smaller and centered on the page."""
    left, center, right = st.columns([1, 2.2, 1])

    with center:
        st.pyplot(
            fig,
            clear_figure=True,
        )


def show_model_image(image_path, caption):
    """Load the latest image bytes from the figures folder on every rerun."""
    if not image_path.exists():
        st.warning(
            f"Image not found: {image_path}"
        )
        return

    # Read the current file contents instead of passing only the file path.
    # This prevents Streamlit/browser caching from keeping an older image
    # after the notebook overwrites the PNG.
    image_bytes = image_path.read_bytes()

    left, center, right = st.columns([1, 2.2, 1])

    with center:
        st.image(
            image_bytes,
            caption=caption,
            width=590,
        )


def grouped_score_figure(results):
    fig, ax = plt.subplots(figsize=(8, 4.6))

    x = np.arange(len(results))
    width = 0.24

    ax.bar(
        x - width,
        results["CV R²"],
        width,
        label="CV",
    )

    ax.bar(
        x,
        results["Train R²"],
        width,
        label="Train",
    )

    ax.bar(
        x + width,
        results["Test R²"],
        width,
        label="Test",
    )

    for offset, column in [
        (-width, "CV R²"),
        (0, "Train R²"),
        (width, "Test R²"),
    ]:
        for idx, value in enumerate(results[column]):
            ax.text(
                idx + offset,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(results["Model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("R²")
    ax.set_title("Model comparison on log_price")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def shuffle_split_figure(scores):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    split_numbers = np.arange(1, len(scores) + 1)

    ax.plot(
        split_numbers,
        scores,
        marker="o",
    )

    ax.axhline(
        scores.mean(),
        linestyle="--",
        label=f"Mean = {scores.mean():.3f}",
    )

    ax.set_xticks(split_numbers)
    ax.set_xlabel("ShuffleSplit iteration")
    ax.set_ylabel("R²")
    ax.set_title("XGBoost – ShuffleSplit validation")
    ax.set_ylim(0.70, 0.80)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def feature_importance_figure(results):
    top_features = (
        results
        .head(10)
        .sort_values("Importance")
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        top_features["Feature"],
        top_features["Importance"],
        xerr=top_features["Std"],
        capsize=3,
    )

    ax.set_xlabel("Decrease in R² after permutation")
    ax.set_ylabel("Feature")
    ax.set_title("XGBoost – Permutation Feature Importance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def price_scale_figure():
    labels = [
        "log_price",
        "price",
        "price\nwithout top 2%",
    ]

    values = [
        XGB_LOG_PRICE_R2,
        XGB_PRICE_R2,
        XGB_PRICE_TRIMMED_R2,
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    bars = ax.bar(labels, values)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    ax.set_ylim(0, 1)
    ax.set_ylabel("R²")
    ax.set_title("Effect of evaluation scale and extreme rents")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig



# ------------------------------------------------------
# SIDEBAR SECTION NAVIGATION
# ------------------------------------------------------

st.sidebar.markdown(
"""<div style="
margin-top: 0.35rem;
margin-left: 0.6rem;
padding-left: 0.75rem;
border-left: 2px solid rgba(128, 128, 128, 0.25);
line-height: 1.85;
font-size: 0.88rem;
">
<div style="
font-weight: 600;
margin-bottom: 0.25rem;
opacity: 0.85;
">
Model Development sections
</div>
<a href="#modeling-approach" target="_self" style="text-decoration:none;">1. Modeling Approach</a><br>
<a href="#random-forest" target="_self" style="text-decoration:none;">2. Random Forest</a><br>
<a href="#xgboost" target="_self" style="text-decoration:none;">3. XGBoost</a><br>
<a href="#lasso" target="_self" style="text-decoration:none;">4. LASSO</a><br>
<a href="#ridge" target="_self" style="text-decoration:none;">5. Ridge</a><br>
<a href="#model-comparison" target="_self" style="text-decoration:none;">6. Model Comparison</a><br>
<a href="#shufflesplit" target="_self" style="text-decoration:none;">7. ShuffleSplit</a><br>
<a href="#feature-importance" target="_self" style="text-decoration:none;">8. Feature Importance</a><br>
<a href="#actual-rent" target="_self" style="text-decoration:none;">9. Actual Rent</a><br>
<a href="#outliers" target="_self" style="text-decoration:none;">10. Expensive Outliers</a><br>
<a href="#model-persistence" target="_self" style="text-decoration:none;">11. Saving Models</a><br>
<a href="#final-interpretation" target="_self" style="text-decoration:none;">12. Final Interpretation</a>
</div>""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------
# HEADER
# ------------------------------------------------------

st.title("Model Development & Evaluation")

st.markdown(
    """
    This page follows the modeling and evaluation section of the reference
    notebook.

    The target is **`log_price`**, and four regression approaches are
    compared:

    1. **Random Forest**
    2. **XGBoost**
    3. **LASSO**
    4. **Ridge**

    Hyperparameters are selected with cross-validation, after which the best
    model is checked on the held-out test set and finally evaluated again on
    the original rent scale in US dollars.
    """
)

st.divider()


# ======================================================
# 1. MODELING APPROACH
# ======================================================

st.markdown('<div id="modeling-approach"></div>', unsafe_allow_html=True)
st.header("1. Modeling Approach")

st.markdown(
    """
    The preprocessing pipeline from the previous page is combined directly
    with each estimator.

    This is important because the hyperparameter search then evaluates the
    **entire workflow** — preprocessing plus model — inside each
    cross-validation fold.

    As a result, parameters of preprocessing steps can be tuned in exactly
    the same way as model parameters.
    """
)

st.subheader("Helper Function for Hyperparameter Search")

st.markdown(
    """
    The notebook uses a helper function named
    `search_grid_and_print_score()`. Despite the name, the implementation uses
    **`RandomizedSearchCV`**, not `GridSearchCV`.

    Rather than evaluating every possible combination, it samples a fixed
    number of parameter combinations (`n_iter=30`). This is faster when the
    search space contains many hyperparameters.
    """
)

st.code(
    """search = RandomizedSearchCV(
    estimator=estimator,
    param_distributions=param_distributions,
    n_iter=n_iter,
    scoring='r2',
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=3
)

search.fit(X_train, y_train)""",
    language="python",
)

st.markdown(
    """
    For each model, the notebook reports:

    - **Best CV score** — the mean cross-validation R² of the selected
      hyperparameter combination,
    - **Train score** — R² after fitting the selected model on the full
      training set,
    - **Test score** — R² on the held-out test set.

    Cross-validation is the main criterion for choosing hyperparameters.
    The test set is reserved as the final generalization check.
    """
)

st.divider()


# ======================================================
# 2. RANDOM FOREST
# ======================================================

st.markdown('<div id="random-forest"></div>', unsafe_allow_html=True)
st.header("2. Random Forest")

st.markdown(
    """
    Random Forest combines many decision trees and aggregates their
    predictions. It can naturally capture nonlinear relationships and
    interactions between variables.

    The search does not tune only the Random Forest itself. It also tunes
    parameters of the earlier preprocessing pipeline, such as the number of
    One-Hot categories and the KNN imputation settings.

    This demonstrates an important property of sklearn pipelines:
    **almost any parameter inside the complete workflow can become a
    hyperparameter.**
    """
)

st.code(
    """pipe_model_rf = make_pipeline(
    pipe_transform,
    RandomForestRegressor(random_state=42)
)""",
    language="python",
)

st.subheader("Random Forest Search Space")

rf_search = pd.DataFrame(
    {
        "Parameter": [
            "OHE max_categories",
            "KNN n_neighbors",
            "KNN weights",
            "max_features",
            "min_samples_leaf",
            "max_depth",
            "min_samples_split",
            "n_estimators",
        ],
        "Values Tested": [
            "3, 5, 8",
            "5, 10, 20",
            "distance, uniform",
            "0.5, 0.6, 0.7",
            "1, 5, 8",
            "None, 5, 10, 18",
            "2, 5, 9",
            "80, 110, 140",
        ],
    }
)

compact_dataframe(rf_search)

st.subheader("Best Random Forest Parameters")

rf_best = pd.DataFrame(
    {
        "Parameter": [
            "n_estimators",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "max_depth",
            "KNN weights",
            "KNN n_neighbors",
            "OHE max_categories",
        ],
        "Best Value": [
            "80",
            "5",
            "1",
            "0.5",
            "None",
            "uniform",
            "10",
            "5",
        ],
    }
)

compact_dataframe(rf_best)

rf_scores = pd.DataFrame(
    {
        "Metric": [
            "Best CV R²",
            "Train R²",
            "Test R²",
        ],
        "Score": [
            0.751,
            0.895,
            0.752,
        ],
    }
)

compact_dataframe(rf_scores)

st.markdown(
    """
    The training score is clearly higher than both CV and test scores.
    That gap suggests that Random Forest fits the training data more closely
    and shows **more overfitting** than XGBoost in this experiment.

    The test score nevertheless remains very close to the CV score, which is
    a useful sign that the cross-validation estimate was realistic.
    """
)

st.divider()


# ======================================================
# 3. XGBOOST
# ======================================================

st.markdown('<div id="xgboost"></div>', unsafe_allow_html=True)
st.header("3. XGBoost")

st.markdown(
    """
    XGBoost is a gradient-boosting model. Trees are created **sequentially**,
    and each new tree attempts to improve the errors remaining from the
    previous trees.

    As with Random Forest, the search jointly optimizes preprocessing and
    model parameters.

    The objective is not to maximize the training score, but to find the
    combination with the strongest **cross-validation R²**.
    """
)

st.code(
    """pipe_model_xgb = make_pipeline(
    pipe_transform,
    XGBRegressor(random_state=42)
)""",
    language="python",
)

st.subheader("XGBoost Search Space")

xgb_search = pd.DataFrame(
    {
        "Parameter": [
            "OHE max_categories",
            "KNN n_neighbors",
            "KNN weights",
            "learning_rate",
            "colsample_bytree",
            "max_depth",
            "subsample",
            "n_estimators",
        ],
        "Values Tested": [
            "3, 5, 8",
            "5, 10, 20",
            "distance, uniform",
            "0.05, 0.10, 0.20",
            "0.3, 0.5, 0.7",
            "None, 5, 10, 15",
            "0.3, 0.5, 0.7",
            "100, 150, 200",
        ],
    }
)

compact_dataframe(xgb_search)

st.markdown(
    """
    **Interpretation of the main XGBoost hyperparameters**

    - `learning_rate` — contribution of each new boosting tree,
    - `n_estimators` — number of trees,
    - `max_depth` — maximum tree complexity,
    - `subsample` — fraction of observations available to each tree,
    - `colsample_bytree` — fraction of features available to each tree.
    """
)

st.subheader("Best XGBoost Parameters")

xgb_best = pd.DataFrame(
    {
        "Parameter": [
            "subsample",
            "n_estimators",
            "max_depth",
            "learning_rate",
            "colsample_bytree",
            "KNN weights",
            "KNN n_neighbors",
            "OHE max_categories",
        ],
        "Best Value": [
            "0.5",
            "200",
            "None",
            "0.05",
            "0.3",
            "distance",
            "20",
            "8",
        ],
    }
)

compact_dataframe(xgb_best)

xgb_scores = pd.DataFrame(
    {
        "Metric": [
            "Best CV R²",
            "Train R²",
            "Test R²",
        ],
        "Score": [
            0.766,
            0.861,
            0.768,
        ],
    }
)

compact_dataframe(xgb_scores)

st.markdown(
    """
    XGBoost has the highest CV and test scores of the four compared models.
    Its training score is also lower than Random Forest's, while its test
    score is slightly higher.

    That pattern indicates a **better balance between fit and
    generalization** in this experiment.
    """
)

st.divider()


# ======================================================
# 4. LASSO
# ======================================================

st.markdown('<div id="lasso"></div>', unsafe_allow_html=True)
st.header("4. LASSO – Reference Linear Model")

st.markdown(
    """
    LASSO is a linear regression model with **L1 regularization**.

    It serves as a simpler reference model against the two tree-based ensemble
    methods.

    L1 regularization can shrink some coefficients all the way to zero,
    meaning that it also performs a form of feature selection. Because the
    penalty depends on coefficient magnitude, input scaling is particularly
    important; this is why the preprocessing pipeline contains
    `RobustScaler`.
    """
)

st.code(
    """pipe_model_lasso = make_pipeline(
    pipe_transform,
    Lasso()
)""",
    language="python",
)

st.subheader("LASSO Search Space")

lasso_search = pd.DataFrame(
    {
        "Parameter": [
            "OHE max_categories",
            "KNN n_neighbors",
            "KNN weights",
            "alpha",
        ],
        "Values Tested": [
            "3, 5, 8",
            "5, 10, 20",
            "distance, uniform",
            "0.005, 0.01, 0.05, 0.10",
        ],
    }
)

compact_dataframe(lasso_search)

st.markdown(
    """
    LASSO has fewer model-specific hyperparameters. The main one is `alpha`,
    which controls the strength of L1 regularization.

    Preprocessing is still tuned jointly, so the simpler model receives the
    same fair optimization framework as Random Forest and XGBoost.
    """
)

st.subheader("Best LASSO Parameters")

lasso_best = pd.DataFrame(
    {
        "Parameter": [
            "KNN weights",
            "KNN n_neighbors",
            "OHE max_categories",
            "alpha",
        ],
        "Best Value": [
            "distance",
            "20",
            "8",
            "0.005",
        ],
    }
)

compact_dataframe(lasso_best)

lasso_scores = pd.DataFrame(
    {
        "Metric": [
            "Best CV R²",
            "Train R²",
            "Test R²",
        ],
        "Score": [
            0.679,
            0.735,
            0.680,
        ],
    }
)

compact_dataframe(lasso_scores)

st.markdown(
    """
    LASSO performs materially worse than the two tree-based models. This
    suggests that apartment rent contains important **nonlinearities and
    interactions** that are difficult to capture with a regularized linear
    specification alone.
    """
)

st.divider()


# ======================================================
# 5. RIDGE
# ======================================================

st.markdown('<div id="ridge"></div>', unsafe_allow_html=True)
st.header("5. Ridge – L2 Regularized Linear Model")

st.markdown(
    """
    Ridge is a linear regression model with **L2 regularization**. Like LASSO,
    it serves as a simpler reference model against the two tree-based ensemble
    methods.

    L2 regularization penalizes large coefficient values and shrinks them
    toward zero, but unlike LASSO, it usually does **not** reduce coefficients
    exactly to zero. Ridge therefore typically keeps all features in the model
    while reducing their individual influence.

    Ridge can be particularly useful when several predictors are correlated,
    because the model can distribute their contribution across multiple
    related variables instead of selecting only some of them. As with LASSO,
    scaling is important because the regularization penalty depends on
    coefficient magnitude; here this is handled by `RobustScaler`.
    """
)

st.code(
    """pipe_model_ridge = make_pipeline(
    pipe_transform,
    Ridge()
)""",
    language="python",
)

st.subheader("Ridge Search Space")

ridge_search = pd.DataFrame(
    {
        "Parameter": [
            "OHE max_categories",
            "KNN n_neighbors",
            "KNN weights",
            "alpha",
        ],
        "Values Tested": [
            "3, 5, 8",
            "5, 10, 20",
            "distance, uniform",
            "0.01, 0.1, 1, 10, 100",
        ],
    }
)

compact_dataframe(ridge_search)

st.markdown(
    """
    Ridge has fewer model-specific hyperparameters; the main one is `alpha`,
    which controls the strength of **L2 regularization**.

    A higher `alpha` applies stronger regularization and shrinks the model
    coefficients more strongly toward zero, while a lower `alpha` allows the
    model to fit the training data more freely.

    Preprocessing is still tuned at the same time — including the number of
    OHE categories and KNN imputation settings — so Ridge is evaluated within
    the same optimized preprocessing framework as the other models.
    """
)

st.subheader("Best Ridge Parameters")

ridge_best = pd.DataFrame(
    {
        "Parameter": [
            "KNN weights",
            "KNN n_neighbors",
            "OHE max_categories",
            "alpha",
        ],
        "Best Value": [
            "distance",
            "20",
            "8",
            "0.1",
        ],
    }
)

compact_dataframe(ridge_best)

ridge_scores = pd.DataFrame(
    {
        "Metric": [
            "Best CV R²",
            "Train R²",
            "Test R²",
        ],
        "Score": [
            0.696,
            0.747,
            0.693,
        ],
    }
)

compact_dataframe(ridge_scores)

st.subheader("Ridge vs. LASSO")

st.markdown(
    """
    Both models are regularized linear regressions, but they handle
    coefficients differently:

    - **LASSO (L1)** can shrink some coefficients exactly to zero and therefore
      performs a form of feature selection.
    - **Ridge (L2)** shrinks coefficients toward zero while usually retaining
      all features, which can be helpful when predictors contain overlapping
      or correlated information.

    In this project, Ridge performs better than LASSO on both cross-validation
    and the held-out test set:

    - Ridge: CV R² = **0.696**, Test R² = **0.693**
    - LASSO: CV R² = **0.679**, Test R² = **0.680**

    The improvement is modest, and both linear models remain below the two
    tree-based ensemble methods. This suggests that retaining correlated
    information helps the linear benchmark, while important nonlinear
    relationships still remain.
    """
)

st.divider()


# ======================================================
# 6. MODEL COMPARISON
# ======================================================

st.markdown('<div id="model-comparison"></div>', unsafe_allow_html=True)
st.header("6. Model Comparison")

st.markdown(
    """
    The models are first compared on **`log_price`**, because this is the target
    on which they were trained.

    A large difference between train and test scores can indicate overfitting.
    For hyperparameter selection, however, the most important number is the
    cross-validation score; the held-out test score is used only as the final
    check of generalization.
    """
)

compact_dataframe(MODEL_RESULTS)

fig = grouped_score_figure(MODEL_RESULTS)
centered_pyplot(fig)
plt.close(fig)

st.markdown(
    """
    **XGBoost is selected as the best model in the notebook.**

    - XGBoost: CV R² = **0.766**, Test R² = **0.768**
    - Random Forest: CV R² = **0.751**, Test R² = **0.752**
    - Ridge: CV R² = **0.696**, Test R² = **0.693**
    - LASSO: CV R² = **0.679**, Test R² = **0.680**

    Random Forest obtains the highest training score, but that does not make it
    the strongest model. Its larger train–validation gap is evidence that the
    extra fit does not generalize as well.
    """
)

st.caption(
    "Notebook note: one later print statement labels the 0.768 test score as "
    "'random forest', but the code immediately above defines `best_model` as "
    "`model_xgb.best_estimator_`. The numeric score therefore belongs to the "
    "selected XGBoost model."
)

st.divider()


# ======================================================
# 7. SHUFFLESPLIT VALIDATION
# ======================================================

st.markdown('<div id="shufflesplit"></div>', unsafe_allow_html=True)
st.header("7. Additional Validation with ShuffleSplit")

st.markdown(
    """
    The selected XGBoost estimator is checked once more using
    **`ShuffleSplit`**.

    Unlike ordinary K-Fold cross-validation, ShuffleSplit repeatedly creates
    random train/validation partitions. Looking at both the mean and the
    standard deviation gives an additional indication of how stable the model
    is across different samples.
    """
)

st.code(
    """cv_shs = ShuffleSplit(
    n_splits=5,
    test_size=0.3,
    random_state=0
)

scores_shs = cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=cv_shs
)""",
    language="python",
)

shuffle_table = pd.DataFrame(
    {
        "Split": [
            "Split 1",
            "Split 2",
            "Split 3",
            "Split 4",
            "Split 5",
        ],
        "R²": SHUFFLE_SPLIT_SCORES,
    }
)

compact_dataframe(shuffle_table)

fig = shuffle_split_figure(SHUFFLE_SPLIT_SCORES)
centered_pyplot(fig)
plt.close(fig)

st.markdown(
    f"""
    The five scores have a mean R² of
    **{SHUFFLE_SPLIT_SCORES.mean():.3f}** and a standard deviation of
    **{SHUFFLE_SPLIT_SCORES.std():.3f}**.

    The relatively small standard deviation suggests that model performance is
    reasonably stable across these alternative random splits.
    """
)

st.divider()


# ======================================================
# 8. PERMUTATION FEATURE IMPORTANCE
# ======================================================

st.markdown('<div id="feature-importance"></div>', unsafe_allow_html=True)
st.header("8. Permutation Feature Importance")

st.markdown(
    """
    Permutation importance is used to understand which original input
    variables contribute most to the predictive performance of the selected
    XGBoost pipeline.

    The method measures how much the model's performance decreases when the
    information contained in one feature is destroyed by randomly shuffling
    its values.

    A larger decrease in R² means that the model relies more strongly on that
    feature for prediction.
    """
)

feature_table = FEATURE_IMPORTANCE.copy()
feature_table["Importance"] = feature_table["Importance"].round(6)
feature_table["Std"] = feature_table["Std"].round(6)

compact_dataframe(feature_table)

fig = feature_importance_figure(FEATURE_IMPORTANCE)
centered_pyplot(fig)
plt.close(fig)

st.markdown(
    """
    The results show that **apartment size and geographic location are the
    dominant sources of predictive information**.

    - **`log_square_feet`** is the most important feature. Shuffling it reduces
      R² by approximately **0.330** on average.
    - **`state`** is the second most important feature, with an average R²
      decrease of approximately **0.246**.
    - **`cityname`** follows with an importance of approximately **0.153**.
    - `bathrooms`, `bedrooms`, latitude, longitude and their squared terms
      contribute additional predictive information, but substantially less.
    - `n_amenities`, `source` and `pets_allowed_int` have relatively small
      importance.
    - `has_photo_int` has almost no effect on test-set R² in this model.

    An importance value of **0.330** for `log_square_feet` does **not** mean
    that apartment size explains 33% of rent. It means that randomly
    destroying the information contained in this feature lowers the model's
    R² by about **0.33 points on average**.

    Permutation importance measures **predictive importance, not causality**.
    It also does not show whether increasing a feature raises or lowers the
    predicted rent.

    Correlated predictors should be interpreted with caution. For example,
    `state`, `cityname`, `latitude`, and `longitude` contain overlapping
    geographic information, so their predictive contribution may be shared
    across several variables.
    """
)

st.divider()


# ======================================================
# 9. FROM LOG PRICE BACK TO ACTUAL RENT
# ======================================================

st.markdown('<div id="actual-rent"></div>', unsafe_allow_html=True)
st.header("9. Evaluation on Actual Rent")

st.markdown(
    """
    This is one of the most important parts of the notebook.

    The model predicts `log_price`, but the business quantity of interest is
    normally the actual monthly rent in dollars.

    Therefore:

    1. predict `log_price`,
    2. apply `np.exp()` to both the target and prediction,
    3. calculate R² again on the original price scale.
    """
)

st.code(
    """y_pred_test = best_model.predict(X_test)

# R² on log_price
r2_score(y_test, y_pred_test)

# R² after returning to actual rent
r2_score(
    np.exp(y_test),
    np.exp(y_pred_test)
)""",
    language="python",
)

price_scale_results = pd.DataFrame(
    {
        "Evaluation": [
            "XGBoost – log_price",
            "XGBoost – original price",
            "Original price excluding top 2%",
        ],
        "R²": [
            XGB_LOG_PRICE_R2,
            XGB_PRICE_R2,
            XGB_PRICE_TRIMMED_R2,
        ],
    }
)

compact_dataframe(price_scale_results)

fig = price_scale_figure()
centered_pyplot(fig)
plt.close(fig)

st.markdown(
    """
    The difference is substantial:

    - R² on `log_price`: **0.768**
    - R² on actual rent: **0.369**

    A strong score on a transformed target therefore does **not**
    automatically imply equally strong performance on the original business
    scale.
    """
)

st.divider()


# ======================================================
# 10. WHY DOES R² DROP?
# ======================================================

st.markdown('<div id="outliers"></div>', unsafe_allow_html=True)
st.header("10. Influence of Expensive Outliers")

st.markdown(
    f"""
    The notebook identifies the upper **2%** of test-set rents using the
    98th percentile of `log_price`.

    The threshold is:

    - `log_price` ≈ **{PCT98_LOG_PRICE:.3f}**
    - after exponentiation, approximately **${PCT98_PRICE:,.0f} per month**

    This cutoff is used only as a **diagnostic experiment**.

    R² depends on squared errors. A very large prediction error therefore has
    much more influence than a small one. A few very expensive apartments can
    consequently reduce the original-price R² dramatically.
    """
)

refresh_col1, refresh_col2, refresh_col3 = st.columns([1, 2.2, 1])

with refresh_col1:
    if st.button(
        "↻ Refresh figures",
        width="content",
        help="Reload the latest PNG files exported by the notebook.",
    ):
        st.rerun()

show_model_image(
    KDE_IMAGE_PATH,
    "Distribution of log_price in the test set and the 98th-percentile cutoff.",
)

st.markdown(
    """
    After excluding observations above that diagnostic threshold, R² on the
    original price scale rises from approximately **0.369 to 0.739**.

    This does **not** mean that removing the most expensive 2% is automatically
    the correct production solution. Whether the high-rent segment may be
    excluded is a **business decision** and must be stated explicitly whenever
    results are reported.
    """
)

show_model_image(
    SCATTER_IMAGE_PATH,
    "Predicted vs. actual rents. Extreme observations have a disproportionate effect on squared-error metrics.",
)

st.markdown(
    """
    In the predicted-vs-actual plot, points far from the diagonal represent
    large prediction errors. Because those errors are squared inside the
    calculation underlying R², a small number of extreme misses can dominate
    the metric.

    This explains why the model can perform well for the majority of listings
    while still producing a much lower R² when evaluated across the complete
    untransformed price range.
    """
)

st.divider()


# ======================================================
# 11. MODEL PERSISTENCE
# ======================================================

st.markdown('<div id="model-persistence"></div>', unsafe_allow_html=True)
st.header("11. Saving Trained Models")

st.markdown(
    """
    Hyperparameter search can take a long time. The notebook therefore saves
    trained search objects using `pickle`.

    - `train_flag=True` → retrain the model and save it,
    - `train_flag=False` → load the already trained object.

    This allows the notebook to be reopened for evaluation without repeating
    the full search every time.
    """
)

st.code(
    """if train_flag:
    model_xgb = search_grid_and_print_score(...)
    with open("model_xgb.pkl", "wb") as f:
        pickle.dump(model_xgb, f)
else:
    with open("model_xgb.pkl", "rb") as f:
        model_xgb = pickle.load(f)""",
    language="python",
)

st.markdown(
    """
    The Streamlit dashboard follows the same practical idea: this page shows
    the **stored notebook results** instead of launching a new 5-fold,
    30-candidate search whenever a visitor opens the application.
    """
)

st.divider()


# ======================================================
# 12. FINAL INTERPRETATION
# ======================================================

st.markdown('<div id="final-interpretation"></div>', unsafe_allow_html=True)
st.header("12. Final Interpretation")

st.markdown(
    """
    The modeling results support several conclusions:

    - **XGBoost provides the strongest generalization** among the four tested
      approaches on the `log_price` target.
    - Random Forest fits the training data more strongly, but its larger
      train–validation gap indicates more overfitting.
    - Ridge outperforms LASSO, suggesting that retaining correlated features
      is slightly more effective than the stronger feature-selection behavior
      of L1 regularization in this dataset.
    - Both Ridge and LASSO remain below the tree-based models, suggesting that
      nonlinear relationships matter for apartment rents.
    - Permutation importance shows that the model relies most strongly on
      **apartment size and geographic location**, especially
      `log_square_feet`, `state`, and `cityname`.
    - The approximately **0.768 R² applies to the logarithm of rent**, not
      directly to rent in dollars.
    - On the original price scale, extreme expensive listings materially
      reduce R².
    - Removing the upper 2% improves the diagnostic score substantially, but
      such filtering cannot be treated as a neutral technical decision.

    The notebook also notes that the exact business case is not specified.
    In a real project, model evaluation should therefore go beyond a technical
    metric such as R² and define **who will use the prediction, how it will be
    used, and what size of prediction error is acceptable in business terms**.
    """
)

st.caption(
    "Results, parameter values, diagnostics and interpretations on this page "
    "are based on the stored outputs and comments in "
    "`projekt_reseni_EDA_komentare.ipynb`."
)
