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

KDE_IMAGE_PATH = (FIGURES_DIR / "log_price_distribution_SK.png")

SCATTER_IMAGE_PATH = (FIGURES_DIR / "predicted_vs_actual_SK.png")


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
        "Premenná": [
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
    ax.set_title("Porovnanie modelov na log_price")
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
    ax.set_xlabel("Iterácia ShuffleSplit")
    ax.set_ylabel("R²")
    ax.set_title("XGBoost – ShuffleSplit validácia")
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
        top_features["Premenná"],
        top_features["Importance"],
        xerr=top_features["Std"],
        capsize=3,
    )

    ax.set_xlabel("Pokles R² po permutácii")
    ax.set_ylabel("Premenná")
    ax.set_title("XGBoost – permutačná dôležitosť premenných")
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
    ax.set_title("Vplyv škály vyhodnotenia a extrémnych nájmov")
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
Sekcie vývoja modelov
</div>
<a href="#modeling-approach" target="_self" style="text-decoration:none;">1. Modelovací prístup</a><br>
<a href="#random-forest" target="_self" style="text-decoration:none;">2. Random Forest</a><br>
<a href="#xgboost" target="_self" style="text-decoration:none;">3. XGBoost</a><br>
<a href="#lasso" target="_self" style="text-decoration:none;">4. LASSO</a><br>
<a href="#ridge" target="_self" style="text-decoration:none;">5. Ridge</a><br>
<a href="#model-comparison" target="_self" style="text-decoration:none;">6. Porovnanie modelov</a><br>
<a href="#shufflesplit" target="_self" style="text-decoration:none;">7. ShuffleSplit</a><br>
<a href="#feature-importance" target="_self" style="text-decoration:none;">8. Dôležitosť premenných</a><br>
<a href="#actual-rent" target="_self" style="text-decoration:none;">9. Reálne nájomné</a><br>
<a href="#outliers" target="_self" style="text-decoration:none;">10. Drahé odľahlé hodnoty</a><br>
<a href="#model-persistence" target="_self" style="text-decoration:none;">11. Ukladanie modelov</a><br>
<a href="#final-interpretation" target="_self" style="text-decoration:none;">12. Záverečná interpretácia</a>
</div>""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------
# HEADER
# ------------------------------------------------------

st.title("Vývoj a vyhodnotenie modelov")

st.markdown(
    """
    Táto stránka sleduje časť referenčného notebooku venovanú modelovaniu
    a vyhodnoteniu.

    Cieľovou premennou je **`log_price`** a porovnávame štyri regresné prístupy:

    1. **Random Forest**
    2. **XGBoost**
    3. **LASSO**
    4. **Ridge**

    Hyperparametre sa vyberajú pomocou cross-validation. Následne sa najlepší
    model overí na odloženej testovacej vzorke a napokon sa vyhodnotí aj na
    pôvodnej škále nájomného v amerických dolároch.
    """
)

st.divider()


# ======================================================
# 1. MODELING APPROACH
# ======================================================

st.markdown('<div id="modeling-approach"></div>', unsafe_allow_html=True)
st.header("1. Modelovací prístup")

st.markdown(
    """
    Preprocessing pipeline z predchádzajúcej stránky sa priamo kombinuje
    s každým estimátorom.

    Je to dôležité, pretože vyhľadávanie hyperparametrov následne vyhodnocuje
    **celý workflow** — preprocessing aj model — vo vnútri každého
    cross-validation foldu.

    Parametre preprocessingových krokov preto môžeme ladiť rovnakým spôsobom
    ako parametre samotného modelu.
    """
)

st.subheader("Pomocná funkcia pre hľadanie hyperparametrov")

st.markdown(
    """
    Notebook používa pomocnú funkciu `search_grid_and_print_score()`.
    Napriek názvu implementácia používa **`RandomizedSearchCV`**, nie
    `GridSearchCV`.

    Namiesto testovania každej možnej kombinácie náhodne vyberie pevný počet
    kombinácií parametrov (`n_iter=30`). Pri veľkom priestore hyperparametrov
    je tento postup výrazne rýchlejší.
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
    Pre každý model notebook uvádza:

    - **Najlepšie CV skóre** — priemerné cross-validation R² zvolenej
      kombinácie hyperparametrov,
    - **Train skóre** — R² po fitnutí vybraného modelu na celej tréningovej
      vzorke,
    - **Test skóre** — R² na odloženej testovacej vzorke.

    Cross-validation je hlavným kritériom pri výbere hyperparametrov.
    Testovacia vzorka zostáva rezervovaná na finálnu kontrolu generalizácie.
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
    Random Forest kombinuje veľké množstvo rozhodovacích stromov a agreguje
    ich predikcie. Prirodzene dokáže zachytiť nelineárne vzťahy a interakcie
    medzi premennými.

    Ladenie sa netýka iba samotného Random Forestu. Súčasne sa ladia aj
    parametre predchádzajúcej preprocessing pipeline, napríklad počet
    One-Hot kategórií a nastavenie KNN imputácie.

    To ukazuje dôležitú vlastnosť sklearn pipelines:
    **takmer každý parameter v celom workflow sa môže stať hyperparametrom.**
    """
)

st.code(
    """pipe_model_rf = make_pipeline(
    pipe_transform,
    RandomForestRegressor(random_state=42)
)""",
    language="python",
)

st.subheader("Priestor hyperparametrov Random Forest")

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
        "Testované hodnoty": [
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

st.subheader("Najlepšie parametre Random Forest")

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
        "Najlepšia hodnota": [
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
        "Metrika": [
            "Best CV R²",
            "Train R²",
            "Test R²",
        ],
        "Skóre": [
            0.751,
            0.895,
            0.752,
        ],
    }
)

compact_dataframe(rf_scores)

st.markdown(
    """
    Tréningové skóre je zreteľne vyššie než CV aj test skóre. Tento rozdiel
    naznačuje, že Random Forest sa prispôsobuje tréningovým dátam silnejšie a
    v tomto experimente vykazuje **viac overfittingu** než XGBoost.

    Test skóre však zostáva veľmi blízko CV skóre, čo je dobrým signálom, že
    odhad z cross-validation bol realistický.
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
    XGBoost je gradient-boosting model. Stromy vznikajú **sekvenčne** a každý
    nový strom sa snaží zlepšiť chyby, ktoré zostali po predchádzajúcich stromoch.

    Rovnako ako pri Random Foreste sa spoločne optimalizujú parametre
    preprocessingu aj samotného modelu.

    Cieľom nie je maximalizovať tréningové skóre, ale nájsť kombináciu s
    najlepším **cross-validation R²**.
    """
)

st.code(
    """pipe_model_xgb = make_pipeline(
    pipe_transform,
    XGBRegressor(random_state=42)
)""",
    language="python",
)

st.subheader("Priestor hyperparametrov XGBoost")

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
        "Testované hodnoty": [
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
    **Interpretácia hlavných hyperparametrov XGBoost**

    - `learning_rate` — príspevok každého nového boosting stromu,
    - `n_estimators` — počet stromov,
    - `max_depth` — maximálna komplexita stromu,
    - `subsample` — podiel pozorovaní dostupných pre každý strom,
    - `colsample_bytree` — podiel features dostupných pre každý strom.
    """
)

st.subheader("Najlepšie parametre XGBoost")

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
        "Najlepšia hodnota": [
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
        "Metrika": [
            "Best CV R²",
            "Train R²",
            "Test R²",
        ],
        "Skóre": [
            0.766,
            0.861,
            0.768,
        ],
    }
)

compact_dataframe(xgb_scores)

st.markdown(
    """
    XGBoost má spomedzi štyroch porovnávaných modelov najvyššie CV aj test
    skóre. Jeho tréningové skóre je zároveň nižšie než pri Random Foreste,
    zatiaľ čo test skóre je mierne vyššie.

    Tento vzor naznačuje v tomto experimente **lepší kompromis medzi fitom a
    generalizáciou**.
    """
)

st.divider()


# ======================================================
# 4. LASSO
# ======================================================

st.markdown('<div id="lasso"></div>', unsafe_allow_html=True)
st.header("4. LASSO – referenčný lineárny model")

st.markdown(
    """
    LASSO je lineárny regresný model s **L1 regularizáciou**.

    Slúži ako jednoduchší referenčný model voči dvom stromovým ensemble metódam.

    L1 regularizácia môže niektoré koeficienty zmenšiť až presne na nulu, takže
    zároveň vykonáva určitú formu feature selection. Keďže penalizácia závisí
    od veľkosti koeficientov, správne škálovanie vstupov je veľmi dôležité;
    preto preprocessing pipeline obsahuje `RobustScaler`.
    """
)

st.code(
    """pipe_model_lasso = make_pipeline(
    pipe_transform,
    Lasso()
)""",
    language="python",
)

st.subheader("Priestor hyperparametrov LASSO")

lasso_search = pd.DataFrame(
    {
        "Parameter": [
            "OHE max_categories",
            "KNN n_neighbors",
            "KNN weights",
            "alpha",
        ],
        "Testované hodnoty": [
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
    LASSO má menej hyperparametrov špecifických pre model. Hlavným je
    `alpha`, ktorý riadi silu L1 regularizácie.

    Preprocessing sa stále ladí spoločne, takže aj jednoduchší model dostáva
    rovnaký optimalizačný rámec ako Random Forest a XGBoost.
    """
)

st.subheader("Najlepšie parametre LASSO")

lasso_best = pd.DataFrame(
    {
        "Parameter": [
            "KNN weights",
            "KNN n_neighbors",
            "OHE max_categories",
            "alpha",
        ],
        "Najlepšia hodnota": [
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
        "Metrika": [
            "Best CV R²",
            "Train R²",
            "Test R²",
        ],
        "Skóre": [
            0.679,
            0.735,
            0.680,
        ],
    }
)

compact_dataframe(lasso_scores)

st.markdown(
    """
    LASSO dosahuje podstatne horšie výsledky než oba stromové modely.
    To naznačuje, že nájomné obsahuje dôležité **nelinearity a interakcie**,
    ktoré sa samotnou regularizovanou lineárnou špecifikáciou zachytávajú ťažko.
    """
)

st.divider()


# ======================================================
# 5. RIDGE
# ======================================================

st.markdown('<div id="ridge"></div>', unsafe_allow_html=True)
st.header("5. Ridge – lineárny model s L2 regularizáciou")

st.markdown(
    """
    Ridge je lineárny regresný model s **L2 regularizáciou**. Rovnako ako
    LASSO slúži ako jednoduchší referenčný model voči dvom stromovým ensemble
    metódam.

    L2 regularizácia penalizuje veľké hodnoty koeficientov a zmenšuje ich
    smerom k nule, no na rozdiel od LASSO ich zvyčajne nezníži **presne na
    nulu**. Ridge preto typicky ponechá všetky features v modeli a zníži ich
    individuálny vplyv.

    Ridge môže byť obzvlášť užitočný pri korelovaných prediktoroch, pretože
    dokáže ich príspevok rozdeliť medzi viac súvisiacich premenných namiesto
    výberu iba niektorých z nich. Rovnako ako pri LASSO je dôležité škálovanie;
    tu ho zabezpečuje `RobustScaler`.
    """
)

st.code(
    """pipe_model_ridge = make_pipeline(
    pipe_transform,
    Ridge()
)""",
    language="python",
)

st.subheader("Priestor hyperparametrov Ridge")

ridge_search = pd.DataFrame(
    {
        "Parameter": [
            "OHE max_categories",
            "KNN n_neighbors",
            "KNN weights",
            "alpha",
        ],
        "Testované hodnoty": [
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
    Ridge má menej hyperparametrov špecifických pre model; hlavným je
    `alpha`, ktorý riadi silu **L2 regularizácie**.

    Vyššie `alpha` znamená silnejšiu regularizáciu a výraznejšie zmenšovanie
    koeficientov smerom k nule, zatiaľ čo nižšie `alpha` umožňuje modelu
    voľnejšie sa prispôsobiť tréningovým dátam.

    Súčasne sa stále ladí aj preprocessing — napríklad počet OHE kategórií a
    nastavenie KNN imputácie — takže Ridge sa vyhodnocuje v rovnakom
    optimalizovanom preprocessingovom rámci ako ostatné modely.
    """
)

st.subheader("Najlepšie parametre Ridge")

ridge_best = pd.DataFrame(
    {
        "Parameter": [
            "KNN weights",
            "KNN n_neighbors",
            "OHE max_categories",
            "alpha",
        ],
        "Najlepšia hodnota": [
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
        "Metrika": [
            "Best CV R²",
            "Train R²",
            "Test R²",
        ],
        "Skóre": [
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
    Oba modely sú regularizované lineárne regresie, ale s koeficientmi
    pracujú odlišne:

    - **LASSO (L1)** môže niektoré koeficienty zmenšiť presne na nulu, a preto
      vykonáva formu feature selection.
    - **Ridge (L2)** zmenšuje koeficienty smerom k nule, ale zvyčajne ponechá
      všetky features, čo môže byť výhodné pri prekrývajúcej sa alebo
      korelovanej informácii.

    V tomto projekte dosahuje Ridge lepší výsledok než LASSO pri cross-validation
    aj na odloženej testovacej vzorke:

    - Ridge: CV R² = **0.696**, Test R² = **0.693**
    - LASSO: CV R² = **0.679**, Test R² = **0.680**

    Zlepšenie je mierne a oba lineárne modely zostávajú pod stromovými ensemble
    modelmi. To naznačuje, že zachovanie korelovanej informácie pomáha
    lineárnemu benchmarku, no stále zostávajú dôležité nelineárne vzťahy.
    """
)

st.divider()


# ======================================================
# 6. MODEL COMPARISON
# ======================================================

st.markdown('<div id="model-comparison"></div>', unsafe_allow_html=True)
st.header("6. Porovnanie modelov")

st.markdown(
    """
    Modely najskôr porovnávame na **`log_price`**, pretože práve na tejto
    cieľovej premennej boli trénované.

    Veľký rozdiel medzi train a test skóre môže signalizovať overfitting.
    Pri výbere hyperparametrov je však najdôležitejšie cross-validation skóre;
    odložené test skóre používame až ako finálnu kontrolu generalizácie.
    """
)

compact_dataframe(MODEL_RESULTS)

fig = grouped_score_figure(MODEL_RESULTS)
centered_pyplot(fig)
plt.close(fig)

st.markdown(
    """
    **XGBoost je v notebooku zvolený ako najlepší model.**

    - XGBoost: CV R² = **0.766**, Test R² = **0.768**
    - Random Forest: CV R² = **0.751**, Test R² = **0.752**
    - Ridge: CV R² = **0.696**, Test R² = **0.693**
    - LASSO: CV R² = **0.679**, Test R² = **0.680**

    Random Forest dosahuje najvyššie tréningové skóre, to z neho však nerobí
    najsilnejší model. Väčší rozdiel medzi train a validation skóre ukazuje,
    že dodatočný fit sa negeneralizuje rovnako dobre.
    """
)

st.caption(
    "Poznámka k notebooku: jeden neskorší print označuje test skóre 0.768 ako "
    "'random forest', ale kód bezprostredne nad ním definuje `best_model` ako "
    "`model_xgb.best_estimator_`. Číselné skóre preto patrí zvolenému modelu "
    "XGBoost."
)

st.divider()


# ======================================================
# 7. SHUFFLESPLIT VALIDATION
# ======================================================

st.markdown('<div id="shufflesplit"></div>', unsafe_allow_html=True)
st.header("7. Dodatočná validácia pomocou ShuffleSplit")

st.markdown(
    """
    Zvolený XGBoost estimátor ešte raz overujeme pomocou **`ShuffleSplit`**.

    Na rozdiel od bežnej K-Fold cross-validation vytvára ShuffleSplit opakovane
    náhodné train/validation rozdelenia. Priemer a smerodajná odchýlka dávajú
    dodatočnú informáciu o stabilite modelu pri rôznych vzorkách.
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
    **{SHUFFLE_SPLIT_SCORES.mean():.3f}** a smerodajnú odchýlku
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
st.header("8. Permutačná dôležitosť premenných")

st.markdown(
    """
    Permutation importance používame na zistenie, ktoré pôvodné vstupné
    premenné najviac prispievajú k prediktívnemu výkonu zvolenej XGBoost pipeline.

    Metóda meria, o koľko sa výkon modelu zhorší, keď informáciu v jednej
    feature zničíme náhodným premiešaním jej hodnôt.

    Väčší pokles R² znamená, že model sa pri predikcii na danú feature spolieha
    vo väčšej miere.
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
    Výsledky ukazujú, že **veľkosť bytu a geografická poloha sú dominantnými
    zdrojmi prediktívnej informácie**.

    - **`log_square_feet`** je najdôležitejšia feature. Jej premiešanie zníži
      R² v priemere približne o **0.330**.
    - **`state`** je druhá najdôležitejšia feature s priemerným poklesom R²
      približne **0.246**.
    - **`cityname`** nasleduje s importance približne **0.153**.
    - `bathrooms`, `bedrooms`, latitude, longitude a ich druhé mocniny
      pridávajú ďalšiu prediktívnu informáciu, ale podstatne menšiu.
    - `n_amenities`, `source` a `pets_allowed_int` majú relatívne malú importance.
    - `has_photo_int` má v tomto modeli takmer nulový vplyv na testovacie R².

    Hodnota importance **0.330** pri `log_square_feet` **neznamená**, že veľkosť
    bytu sama vysvetľuje 33 % nájomného. Znamená, že náhodné zničenie informácie
    v tejto feature zníži R² modelu približne o **0.33 bodu v priemere**.

    Permutation importance meria **prediktívnu dôležitosť, nie kauzalitu**.
    Zároveň nehovorí, či zvýšenie feature zvyšuje alebo znižuje predikované nájomné.

    Korelované prediktory treba interpretovať opatrne. Napríklad `state`,
    `cityname`, `latitude` a `longitude` obsahujú prekrývajúcu sa geografickú
    informáciu, takže ich prediktívny prínos sa môže deliť medzi viac premenných.
    """
)

st.divider()


# ======================================================
# 9. FROM LOG PRICE BACK TO ACTUAL RENT
# ======================================================

st.markdown('<div id="actual-rent"></div>', unsafe_allow_html=True)
st.header("9. Vyhodnotenie na reálnom nájomnom")

st.markdown(
    """
    Toto je jedna z najdôležitejších častí notebooku.

    Model predikuje `log_price`, ale z business pohľadu nás zvyčajne zaujíma
    skutočné mesačné nájomné v dolároch.

    Preto:

    1. predikujeme `log_price`,
    2. na target aj predikciu aplikujeme `np.exp()`,
    3. znovu vypočítame R² na pôvodnej cenovej škále.
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
        "Vyhodnotenie": [
            "XGBoost – log_price",
            "XGBoost – pôvodná cena",
            "Pôvodná cena bez horných 2 %",
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
    Rozdiel je výrazný:

    - R² na `log_price`: **0.768**
    - R² na reálnom nájomnom: **0.369**

    Silné skóre na transformovanej cieľovej premennej preto **automaticky
    neznamená** rovnako silný výkon na pôvodnej business škále.
    """
)

st.divider()


# ======================================================
# 10. WHY DOES R² DROP?
# ======================================================

st.markdown('<div id="outliers"></div>', unsafe_allow_html=True)
st.header("10. Vplyv drahých odľahlých hodnôt")

st.markdown(
    f"""
    Notebook identifikuje horné **2 %** nájomného v testovacej vzorke pomocou
    98. percentilu `log_price`.

    Prahová hodnota je:

    - `log_price` ≈ **{PCT98_LOG_PRICE:.3f}**
    - po exponenciácii približne **${PCT98_PRICE:,.0f} mesačne**

    Tento cutoff sa používa iba ako **diagnostický experiment**.

    R² závisí od štvorcov chýb. Veľká predikčná chyba má preto
    oveľa väčší vplyv než malá chyba. Niekoľko veľmi drahých bytov tak môže
    výrazne znížiť R² na pôvodnej cenovej škále.
    """
)

refresh_col1, refresh_col2, refresh_col3 = st.columns([1, 2.2, 1])

with refresh_col1:
    if st.button(
        "↻ Obnoviť grafy",
        width="content",
        help="Znovu načíta najnovšie PNG súbory exportované z notebooku.",
    ):
        st.rerun()

show_model_image(
    KDE_IMAGE_PATH,
    "Rozdelenie log_price v testovacej vzorke a hranica 98. percentilu.",
)

st.markdown(
    """
    Po vylúčení pozorovaní nad týmto diagnostickým prahom stúpne R² na
    pôvodnej cenovej škále približne z **0.369 na 0.739**.

    To **neznamená**, že odstránenie najdrahších 2 % je automaticky správnym
    produkčným riešením. To, či možno segment vysokého nájomného vylúčiť, je
    **business rozhodnutie** a pri reportovaní výsledkov musí byť uvedené
    explicitne.
    """
)

show_model_image(
    SCATTER_IMAGE_PATH,
    "Predikované vs. skutočné nájomné. Extrémne pozorovania majú neúmerný vplyv na metriky založené na štvorcoch chýb.",
)

st.markdown(
    """
    V grafe predicted-vs-actual predstavujú body ďaleko od diagonály veľké
    predikčné chyby. Keďže tieto chyby vstupujú do výpočtu R² v druhej mocnine,
    malý počet extrémnych omylov môže metriku výrazne ovplyvniť.

    To vysvetľuje, prečo môže model fungovať dobre pre väčšinu ponúk, no pri
    vyhodnotení na celom netransformovanom cenovom rozsahu stále dosiahne
    podstatne nižšie R².
    """
)

st.divider()


# ======================================================
# 11. MODEL PERSISTENCE
# ======================================================

st.markdown('<div id="model-persistence"></div>', unsafe_allow_html=True)
st.header("11. Ukladanie natrénovaných modelov")

st.markdown(
    """
    Hľadanie hyperparametrov môže trvať dlho. Notebook preto ukladá
    natrénované search objekty pomocou `pickle`.

    - `train_flag=True` → model znovu natrénovať a uložiť,
    - `train_flag=False` → načítať už natrénovaný objekt.

    Notebook sa tak dá znovu otvoriť na vyhodnotenie bez opakovania celého
    vyhľadávania pri každom spustení.
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
    Streamlit dashboard používa rovnaký praktický princíp: stránka zobrazuje
    **uložené výsledky notebooku** namiesto spúšťania nového 5-fold,
    30-kandidátového vyhľadávania pri každom otvorení aplikácie.
    """
)

st.divider()


# ======================================================
# 12. FINAL INTERPRETATION
# ======================================================

st.markdown('<div id="final-interpretation"></div>', unsafe_allow_html=True)
st.header("12. Záverečná interpretácia")

st.markdown(
    """
    Výsledky modelovania podporujú niekoľko záverov:

    - **XGBoost má spomedzi štyroch testovaných prístupov na targete `log_price`
      najsilnejšiu generalizáciu**.
    - Random Forest fituje tréningové dáta silnejšie, no väčší rozdiel medzi
      train a validation skóre naznačuje vyšší overfitting.
    - Ridge prekonáva LASSO, čo naznačuje, že zachovanie korelovaných features
      je v tomto datasete mierne efektívnejšie než silnejší feature-selection
      efekt L1 regularizácie.
    - Ridge aj LASSO zostávajú pod stromovými modelmi, čo naznačuje význam
      nelineárnych vzťahov pri nájomnom.
    - Permutation importance ukazuje, že model sa najviac spolieha na
      **veľkosť bytu a geografickú polohu**, najmä `log_square_feet`, `state`
      a `cityname`.
    - Približné **R² = 0.768 platí pre logaritmus nájomného**, nie priamo pre
      nájomné v dolároch.
    - Na pôvodnej cenovej škále drahé extrémne ponuky výrazne znižujú R².
    - Odstránenie horných 2 % výrazne zlepšuje diagnostické skóre, no takéto
      filtrovanie nemožno považovať za neutrálne technické rozhodnutie.

    Notebook zároveň uvádza, že presný business case nie je definovaný.
    V reálnom projekte by vyhodnotenie modelu malo ísť nad rámec technickej
    metriky R² a definovať **kto bude predikciu používať, ako sa bude používať
    a aká veľkosť predikčnej chyby je prijateľná z business pohľadu**.
    """
)

st.caption(
    "Výsledky, hodnoty parametrov, diagnostiky a interpretácie na tejto "
    "stránke vychádzajú z uložených výstupov a komentárov v notebooku "
    "`projekt_reseni_EDA_komentare.ipynb`."
)
