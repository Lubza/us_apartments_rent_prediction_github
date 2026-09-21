from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    ax.set_ylabel("Počet")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def compact_dataframe(frame, height=None):
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        height=height,
    )


raw_data = load_data()
data, data_narrow = prepare_data(raw_data)


# ------------------------------------------------------
# HEADER
# ------------------------------------------------------

st.title("Príprava dát")

st.markdown(
    """
    Po exploratívnej analýze dát sa surové premenné transformujú do podoby
    vhodnej na modelovanie.

    **Dôležitý princíp:** transformácie, ktoré sa *učia z dát* — napríklad
    imputácia, Target Encoding alebo škálovanie — sa nesmú fitovať na celom
    datasete pred rozdelením na tréningovú a testovaciu vzorku. Preto sú tieto
    operácie umiestnené až neskôr v machine-learning pipeline a učia sa iba
    z tréningových dát.
    """
)

st.divider()


# ======================================================
# 1. COLUMN-LEVEL PREPROCESSING
# ======================================================

st.header("1. Preprocessing jednotlivých premenných")

st.markdown(
    """
    Prvé kroky sú deterministické operácie feature engineeringu. Surové
    textové alebo rozdielne škálované premenné prevádzajú na kompaktné
    numerické reprezentácie bez toho, aby sa učili štatistiky z celého datasetu.
    """
)


# ------------------------------------------------------
# AMENITIES
# ------------------------------------------------------

st.subheader("`amenities`")

st.markdown(
    """
    `amenities` obsahuje zoznam vybavenia bytu oddelený čiarkami.

    V referenčnom riešení používame zámerne jednoduchú reprezentáciu:
    **počet uvedených prvkov vybavenia** (`n_amenities`).

    **Výhoda:** výpočet je rýchly a takmer nezvyšuje dimenzionalitu.

    **Obmedzenie:** model už nevie, *aké konkrétne* vybavenie je prítomné.
    Chladnička, bazén, tenisový kurt alebo vírivka môžu mať na nájomné veľmi
    odlišný vplyv, no všetky tu zvýšia iba počet o jednu.

    Detailnejšou alternatívou by bolo vytvoriť samostatnú 0/1 multi-hot
    premennú pre každý typ vybavenia.
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
    "Rozdelenie n_amenities",
    "Počet prvkov vybavenia",
    bins=max(10, int(data["n_amenities"].max()) + 1),
)
st.pyplot(fig, use_container_width=False)
plt.close(fig)

st.divider()


# ------------------------------------------------------
# PETS ALLOWED
# ------------------------------------------------------

st.subheader("`pets_allowed`")

st.markdown(
    """
    Podobne ako pri `amenities` sa textový zoznam prevedie na **počet
    povolených typov domácich zvierat**:

    - nič nie je uvedené → `0`
    - pes alebo mačka → `1`
    - oba typy → `2`

    Ide o jednoduchú numerickú reprezentáciu. Rozdiel medzi hodnotami 1 a 2
    však nemusí mať lineárny ekonomický význam pre výšku nájomného.
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
    st.markdown("**Pôvodné hodnoty**")
    pets_original = (
        raw_data["pets_allowed"]
        .fillna("Missing")
        .value_counts()
        .rename_axis("pets_allowed")
        .reset_index(name="count")
    )
    compact_dataframe(pets_original)

with pets_col2:
    st.markdown("**Vytvorené hodnoty**")
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
    Pred modelovaním skontrolujeme, aké frekvencie cien sa v dátach vyskytujú.

    Týždenné a mesačné nájomné musia byť vyjadrené na **rovnakom časovom
    základe**. Inak by týždenná ponuka mohla vyzerať ako extrémne lacný
    mesačný prenájom a skresliť chybu modelu aj $R^2$.

    Referenčné riešenie prevádza záznamy `Weekly` a `Monthly|Weekly` na
    približnú mesačnú cenu pomocou **price × 4**. V datasete je takých záznamov
    iba niekoľko, ale preprocessing má zostať zmysluplný aj pri budúcom
    rozšírení datasetu.
    """
)

price_type_col1, price_type_col2 = st.columns([1, 2])

with price_type_col1:
    st.markdown("**Typy cien**")
    price_type_counts = (
        raw_data["price_type"]
        .value_counts()
        .rename_axis("price_type")
        .reset_index(name="count")
    )
    compact_dataframe(price_type_counts)

with price_type_col2:
    st.markdown("**Nemesiačné pozorovania pred konverziou**")
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
    Pôvodná premenná obsahuje hodnoty `Yes`, `No` a `Thumbnail`.

    Referenčné riešenie ju prevádza na binárnu informáciu:

    - `Yes` alebo `Thumbnail` → `1`
    - inak → `0`

    Myšlienkou je, že fotografia — alebo aspoň náhľad — môže súvisieť s
    kvalitou prezentácie či marketingu ponuky, a tým nepriamo aj s nájomným.
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

st.header("2. Logaritmická transformácia")

st.markdown(
    """
    ### Prečo log-transformujeme `square_feet`?

    Veľkosť bytov má typicky pravostranné zošikmené rozdelenie: väčšina bytov
    je v bežnom rozsahu veľkosti, zatiaľ čo malý počet nehnuteľností je
    extrémne veľký. Logaritmus tieto extrémne hodnoty stlačí a zvyčajne
    vytvorí symetrickejšie rozdelenie.

    ### Prečo je `price` komplikovanejšia?

    `price` je cieľová premenná. Modelovanie `log_price` môže byť užitočné,
    výsledok však treba interpretovať opatrne:

    - model sa trénuje na `log_price`,
    - pri vyhodnocovaní chýb v dolároch treba predikcie previesť späť
      exponenciálnou funkciou,
    - $R^2$ na `log_price` **nie je tá istá metrika** ako $R^2$ na pôvodnej
      škále nájomného.

    K tomuto rozdielu sa vraciame na stránke vyhodnotenia modelov.
    """
)

st.markdown("#### Pred logaritmickou transformáciou")

before_col1, before_col2 = st.columns(2)

with before_col1:
    fig = histogram_figure(
        raw_data["square_feet"],
        "square_feet",
        "Plocha v ft²",
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with before_col2:
    fig = histogram_figure(
        data["price"],
        "price",
        "Mesačné nájomné ($)",
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.code(
    """data['log_square_feet'] = np.log(data['square_feet'])
data['log_price'] = np.log(data['price'])""",
    language="python",
)

st.markdown("#### Po logaritmickej transformácii")

after_col1, after_col2 = st.columns(2)

with after_col1:
    fig = histogram_figure(
        data["log_square_feet"],
        "log_square_feet",
        "log(Plocha v ft²)",
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with after_col2:
    fig = histogram_figure(
        data["log_price"],
        "log_price",
        "log(Mesačné nájomné)",
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.divider()


# ======================================================
# 3. CHECKING REMAINING COLUMNS
# ======================================================

st.header("3. Kontrola zostávajúcich premenných")

st.markdown(
    """
    `value_counts()` používame ako rýchlu diagnostiku kategorizovaných
    premenných. Pomáha identifikovať:

    - konštantné alebo takmer konštantné stĺpce,
    - veľmi zriedkavé kategórie,
    - duplicitné informácie,
    - premenné, ktoré môžu alebo nemusia byť užitočné pre modelovanie.

    V tomto riešení:

    - `currency` je konštantná → vylúčená,
    - `fee` je konštantná → vylúčená,
    - `category` je takmer konštantná → vylúčená,
    - `price_display` duplikuje informáciu o cieľovej cene → vylúčená,
    - `source` ponechávame, pretože rôzne portály môžu obsahovať systematicky
      odlišné skupiny nehnuteľností a cien.
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

st.header("4. Zúženie datasetu na relevantné stĺpce")

st.markdown(
    """
    Teraz vytvoríme `data_narrow`, ktorý obsahuje iba premenné pokračujúce
    do modelovania.

    Cieľom je:

    - odstrániť konštantné alebo takmer konštantné informácie,
    - odstrániť identifikátory a duplicitné informácie,
    - vylúčiť dlhé textové polia, ktoré v tomto riešení nespracúvame pomocou NLP,
    - ponechať `source`, pretože zdrojový portál môže obsahovať užitočnú
      informáciu o cenách.

    Zároveň sa tým nasledujúca preprocessing pipeline stáva prehľadnejšou.
    """
)

selection_table = pd.DataFrame(
    {
        "Premenná": [
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
        "Rozhodnutie": [
            "Vylúčiť",
            "Vylúčiť",
            "Vylúčiť",
            "Vylúčiť",
            "Vylúčiť",
            "Vylúčiť",
            "Vylúčiť",
            "Vylúčiť",
            "Ponechať",
        ],
        "Dôvod": [
            "Prakticky žiadna variabilita",
            "Všetky ceny sú v USD",
            "Takmer všetky pozorovania patria do rovnakej kategórie",
            "Dlhý text; NLP sa v tomto riešení nepoužíva",
            "Dlhý text; NLP sa v tomto riešení nepoužíva",
            "Identifikátor, nie charakteristika bytu",
            "Duplikuje informáciu o cieľovej cene",
            "V tomto riešení sa nepoužíva ako časová premenná",
            "Zdroj ponuky môže obsahovať prediktívnu informáciu o cene",
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

st.markdown("**Ukážka datasetu pre modelovanie**")
compact_dataframe(data_narrow.head(10))

st.divider()


# ======================================================
# 5. MISSING VALUES
# ======================================================

st.header("5. Chýbajúce hodnoty")

st.markdown(
    """
    V tejto fáze chýbajúce hodnoty a nuly iba **identifikujeme**.

    > **Kľúčové pravidlo projektu:** imputácia sa nevykonáva na celom datasete
    > pred rozdelením na train/test.

    Imputer sa fituje až neskôr v pipeline iba na tréningových dátach. Inak by
    informácie z testovacej vzorky mohli nepriamo ovplyvniť imputované hodnoty,
    čo by predstavovalo **data leakage**.

    Nula automaticky neznamená chybu. Napríklad `bedrooms = 0` môže označovať
    štúdiový byt. V produkčnom projekte by sa význam podozrivých nulových
    hodnôt mal ideálne overiť voči zdrojovým portálom.
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

st.header("6. Druhá mocnina zemepisnej šírky a dĺžky")

st.markdown(
    """
    Geografická EDA naznačuje, že nájomné nie je lineárnou funkciou polohy:
    niektoré regióny — napríklad časti pobrežia — majú systematicky vyššie ceny.

    Preto pridávame:

    - `latitude_sq = latitude²`
    - `longitude_sq = longitude²`

    Neskôr `PolynomialFeatures` vytvára ďalšie interakcie. Keďže druhé mocniny
    geografických premenných už existujú, interakcie môžu reprezentovať aj
    vzťahy vyšších rádov vrátane kombinácií zodpovedajúcich tretím a štvrtým
    mocninám. Model tak získava väčšiu flexibilitu pri zachytení zakrivených
    geografických cenových vzorcov.

    Tieto features by sa dali vytvoriť aj priamo v pipeline; tu ich vytvárame
    vopred najmä kvôli jednoduchšej práci so stĺpcami.
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

st.header("7. Rozdelenie train / test")

st.markdown(
    """
    Dáta rozdelíme na **75 % tréningových a 25 % testovacích dát**
    (`test_size=0.25`, `random_state=42`).

    Testovacia vzorka predstavuje budúce, nevidené pozorovania. Imputácia,
    encoding ani scaling preto **nesmú** byť fitované na testovacích dátach.
    Tieto kroky zostávajú v pipeline a učia sa iba z tréningovej časti.
    """
)

n_total = len(data_narrow)
n_test = int(np.ceil(n_total * 0.25))
n_train = n_total - n_test

split_table = pd.DataFrame(
    {
        "Dataset": ["Tréningová vzorka", "Testovacia vzorka"],
        "Podiel": ["75%", "25%"],
        "Riadky": [n_train, n_test],
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

st.header("8. Transformačná pipeline")

st.markdown(
    """
    Všetky naučené transformácie sú zabalené do pipeline, takže každý krok
    sa fituje iba na tréningových pozorovaniach — a počas cross-validation iba
    na tréningovom folde daného splitu.

    **Poradie transformácií**

    1. **Kódovanie kategórií**
       - `cityname`, `state` a `source` dostávajú Target Encoding aj obmedzený
         One-Hot Encoding.
       - `max_categories` obmedzuje počet OHE stĺpcov; menej časté kategórie
         sa zoskupia do `infrequent`.
       - `max_categories` je zároveň ladený hyperparameter.

    2. **KNN imputácia**
       - prebieha po encodingu, pretože `KNNImputer` pracuje s numerickými
         hodnotami,
       - neskôr sa ladí počet susedov aj spôsob váženia.

    3. **Polynomiálne / interakčné features**
       - vytvárajú sa iba pre vybrané premenné, aby dimenzionalita nekontrolovane
         nerástla.

    4. **RobustScaler**
       - znižuje vplyv extrémnych hodnôt,
       - škálovanie je obzvlášť dôležité pri regularizovaných lineárnych
         modeloch ako LASSO a Ridge.

    `verbose_feature_names_out=False` udržiava názvy transformovaných features
    kratšie a jednoduchšie na ďalšie použitie.
    """
)

st.markdown("#### Stĺpce používané v pipeline")

pipeline_columns = pd.DataFrame(
    {
        "Objekt": [
            "cols_ohe",
            "cols_nan",
            "cols_interactions",
        ],
        "Stĺpce": [
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

st.markdown("#### 8.1 Kódovanie")

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
    Target Encoding zachytáva vzťah medzi kategóriou a cieľovou premennou,
    zatiaľ čo obmedzený One-Hot Encoding zachováva explicitnú informáciu o
    najčastejších kategóriách. Použitie oboch dáva modelu dva rôzne pohľady
    na tú istú kategorizovanú informáciu.
    """
)

st.markdown("#### 8.2 KNN imputácia")

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
    KNN imputácia sa vykonáva **po rozdelení train/test a vo vnútri
    pipeline**, takže testovacie dáta neovplyvňujú odhady chýbajúcich hodnôt.
    `n_neighbors` a `weights` sú neskôr zahrnuté do ladenia hyperparametrov.
    """
)

st.markdown("#### 8.3 Interakčné features")

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
    `interaction_only=True` priamo nevytvára všetky druhé mocniny.
    Keďže však `latitude_sq` a `longitude_sq` už existujú, ich interakcie môžu
    reprezentovať komplexnejšie geografické vzťahy vyšších rádov.
    """
)

st.markdown("#### 8.4 Kompletná transformačná pipeline")

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
    Počas cross-validation sa fitovanie **každého kroku** opakuje samostatne
    na tréningovom folde. To je hlavný dôvod, prečo držíme preprocessing a
    model v jednej sklearn pipeline: zabraňuje to leakage a umožňuje spoločne
    optimalizovať hyperparametre preprocessingu aj modelu.
    """
)

st.divider()

st.caption(
    "Táto stránka sleduje štruktúru, transformácie a vysvetľujúce komentáre "
    "referenčného notebooku `projekt_reseni_EDA_komentare.ipynb`, preložené "
    "a prispôsobené pre aplikáciu Streamlit."
)
