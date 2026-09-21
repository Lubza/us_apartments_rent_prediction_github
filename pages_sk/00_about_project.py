import pandas as pd
import streamlit as st


# ------------------------------------------------------
# HEADER
# ------------------------------------------------------

st.title("O projekte")

st.markdown(
    """
    Cieľom projektu je modelovať ceny prenájmu bytov v Spojených štátoch.
    Dataset obsahuje približne **10 000 pozorovaní z roku 2019** a podľa
    štruktúry premennej `source` bol pravdepodobne vytvorený web scrapingom
    viacerých realitných portálov.

    Pri použitom postupe dosiahol model na testovacej vzorke pre
    **logaritmus ceny** hodnotu **R² = 76,8 %**.

    Pre cenu bez logaritmickej transformácie je **R² = 37,0 %**. Po odfiltrovaní
    horných 2 % cien však skóre stúpne na **R² = 74,0 %**.
    """
)

st.info(
    """
    **Poznámka k riešeniu:** konkrétny business case nie je definovaný,
    preto sa toto zhrnutie zameriava najmä na dataset a dosiahnuté výsledky
    modelovania. V reálnom projekte by táto časť mala jasne uvádzať, kto bude
    model používať, na čo budú predikcie slúžiť a aká veľkosť chyby je z
    obchodného pohľadu prijateľná. Zároveň je vhodné zvážiť, či technická
    metrika ako R² patrí priamo do zhrnutia.
    """
)

st.divider()


# ======================================================
# 2. DATASET DESCRIPTION
# ======================================================

st.header("Opis datasetu")

st.markdown(
    """
    Dataset obsahuje ponuky prenájmu bytov, ich charakteristiky,
    geografické informácie, metadáta inzerátov a ceny prenájmu.
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
        "Popis": [
            "Jedinečný identifikátor ponuky bytu.",
            "Kategória ponuky.",
            "Text názvu inzerátu.",
            "Textový opis inzerátu.",
            (
                "Vybavenie bytu, napríklad klimatizácia, basketbalové ihrisko, "
                "káblová TV, fitness centrum, internet, bazén, chladnička a pod."
            ),
            "Počet kúpeľní.",
            "Počet spální.",
            "Mena ponuky.",
            "Prípadný dodatočný poplatok.",
            "Informácia, či ponuka obsahuje fotografie bytu.",
            "Aké domáce zvieratá sú povolené (napr. psy, mačky).",
            "Cena prenájmu bytu.",
            "Cena vo formáte určenom na zobrazenie.",
            "Frekvencia alebo typ ceny v USD.",
            "Veľkosť bytu v štvorcových stopách.",
            "Poloha alebo adresa bytu.",
            "Mesto, v ktorom sa byt nachádza.",
            "Štát USA, v ktorom sa byt nachádza.",
            "Zemepisná šírka polohy bytu.",
            "Zemepisná dĺžka polohy bytu.",
            "Zdroj ponuky.",
            "Čas vytvorenia ponuky.",
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
        "Popis": st.column_config.TextColumn(
            "Popis",
            width="large",
        ),
    },
)

st.caption(
    "Zhrnutie a opis datasetu na tejto stránke vychádzajú z úvodných "
    "častí projektového Jupyter notebooku."
)
