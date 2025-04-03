import streamlit as st
st.set_page_config(
    page_title="Titanic verbetercase",
    page_icon="❄️🚢",
    layout="wide"
)

st.title("Veranderingen op de Titanic Case")
st.caption('Team 5, Quinn en Sanne')
st.write('Welkom op deze site, waar de Titanic case 2 keer is uitgevoerd. '
'De eerste keer met 2 weken programmeer kennis en de tweede keer met 9 weken programmeer kennis.'
'  \nMet de 2e keer, zijn de visualisaties verbeterd, zijn nieuwe variabele opgeroepen en is er statistisch meer naar de modellen gekeken.')

st.markdown('# Opzet')
st.write('Voor de opzet is gekozen om gebruik te maken van de streamlit multipager. Dit zorgt ervoor dat je met meerdere streamlit py files werkt en 1 hoofddocument.'
'   \nIn de sidebar valt te kiezen welke webpagina gebruikt wordt(welk thema), dan zijn er nog hoofdstukken en paragraven in de tekst beschreven')

st.markdown('# Veranderingen')
st.write('Zoals al aangegeven zijn er verschillende veranderingen gedaan. '
'Om deze veranderingen inzichtelijk te maken is er een keuze te maken voor de soort verandering:')

verandering= st.radio(
    "",
    options=["🔍 Nieuwe variabelen", "📊 Visualisaties", "📈 statistiek en modellen"],
    index=0)

if verandering == "🔍 Nieuwe variabelen":
    st.header(verandering)
    st.write(""
    "- **Bij de eerste keer** is de kolom **Leeftijdscategorie** toegevoegd.  \n"
    "- **Bij de tweede keer** zijn naast de kolom **Leeftijdscategorie** ook de kolommen:  \n"
    "   - **Titel**   \n"
    "   - **Reisgenoten**    \n"
    "   - **Alleen reizen**")

elif verandering == "📊 Visualisaties":
    st.header(verandering)
    st.write(""
    "- **Eerste keer:** De **correlatiematrix** kreeg feedback dat deze mooier kon.  \n"
    "- **Tweede poging:**  \n"
    "   - De correlatiematrix is verbeterd: van een **getalmatig overzicht** naar een **figuur met kleur en getallen**.  \n"
    "   - De eerste keer waren de figuren gemaakt met **Matplotlib** en soms **Seaborn**.  \n"
    "   - Nu zijn de figuren grotendeels gemaakt met **Plotly Express**, en sommige nog met **Seaborn**.  \n")

elif verandering == "📈 statistiek en modellen":
    st.header(verandering)
    st.write(
        "Bij de eerste keer is er niet direct een algoritme gebruikt. Door middel van figuren en percentages zijn er keuzes gemaakt. "
        "Hieruit zijn voorwaarden gesteld om het resultaat te bepalen. Met deze methode is er een Kaggle-score gehaald van **78,47%**.\n"
        
        "\nBij de tweede keer zijn verschillende algoritmen gebruikt. Er is gekozen om gebruik te maken van **logistische** en **lineaire regressie**. "
        "Hiermee werden scores gehaald van **76,00%** en **78,00%**."
    )