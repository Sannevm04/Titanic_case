import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from streamlit_option_menu import option_menu
import plotly.subplots as sp

# opmaak
st.set_page_config(
    page_title="Nieuwe koers",
    page_icon="🚢",
    layout="wide"
)
st.title("🚢 Verbeterpoging")
with st.sidebar: 
    pagina = option_menu('Inhoudsopgave', ["Data verkenning", "Analyse", "Voorspellend model"], icons=['search', 'bar-chart-line', 'graph-up-arrow'], menu_icon='card-list')

# data inladen
@st.cache_data
def load_train_new():
    train_new = pd.read_csv('train.csv')
    train_new.drop(columns = ['Ticket', 'Cabin', 'PassengerId'], 
                   axis = 1, inplace = True)
    
    # nieuwe kolommen    # leeftijdscategorieën toevoegen
    def leeftijdscategorie(leeftijd):
        if leeftijd < 16:
            return 1
        elif 16 <= leeftijd <= 34:
            return 2
        elif 35 <= leeftijd <= 49:
            return 3
        elif 50 <= leeftijd <= 64:
            return 4
        else:
            return 5
    df= pd.DataFrame({'Leeftijd': train_new['Age']})
    train_new['Age_categories'] = df['Leeftijd'].apply(leeftijdscategorie)

    # travel alone, of met gezelschap
    train_new['Travel_budy'] = train_new['SibSp'] + train_new['Parch'] + 1
    train_new['IsAlone'] = (train_new['Travel_budy'] == 1).astype(int)

    # Naam naar titels geven
    train_new['Title'] = train_new['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    train_new['Title'] = train_new['Title'].map(lambda x: title_mapping.get(x, 5))
    train_new['Embarked'] = train_new['Embarked'].replace({'S':'Southampthon', 'C':'Cherbourgh', 'Q': 'Queenstown'})
    return train_new

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, 'Rare':5, "Sir":1,"Lady":3}
title_demapping = {1: "Mr", 2: "Miss", 3: "Mrs", 4: "Master", 5: "Rare"}
sex_mapping = {"male":0, "female":1}
sex_demapping = {0:'male', 1:"female"}
embarked_mapping = {'Southampthon':0, 'Cherbourgh':1,'Queenstown':2}
embarked_demapping = {0:'Southampthon', 1:'Cherbourgh', 2:'Queenstown'}
Pclass_mapping = {'1e klas': 1, '2e klas':2, '3e klas':3}
Pclass_demapping = {1: '1e klas', 2:'2e klas', 3:'3e klas'}
Alone_mapping = {'Alleen':1, 'Samen':0}
Alone_demapping = {1:'Alleen', 0:'Samen'}
Age_cat_mapping = {'Kind (<16)':1, 'Jong volwassen (16<=34)':2, 'Volwassen (35<=49)':3, 'Middelbare leeftijd (50<=64)':4, 'Oudere (>65)':5}
Age_cat_demapping = {1: 'Kind (<16)', 2:'Jong volwassen (16<=34)', 3:'Volwassen (35<=49)', 4:'Middelbare leeftijd (50<=64)', 5:'Oudere (>65)'}

train_new = load_train_new()

if pagina == 'Data verkenning':
    st.header('1. Data verkenning')
    st.dataframe(train_new.head())
    kolommen = train_new.columns

    st.subheader('1.1 Data opvulling')
    st.write('De missende waardes komen voor in leeftijd en Opstaplocatie:')
    missing_values = train_new.isnull().sum()
    missing_values = missing_values[missing_values>0].reset_index()
    missing_values.columns = ['Kolom', 'Aantal missende waardes']
    st.dataframe(missing_values, hide_index=True, use_container_width=True)
    st.info(
        f"In totaal zijn er {missing_values['Aantal missende waardes'].sum()} missende waardes "
        f"verdeeld over {missing_values.shape[0]} kolommen. Deze waardes gaan we dan opvullen."
    )

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Verdeling Leeftijd", "Opstaplocatie"])
    fig.add_trace(
        go.Histogram(x=train_new["Age"], nbinsx=10, marker_color="lightblue", name="Leeftijd"), 
        row=1, col=1)

    # Voeg histogram voor ticketprijs toe
    fig.add_trace(
        go.Histogram(x=train_new["Embarked"], nbinsx=10, marker_color="lightgreen", name="Ticketprijs"), 
        row=1, col=2)

    # Pas de layout aan
    fig.update_layout(
        title_text="Histogrammen van Leeftijd en Opstaplocatie",
        showlegend=False,
        height=500, width=900    )

    # Toon in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"De leeftijd heeft een mediaan ({train_new['Age'].median()}) die bijna gelijk is aan het gemiddelde ({np.round(train_new['Age'].mean())})" 
        "en zal daarom opgevuld worden met de mediaan  \n"
        f"De opstaplocatie heeft bij Southampthon zo'n hoge waarde, dat deze gebruikt wordt om de missende waardes in te vullen"
    )
    train_new['Age'].fillna(train_new['Age'].median(), inplace=True)
    train_new['Embarked'].fillna('S', inplace=True)

    st.subheader('1.2 Nieuwe waardes toevoegen')
    st.write("Naast de basis kolommen, zijn er ook kolommen toegevoegd:  \n"
    "* Leeftijd is onderverdeeld in categoriën  \n"
    "* Reisgenoten is toegevoegd, om aan te geven om iemand alleen of samen reisde  \n"
    "* Voorvoegsel van de naam is achterhaald")

    # Nieuwe data kolommen
    st.dataframe(train_new[['Age','Age_categories','Travel_budy', 'IsAlone','Title']].head())

    st.subheader('1.3 correlatie matrix')
    train_new['Sex'] = train_new['Sex'].map(sex_mapping)
    train_new['Embarked'] = train_new['Embarked'].map(embarked_mapping)
    correlatie_matrix = train_new.drop(columns=['Name']).corr()
    correlatie_matrix['abs'] = correlatie_matrix['Survived'].abs()
    def classificatie(r):
        if abs(r) > 0.5:
            return "Sterk"
        elif abs(r) > 0.3:
            return "Matig"
        elif abs(r) > 0:
            return "Zwak"
        else:
            return "Geen"
    correlatie_matrix['Sterkte'] = correlatie_matrix['Survived'].apply(classificatie)
    correlatie_matrix = correlatie_matrix.sort_values(by='abs', ascending=False).drop(columns=['abs'])
    
    # Maak aangepaste annotaties met zowel de correlatie als de classificatie
    annotaties = correlatie_matrix.apply(lambda row: f"{row['Survived']:.2f}\n({row['Sterkte']})", axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        correlatie_matrix[['Survived']], 
        annot=annotaties.values.reshape(-1,1),  # Gebruik aangepaste annotaties
        cmap="coolwarm", fmt="", linewidths=0.5, center=0, ax=ax
    )
    ax.set_title("Correlatie over de overlevingskans op de Titanic")
    # Display plot in Streamlit
    st.pyplot(fig)
    st.info("Het lijkt interessant te zijn om te kijken naar:  \n* Geslacht  \n* Title  \n* Klas  "
    "\n* Ticketprijs   \n* Alleen of samen reizen  \n* Daarnaast zal leeftijd en opstaplocatie ook nog is meegenomen worden.")

    st.subheader("1.4 Verdieping in de variabelen")
    st.write("Om meer inzicht te krijgen in de variabelen kan hieronder gekozen worden voor variabelen waar meer verdiept in kan worden")
    Variabele = st.selectbox("Selecteer een variabele",options=['Sex','Title','Pclass','Fare','IsAlone','Age_categories','Embarked'])
    
    st.markdown("##### 1.4.1 Histogram")
    train_new['Sex'] = train_new['Sex'].map(sex_demapping)
    train_new['Embarked'] = train_new['Embarked'].map(embarked_demapping)
    train_new['Title'] = train_new['Title'].map(title_demapping)
    train_new['Pclass'] = train_new['Pclass'].map(Pclass_demapping)
    train_new['IsAlone'] = train_new['IsAlone'].map(Alone_demapping)
    train_new['Age_categories'] = train_new['Age_categories'].map(Age_cat_demapping)

    if Variabele == "Sex":
        kleur_dict = {
        "male": "lightblue",  
        "female": "pink"}
        fig = px.histogram(train_new, x=Variabele, title=f"Histogram van {Variabele}",
                        color=train_new['Sex'],
                        color_discrete_map=kleur_dict)
        st.plotly_chart(fig)
    else:
        fig = px.histogram(train_new, x=Variabele, title=f"Histogram van {Variabele}", color_discrete_sequence=['lightgreen'])
        st.plotly_chart(fig)
    
    
    
    st.markdown("##### 1.4.2 Onderlinge correlatie")
    train_new['Sex'] = train_new['Sex'].map(sex_mapping)
    train_new['Embarked'] = train_new['Embarked'].map(embarked_mapping)
    train_new['Title'] = train_new['Title'].map(title_mapping)
    train_new['Pclass'] = train_new['Pclass'].map(Pclass_mapping)
    train_new['IsAlone'] = train_new['IsAlone'].map(Alone_mapping)
    train_new['Age_categories'] = train_new['Age_categories'].map(Age_cat_mapping)

    correlatie_matrix_Var = train_new.drop(columns=['Name','Age','SibSp','Parch']).corr()
    correlatie_matrix_Var['abs'] = correlatie_matrix_Var[Variabele].abs()
    correlatie_matrix_Var['Sterkte'] = correlatie_matrix_Var[Variabele].apply(classificatie)
    correlatie_matrix_Var = correlatie_matrix_Var.sort_values(by='abs', ascending=False).drop(columns=['abs']).head(3)
    
    # Maak aangepaste annotaties met zowel de correlatie als de classificatie
    annotaties = correlatie_matrix_Var.apply(lambda row: f"{row[Variabele]:.2f}\n({row['Sterkte']})", axis=1)
    fig, ax = plt.subplots(figsize=(5, 2))
    sns.heatmap(
        correlatie_matrix_Var[[Variabele]], 
        annot=annotaties.values.reshape(-1,1),  # Gebruik aangepaste annotaties
        cmap="coolwarm", fmt="", linewidths=0.5, center=0, ax=ax
    )
    ax.set_title("Onderling Correlatie")
    # Display plot in Streamlit
    st.pyplot(fig)

    if Variabele == "Sex":
        st.info("Er is een correlatie met Titel. Dit is ook zeker logisch, "
        "gezien een man over het algemeen met Mr/Master aangesproken, en een vrouw met Mrs/Miss."
        "Toch blijft Sex en Title voor nu van belang om te bepalen of bepaalde Titels meer/minder overlevingskans hebben.")
    elif Variabele == 'Title':
        st.info("Er is een correlatie met geslacht. Dit is ook zeker logisch, "
        "gezien een man over het algemeen met Mr/Master aangesproken, en een vrouw met Mrs/Miss."
        "Toch blijft Sex en Title voor nu van belang om te bepalen of bepaalde Titels meer/minder overlevingskans hebben.")
    elif Variabele == 'Pclass':
        st.info("Er is een correlatie met Ticketprijs. Dit is ook wel te verwachten, "
        "gezien dat met de klasse stijging ook de prijs mee zal stijgen.")
    elif Variabele == 'Fare':
        st.info("Er is een correlatie met Klasse. Dit is ook wel te verwachten, "
        "gezien dat met de klasse stijging ook de prijs mee zal stijgen.")
    elif Variabele == 'IsAlone':
        st.info('Alleen reizen correleerd met of je reisgenoten hebt, dit is logisch. Beide zullen onafhankelijk nog woren meegenomen.')
    
    if Variabele == 'Age_categories':
        st.markdown('##### Andere handige visualisaties')
        fig = px.box(train_new, y='Age')
        st.plotly_chart(fig)
    elif Variabele == 'Fare':
        st.markdown('##### Andere handige visualisaties')
        fig = px.box(train_new, y=Variabele)
        st.plotly_chart(fig)

elif pagina == 'Analyse':
    st.header('2. Analyse')
    st.write('De onafhankelijke variabelen zullen hier verder onderzocht worden tegenover de afhankelijke varariabele: overleefingskans.')
    
    st.subheader('2.1 Individuele variabelen tegenover overlevingskans')
    #demapping
    train_new['Title'] = train_new['Title'].map(title_demapping)
    train_new['Age_categories']= train_new['Age_categories'].map(Age_cat_demapping)
    train_new['IsAlone'] = train_new['IsAlone'].map(Alone_demapping)
    train_new["Pclass"] = train_new["Pclass"].map(Pclass_demapping)

    # Voorbeeld dataset (vervang dit door je eigen dataset)
    train_old = pd.DataFrame({
        'Pclass': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        'Survived': [1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
        'Age': [22, 38, 26, 35, 28, 42, 30, 25, 31, 24],
        'Sex': ['male', 'female', 'female', 'male', 'male', 'female', 'male', 'female', 'female', 'male']
    })

    # Functie om overlevingskans per categorie te berekenen en te plotten
    def plot_multiple_graphs(data, variables, target='Survived'):
        rows, cols = 2, 3
        fig = sp.make_subplots(rows=rows, cols=cols, subplot_titles=variables)

        for i, variable in enumerate(variables):
            # Bereken de overlevingskans per variabele
            survival_by_var = data.groupby([variable])[target].mean() * 100
            
            # Maak een bar chart
            trace = go.Bar(
                x=survival_by_var.index,
                y=survival_by_var.values,
                text=np.round(survival_by_var.values, 2),  # Percentage met 2 decimalen als tekst
                hoverinfo='x+y+text',
                name=f'Overlevingskans per {variable}'
            )

            # Voeg de trace toe aan de subplot
            fig.add_trace(trace, row=(i // cols) + 1, col=(i % cols) + 1)

            # Update de y-as limieten voor elke subplot
            fig.update_yaxes(range=[0, 100], row=(i // cols) + 1, col=(i % cols) + 1)

        # Layout aanpassen
        fig.update_layout(
            title="Overlevingskans per categorieën",
            showlegend=False,
            height=800,  # Aangepaste hoogte voor de subplots
            yaxis=dict(range=[0, 100], title='Overlevingskans (%)')
        )

        # Toon de grafiek in Streamlit
        st.plotly_chart(fig)

    # Selecteer welke variabelen je wilt plotten (bijvoorbeeld 'Pclass', 'Age', 'Sex')
    variables_to_plot = st.multiselect('Kies de variabelen om te plotten:', ['Sex','Title','Age_categories','IsAlone','Embarked','Pclass'], default=['Sex','Title','Age_categories','IsAlone','Embarked','Pclass'])

    # Grafieken tonen
    plot_multiple_graphs(train_new, variables_to_plot)

    st.info("De conclusies uit de grafieken, als alles bekeken wordt:  \n"
    "* Als vrouw heb je een hoge overlevingskans, als man was je overlevingskans erg laag.  \n"
    "* Kinderen (<16) hebben een overlevingskans van 60%, alles ouder overleed meer dan 50%.  \n"
    "* Bij de titels hadden Masters (jongens), Miss's (meisjes) en Mrs's (vrouwen) een hoge overlevingskans. Mr's hadden een hele lage overlevingskans. \n"
    "* Bij de opstaplocaties was er een hoge overlevingskans bij Cherbourgh, en een lage bij Queenstown em Southampton.  \n"
    "* De 1e klas had nog redelijke overlevingskans. Daaropvolgde de 2e klas met een kans van 47%, dus meer overleefde het niet. De 3e klas had helaas geen geluk. \n"
    "* Als je alleen reisde was je overlevingskand klein, als je samenreisde was deze iets groter.")

    
    
    
    st.subheader('2.2 elkaar ondersteunende onafhankelijke variabelen tegenover overlevingskans')
    st.write('Naast de hierboven getrokken conclusies kan het zijn dat de overlevingskans hoger wordt als er combinaties gemaakt worden tussen de onafhanklijke variabelen.'
    'Om dit te bekijken kunnen 2 variabelen tegen overlevingskans geplot worden.')
    def plot_barchart(data, target='Survived'):
        # Keuze voor de onafhankelijke variabelen
        keuze = ['Sex','Age_categories','IsAlone','Embarked','Pclass']
        var1 = st.selectbox('Kies de eerste onafhankelijke variabele:', keuze)
        keuze.remove(var1)
        var2 = st.selectbox('Kies de tweede onafhankelijke variabele:', keuze)
        keuze.remove(var2)
        var3 = st.selectbox('Kies de derde onafhankelijke variabele:', keuze +['Geen filtering'])
        if var3 == 'Geen filtering':
            filtered_data = data
        else:
            var3_values = data[var3].unique()
            var3_value = st.selectbox(f'Kies de categorie voor {var3}:', var3_values)
            filtered_data = data[data[var3] == var3_value]

        # data filteren
        grouped_data = filtered_data[filtered_data[target] == 1].groupby([var1, var2]).size().reset_index(name='Aantal')
        total_data = data.groupby([var1, var2]).size().reset_index(name='Totaal')
        merged_data = pd.merge(grouped_data, total_data, on=[var1, var2])
        merged_data['Percentage'] = (merged_data['Aantal'] / merged_data['Totaal']) * 100

        # Maak de bar chart met percentages
        fig = px.bar(merged_data, x=var1, y='Percentage', color=var2,
                    labels={var1: var1, 'Percentage': f'Percentage {target} = 1', var2: var2},
                    title=f'Percentage {target} per {var1} en {var2}',
                    barmode='group')
        fig.update_traces(text=merged_data['Percentage'].round(2).astype(str) + '%',
                      textposition='outside',
                      texttemplate='%{text}')
        # Toon de grafiek
        st.plotly_chart(fig)

    # Gebruik de functie
    plot_barchart(train_new)
    
    st.info("Intressante mogelijkheden zijn: gelacht tegen leeftijd: Dit laat jonge mannen een hogere overlevingskans hebben. De vrouwen lijken allemaal een hoge overlevingskans te hebben")
    
elif pagina == 'Voorspellend model':
    st.header('Voorspelling')
    st.write('Met dit model kwam de kaggle score uit op: ..,..%')