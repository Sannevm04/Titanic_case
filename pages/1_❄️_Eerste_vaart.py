import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

## opzet
st.set_page_config(
    page_title="Eerste vaart",
    page_icon="❄️",
    layout="wide"
)
st.title('Oorspronkelijke Titanic Case')


# data inladen
@st.cache_data
def load_train_old():
    train_old = pd.read_csv('train.csv')
    train_old.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'], 
                   axis = 1, inplace = True)
    return train_old

train_old = load_train_old()

with st.sidebar:
    pagina = option_menu('Inhoudsopgave', ["Data verkenning", "Analyse", "Voorspellend model"], icons=['search', 'bar-chart-line', 'graph-up-arrow'], menu_icon='card-list')
#--------------------------------------------------------------------------------------------------------#

if pagina == 'Data verkenning':
    st.header('1. Data verkenning')
    st.write("De dataset:")
    st.dataframe(train_old.head())
    st.write("Beschrijving van de dataset:")
    st.write(train_old.describe())

    st.subheader('1.1 Data opvulling')
    st.write('De missende waardes komen voor in leeftijd en opstaplocatie:')
    st.write(train_old.isnull().sum())
    st.write('Deze waardes gaan we dan opvullen')

    st.subheader('1.2 Leeftijd opvullen')
    fig, ax = plt.subplots()
    train_old.Age.hist(bins=20)
    st.pyplot(fig)
    st.write('Leeftijd is opgevuld met de mediaan')

    st.subheader('1.3 Opstaplocatie opvullen')
    fig, ax = plt.subplots()
    train_old.Embarked.hist()
    st.pyplot(fig)
    st.write('De modus is gekozen om de opstaplocatie in te vullen')
    train_old['Age'].fillna(train_old['Age'].median(), inplace=True)
    train_old['Embarked'].fillna('S', inplace=True)


elif pagina == 'Analyse':
    st.header('2. Analyse')
    overleden_overleefd_gem = train_old['Survived'].mean()*100
    st.write(f'Wat was uberhaupt de overlevingskans?  \nOveral gezien is er een overlevingskans van: {np.round(overleden_overleefd_gem,2)}%')
        
    st.write('En wat heeft daar allemala invloed op?')
    train_old['Sex'] = train_old['Sex'].replace({'male':0, 'female':1})
    train_old['Embarked'] = train_old['Embarked'].replace({'S':1,'C':2,'Q':3})
    correlatie_matrix = train_old.corr()
    st.write(correlatie_matrix)

    st.write('Er zal worden gekeken naar geslacht, klasse, opstaplocatie, leeftijd en een mix hiervan tegenover geslacht')
    st.subheader('2.1 Geslacht')
    # Survival per geslacht
    Sex_survived_getal = train_old[train_old['Survived'] == 1].groupby('Sex').size()
    gender_mapping_reverse = {0: 'man', 1: 'vrouw'}
    Sex_survived_getal.index=Sex_survived_getal.index.map(gender_mapping_reverse)
    st.write('Overlevingsgetal: ', Sex_survived_getal)

    # kans survival per geslacht
    Sex_survived = (train_old.groupby('Sex')['Survived'].mean()*100)
    Sex_survived.index=Sex_survived.index.map(gender_mapping_reverse)
    st.write('Percentage overlevingskans: ',np.round(Sex_survived,2))

    # bar-chart opstellen
    fig, ax = plt.subplots()

    #Kleur geven
    colors = ['#1F3A5F', '#F06292']

    #grafiek maken
    bars_sex = plt.bar(Sex_survived.index,Sex_survived.values, color = colors)

    # opmaak
    plt.xlabel('Sex')
    plt.xticks(Sex_survived.index, ['Man','Vrouw'])
    plt.ylabel ('Percentage Overlevingkans')
    plt.ylim(0,100)
    plt.title('Overlevingskans per geslacht')

    # Annotaties dynamisch bovenop de balken plaatsen
    for bar in bars_sex:
        height = bar.get_height()  # Hoogte van de balk
        ax.annotate(f'{height:.2f}%',  # Percentage met 2 decimalen
                    xy=(bar.get_x() + bar.get_width() / 2, height),  # Plaatsing bovenop de balk
                    xytext=(0, 5),  # Kleine offset omhoog
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    st.pyplot(fig)
    st.write("Uit bovenstaande grafiek blijkt dat je beter een vrouw dan een man kon zijn aan boord van de titanic voor je overlevingskans te vergroten. 74% van de vrouwen heeft het overleefd, tegenover 18% van de mannen. Dat vrouwen en kinderen eerst mochten lijkt hieruit ook naar boven te komen.")

    st.subheader('2.2 Klasse')
    # Survival per klasse
    Pclass_survived_getal = train_old[train_old['Survived'] == 1].groupby('Pclass').size()
    Pclass_mapping_reverse = {1: '1e klas', 2: '2e klas', 3: '3e klas'}
    Pclass_survived_getal.index=Pclass_survived_getal.index.map(Pclass_mapping_reverse)
    st.write('Overlevingsgetal: ', Pclass_survived_getal)

    # kans survival per class
    Pclass_survived = train_old.groupby(['Pclass'])['Survived'].mean()*100
    Pclass_survived.index=Pclass_survived.index.map(Pclass_mapping_reverse)
    st.write('Overlevingskans per klasse: ', np.round(Pclass_survived,2))

    #Kleur geven
    colors = ['#FFD700', '#C0C0C0', '#CD7F32']

    # grafiek maken
    fig, ax = plt.subplots()
    bars_Pclass = ax.bar(Pclass_survived.index,Pclass_survived.values, color=colors)

    # opmaak
    plt.xlabel('Pclass')
    plt.xticks(Pclass_survived.index, ['1e klasse', '2e klasse', '3e klasse'])
    plt.ylabel ('Percentage Overlevingkans')
    plt.ylim(0,100)
    plt.title('Overlevingskans per klasse')

    # Annotaties dynamisch bovenop de balken plaatsen
    for bar in bars_Pclass:
        height = bar.get_height()  # Hoogte van de balk
        ax.annotate(f'{height:.2f}%',  # Percentage met 2 decimalen
                    xy=(bar.get_x() + bar.get_width() / 2, height),  # Plaatsing bovenop de balk
                    xytext=(0, 5),  # Kleine offset omhoog
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # zichtbaar maken
    st.pyplot(fig)
    st.write("In de eerste klas zijn de meest overlevende mensen 63% waar als je 3e klas zat je 24% overlevingskans had. De mensen in de eerste klas zitten waarschijnlijk meer boven in de boot, waardoor ze sneller bij de reddingsboot waren. Mogelijk is daarnaast door de kosten die ze voor het ticket betaald hebben ook de voorrang aan hun verleend door het personeel.")

    st.subheader('2.3 Opstaplocatie')
    # Survival per opstaplocatie
    Emb_survived_getal = train_old[train_old['Survived'] == 1].groupby('Embarked').size()
    Emb_mapping_reverse = {1: 'Southampton', 2: 'Charbourg', 3: 'Qeenstown'}
    Emb_survived_getal.index=Emb_survived_getal.index.map(Emb_mapping_reverse)
    st.write('Overlevingsgetal: ', Emb_survived_getal)

    # kans survival per opstaplocatie
    Emb_survived = train_old.groupby('Embarked')['Survived'].mean()*100
    Emb_survived.index=Emb_survived.index.map(Emb_mapping_reverse)
    st.write('Overlevingskans: ', np.round(Emb_survived,2))

    # bar-chart opstellen
    fig, ax = plt.subplots()

    #Kleur geven
    colors = ['#C8102E', '#0055A4', '#169B62']

    #grafiek maken
    bars_emb = plt.bar(Emb_survived.index,Emb_survived.values, color = colors)

    # opmaak
    plt.xlabel('Opstaplocatie')
    plt.xticks(Emb_survived.index, ['Southampton','Charbourg', 'Queenstown'])
    plt.ylabel ('Percentage Overlevingkans')
    plt.ylim(0,100)
    plt.title('Overlevingskans per opstaplocatie')

    # Annotaties dynamisch bovenop de balken plaatsen
    for bar in bars_emb:
        height = bar.get_height()  # Hoogte van de balk
        ax.annotate(f'{height:.2f}%',  # Percentage met 2 decimalen
                    xy=(bar.get_x() + bar.get_width() / 2, height),  # Plaatsing bovenop de balk
                    xytext=(0, 5),  # Kleine offset omhoog
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')


    # zichtbaar maken
    st.pyplot(fig)
    st.write("De opstaplocatie maakt nog weinig verschil. Opstappen op Cherbourg lijkt de overlevingskans te vergroten tegenover Queenstown en Southampton.")

    st.subheader('2.4 Leeftijd')
    # Functie om leeftijd in categorieën in te delen
    def leeftijdscategorie(leeftijd):
        if leeftijd < 16:
            return '1 - Kind'
        elif 16 <= leeftijd <= 34:
            return '2 - Jonge Volwassene'
        elif 35 <= leeftijd <= 49:
            return '3 - Volwassene'
        elif 50 <= leeftijd <= 64:
            return '4 - Middelbare Leeftijd'
        else:
            return '5 - Oudere Volwassene'

    # DataFrame maken en leeftijdscategorieën toewijzen
    df = pd.DataFrame({'Leeftijd': train_old['Age']})
    train_old['Leeftijdscategorie'] = df['Leeftijd'].apply(leeftijdscategorie)

    # Survival per Leeftijdscategorie
    Age_survived_getal = train_old[train_old['Survived'] == 1].groupby('Leeftijdscategorie').size()
    st.write('Overlevingsgetal: ', Age_survived_getal)

    #  kans survival per Leeftijdscategorie
    Age_survived = train_old.groupby('Leeftijdscategorie')['Survived'].mean()*100
    st.write('Overlevingskans per leeftijdscategorie',np.round(Age_survived,2))

    # Bar-chart opstellen
    fig, ax = plt.subplots()

    # Kleur per leeftijdscategorie (van licht naar donker)
    colors = ['#FFE599', 'orange', 'chocolate', 'sienna', 'saddlebrown']

    # Balken tekenen
    bars_age = ax.bar(Age_survived.index, Age_survived.values, color=colors)

    # Annotaties dynamisch bovenop de balken plaatsen
    for bar in bars_age:
        height = bar.get_height()  # Hoogte van de balk
        ax.annotate(f'{height:.2f}%',  # Percentage met 2 decimalen
                    xy=(bar.get_x() + bar.get_width() / 2, height),  # Plaatsing bovenop de balk
                    xytext=(0, 5),  # Kleine offset omhoog
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Opmaak
    plt.xlabel('Leeftijdscategorie')
    plt.xticks(rotation=45)
    plt.ylabel('Percentage Overlevingkans')
    plt.ylim(0, 100)
    plt.title('Overlevingskans per Leeftijdscategorie')

    # Grafiek tonen
    st.pyplot(fig)
    st.write("Kinderen lijken de grootste overlevingskans te hebben.")

    st.subheader('2.5 Geslacht over de vorige plotten heen')
    st.markdown('### 2.5.1 Geslacht over klasse')
    # kan survival per class - man
    Pclass_male = train_old.groupby(['Sex','Pclass'])['Survived'].mean().loc[0]*100
    Pclass_male.index=Pclass_male.index.map(Pclass_mapping_reverse)
    st.write('man', np.round(Pclass_male,2))

    # kan survival per class - man
    Pclass_female = train_old.groupby(['Sex','Pclass'])['Survived'].mean().loc[1]*100
    Pclass_female.index=Pclass_female.index.map(Pclass_mapping_reverse)
    st.write('vrouw', np.round(Pclass_female,2))

    # bar-chart opstellen
    fig, ax = plt.subplots()
    Pclass_labels = sorted(train_old['Pclass'].unique())

    #Kleur geven
    x=np.arange(len(Pclass_labels))
    width = 0.35

    #grafiek maken
    Bars_Pclass_m = ax.bar(x-width/2,Pclass_male,width,label='male', color = '#1F3A5F')
    Bars_Pclass_f = ax.bar(x+width/2,Pclass_female,width,label='female', color = '#F06292')

    # opmaak
    plt.xlabel('Pclass')
    plt.xticks(x)
    plt.ylabel ('Percentage Overlevingkans')
    plt.ylim(0,100)
    plt.title('Overlevingskans per klasse per geslacht')
    ax.set_xticklabels([f'Klasse {p}' for p in Pclass_labels]) 

    # **Dynamische annotaties toevoegen**
    for bar in Bars_Pclass_m + Bars_Pclass_f:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',  # Tekst met 2 decimalen
                    xy=(bar.get_x() + bar.get_width() / 2, height),  # Bovenop de balk
                    xytext=(0, 0),  # Kleine offset omhoog
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.legend()

    # zichtbaar maken
    st.pyplot(fig)
    st.write("Voor mannen was de overlevingskans in de eerste klas hoger dan in de 2e of 3e klas. In de eerste klas zitten gaf een man een hogere overlevingskans dan algemeen gezien, onder de mannen. Voor vrouwen is de overlevinskans in de 1e en 2e klas heel hoog. In de 3e klas zitten maakt gelijk de overlevingskans lager, ook lager dan de gemiddelde overlevingskans van de vrouw. Voor in het vervolg kan er specifiek gekeken worden naar mannen in de 1e (en 2e) klasse en vrouwen in de derde klasse, om dit allemaal te specificeren.")
    
    
    st.markdown('### 2.5.2 Geslacht over opstaplocatie')
    # kan survival per class - man
    Emb_male = train_old.groupby(['Sex','Pclass','Embarked'])['Survived'].mean().loc[(0,1)]*100
    Emb_male.index=Emb_male.index.map(Emb_mapping_reverse)
    st.write('Overlevingskans voor man bij opstaplocatie :', np.round(Emb_male,2))

    # kan survival per class - man
    Emb_female = train_old.groupby(['Sex','Pclass','Embarked'])['Survived'].mean().loc[(1,3)]*100
    Emb_female.index=Emb_female.index.map(Emb_mapping_reverse)
    st.write('Overlevingskans voor vrouw bij opstaplocatie :', np.round(Emb_female,2))

    # bar-chart opstellen
    fig, ax = plt.subplots()

    embarked_mapping = {2: 'Cherbourg', 3: 'Queenstown', 1: 'Southampton'}
    Emb_labels = sorted(train_old['Embarked'].dropna().unique())
    #Kleur geven
    x=np.arange(len(Emb_labels))
    width = 0.35

    #grafiek maken
    bars_male= ax.bar(x-width/2,Emb_male,width,label='male in 1e klas', color = '#1F3A5F')
    bars_female= ax.bar(x+width/2,Emb_female,width,label='female in 3e klas', color = '#F06292')

    # opmaak
    plt.xlabel('Opstaplocatie')
    plt.xticks(x)
    plt.ylabel ('Percentage Overlevingkans')
    plt.ylim(0,100)
    plt.title('Overlevingskans per opstaplocatie per geslacht')
    ax.set_xticklabels([embarked_mapping[p] for p in Emb_labels])

    # **Dynamische annotaties toevoegen**
    for bar in bars_male + bars_female:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',  # Tekst met 2 decimalen
                    xy=(bar.get_x() + bar.get_width() / 2, height),  # Bovenop de balk
                    xytext=(0, 5),  # Kleine offset omhoog
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.legend()

    # zichtbaar maken
    st.pyplot(fig)
    st.write("Hierboven blijkt dat vrouwen die op Southampton opstappen een minder grote overlevingskans hebben dan vrouwen die opstappen bij Queenstown en Cherbourg. Mannen lijken nogsteeds sneller te overleiden dan te overleven tussen de opstaplocaties. Nu is het wel zo dat mannen in de 1e klas die opstappen op queenstown sowieso lijken te overleiden, maar na even naar de data kijken blijkt dat er maar 1 man is opgestapt in Queenstown in de 1e klas en hij is overleden.")

    st.markdown("### 2.5.3 geslacht over mannen op de 1e en 2e klas per leeftijdscategorie")
    # kan survival per class - man
    Age_male = train_old.groupby(['Sex', 'Pclass','Leeftijdscategorie'])['Survived'].mean().loc[(0,1)]*100
    st.write('man_1', Age_male)

    # kan survival per class - man
    Age_male_2e = train_old.groupby(['Sex', 'Pclass', 'Leeftijdscategorie'])['Survived'].mean().loc[(0,2)]*100
    st.write('man_2', Age_male_2e)

    # bar-chart opstellen
    fig, ax = plt.subplots()

    Age_labels = sorted(train_old['Leeftijdscategorie'].dropna().unique())
    #Kleur geven
    x=np.arange(len(Age_labels))
    width = 0.45

    #grafiek maken
    bar_age_m1 = ax.bar(x-width/2,Age_male,width,label='male in 1e klas', color = '#1F3A5F')
    bar_age_m2 = ax.bar(x+width/2,Age_male_2e,width,label='male in 2e klas', color = 'lightblue')

    # opmaak ax1
    ax.set_xlabel('Leeftijdscategorie')
    ax.set_xticks(x)
    ax.set_ylabel ('Percentage Overlevingkans')
    ax.set_ylim(0,100)
    ax.set_title('Overlevingskans per leeftijdscategorie per geslacht (gefilterd op klasse)')
    ax.legend()


    # Annotaties dynamisch bovenop de balken plaatsen
    for bar in bar_age_m1 + bar_age_m2:
        height = bar.get_height()  # Hoogte van de balk
        ax.annotate(f'{height:.2f}%',  # Percentage met 2 decimalen
                    xy=(bar.get_x() + bar.get_width() / 2, height),  # Plaatsing bovenop de balk
                    xytext=(0, 0),  # Kleine offset omhoog
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    st.pyplot(fig)
    st.write("Mannen in de 1e klas onder de 16 overleven het. Daarnaast ook alle mannen onder de 16 in de 2e klas overleven het. Voor de rest geven de waardes geen grote overlevingskans weer.")

    st.subheader('2.6 ticketprijs op 1e klas')
    # Filter alleen passagiers met Pclass = 1
    df_first_class = train_old[train_old['Pclass'] == 1].copy()
    
    # Voeg een nieuwe kolom met de index toe aan df_first_class
    df_first_class['Index'] = df_first_class.index
    
    # Maak een scatterplot met seaborn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df_first_class,
        x='Index',
        y='Fare',
        hue='Survived',
        palette={0: 'red', 1: 'green'}  
    )
    
    # Voeg titels en labels toe
    plt.title('Scatterplot van Fare vs Index (Alleen Pclass = 1)')
    plt.xlabel('Index')
    plt.ylabel('Fare')
    
    # Pas de legenda aan met correcte labels
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Overleden', 'Overleefd']  
    plt.legend(handles, labels, title='Survived', loc='upper right')
    
    # Toon de plot
    st.pyplot(fig)


elif pagina == 'Voorspellend model':
    st.header('Voorspelling')
    st.write('Uit de analyse hebben we gekozen voor de volgende punten:  \n* Vrouwen uit de 1e en 2e klas.  \n* Vrouwen uit de 3e klas met opstaplocatie Queenstown of C.  \n* Mannen uit de 1e of 2e klas onder de 16.  \n* Mannen uit de 1e klas met een betaling hoger dan 400')
    st.write('Met deze keuzes kwam de kaggle score uit op: 78,47%')

