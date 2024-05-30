# --------------- CHARGEMENT DES BIBLIOTHÈQUES ---------------
# from bs4 import BeautifulSoup as bs
import time
import datetime as dt
import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import streamlit as st
import io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import plotly.express as px
from urllib.request import urlopen
import json
# ---------------------------------------------------------------

# ------------ INITIALISATION DES VARIABLES GLOBALES ------------
file = "cleaned_reviews-vinted-truspilot.csv"
df_cleaned = pd.read_csv(file, sep= ",", index_col= "id_avis", low_memory= False)
df_cleaned["date_heure_avis"] = pd.to_datetime(df_cleaned["date_heure_avis"])
df_cleaned["date_experience"] = pd.to_datetime(df_cleaned["date_experience"])

file_df_world_map = "./data/world-administrative-boundaries/world-administrative-boundaries.shp"
df_world_map = gpd.read_file(file_df_world_map)

# Récupération de la carte
with open('./data/world_map.json') as response :
    json_countries = json.load(response)

# ----------------- Titre   --------------------------------------------
st.title("")
title = ":orange[Leboncoin] vs :green[Vinted]"
col1, col2, col3 = st.columns([1, 2, 1])
col2.image("./assets/vinted-logo.svg", width= 250)


# ----------------  Sidebar   ------------------------------------------
pages = ["Le projet", "Obtention des données", "Nettoyage du jeu de données", "Quelques visualisations", 
         "Préparation des données", "Machine learning", "Conclusion"]
auteurs = """
        Auteur1  
        Auteur2  
        Auteur3  
        Auteur4
        """
st.sidebar.title("Vinted")
sidebar = st.sidebar.radio("Sélectionnez une partie :", pages)
st.sidebar.write("---")
st.sidebar.subheader("Auteurs")             
st.sidebar.write(auteurs)

# ----------------- Page 0 "Le Projet" -----------------------------------
if sidebar == pages[0]:
    st.header(pages[0])
    presentation = """
    Ce projet a été réalisés dans le cadre de notre formation en data science délivrée par 
    [Datascientest](https://datascientest.com/formation-data-scientist). 
    Nous nous sommes intéressés à Vinted, entreprise spécialisée dans la 
    publication de petites annonces en ligne pour la vente de biens de particulier à particulier.\n
    Vinted.fr (entreprise lithuanienne), avec 16 millions de visiteurs uniques mensuel, est le 4ème site e-commerce
    le plus consulté en France.\n
    """

    objectif = """
    L'objectif du projet est de *PRÉDIRE* la note de satisfaction que laisserait un client à propos de ces deux entreprises 
    sur une plateforme d'avis en ligne, en l'occurence Truspilot.
    \nCe streamlit présente notre démarche pour mener à bien ce projet, depuis la construction du jeu de données jusqu'à 
    la mise en place d'algorithmes de Machine Learning pour prédire les résultats des notes. N'hésitez pas à tester.
    """

    st.subheader("1. Présentation")
    st.write(presentation)
    st.subheader("2. Objectif")
    st.write(objectif)

# ----------------- Page 1 "Obtention des données" -----------------------
if sidebar == pages[1]:
    st.header(pages[1])

    source = """ 
    Les données ont été collectées à partir des derniers avis clients déposés sur Truspilot, et concernant
    [toutes les déclinaisons locales de Vinted à travers le monde](https://fr.trustpilot.com/search?query=vinted)

    En effet, pour prédire la note d'un client, il est nécessaire d'identifier les entités importantes d’un avis : la note, la localisation, 
    le nom de l'entreprise, la date, ....  
    Mais aussi le commentaire laissé par le client afin d'en extraire le propos global : article défectueux ou conforme? 
    livraison correcte ou problématique? sentiment? ...\n
    Ne disposant pas d'une base consolidées avec ces informations, il est apparu nécessaire d'aller collecter ces données directement depuis
    une plateforme d'avis clients.  
    Nous avons fait le choix de **Trustpilot**.
    """

    webscrapping = """
    Grâce à la bibliothèque :orange[***BeautifulSoup***] de python, un programme a été rédigé afin de collecter les données 
    en "webscrappant" le site Trustpilot.
    Afin d'avoir un jeu de données  consistant, mais aussi pour avoir des avis étalés sur plusieurs mois, le code mis en place 
    a permis de récupérer la totalité des avis publiés pour Vinted.  
    C'est-à-dire un peu plus de 100k avis à la date d'exécution du code. Ce qui constitue notre jeu de données de départ.
    
    ***Note** : Plus loin dans le projet, nous nous servirons de la bibliothèque :orange[**Selenium**] pour nous connecter à Google traduction.*  
    *Puis pour chaque avis : détecter, récupérer sa langue de rédaction et traduire automatiquement son titre/commentaire
    sauf s'ils sont déjà rédigés en français.*
    """
    st.subheader("1. Sources de données")
    st.write(source)
    st.subheader("2. Web scrapping")
    st.write(webscrapping)
    st.write("---")

# ----------------- Page 2 "Nettoyage du jeu de données" ------------------------------
if sidebar == pages[2]:
    st.header(pages[2])
    # --------------- Affichage de l'introduction
    st.write(""" 
            Le jeu de données brut récolté d'internet par le webscrapping nécessite quelques 
            transformations afin d'être exploitable pour effectuer les premières visualisations.\n
            Ces transformations impliquent notamment la mise à jour des types, des formattages de date,
            la suppression des doublons ou encore la gestion des NA.\n
            Les sections suivantes détaillent les opérations qui ont été effectuées sur le dataset.
            """)
    st.write("---")

    if st.toggle("Infos de la table", key= 5):
            # Affichage des infos de la table1
        buffer = io.StringIO()
        df_cleaned.info(buf=buffer)
        s1 = buffer.getvalue()
        st.text(s1)
    
    st.write("---")

    # --------------- Suppression des doublons
    st.subheader("1. Suppression des doublons")
    st.write("""Tous les doublons ont été supprimés
             """)

    if st.toggle(":green[Afficher le code]", key= 80):
        code ="""
            # Suppression des doublons
            df = df.drop_duplicates()
            """
        st.code(code)

    # --------------- Suppression des doublons
    st.subheader("2. Harmonisation du nom de l'entreprise")
    st.write("""Nommer les entreprises sous le même format : "VINTED.(extension du pays)"
             """)

    if st.toggle(":green[Afficher le code]", key= 988):
        code ="""
            # Harmonisation du nom de l'entreprise
            df["entreprise"] = df["entreprise"].apply(lambda x: "VINTED." + x.split("VINTED.")[-1])
            """
        st.code(code)

    # --------------- Gestion des NA
    st.subheader("2. Gestion des NA")
    st.write("""
            Il y a 7 avis (sur 100k+) sans "pays". On supprime ces lignes.  
            On remplace les noms, titres, commentaires et commentaire_vinted en NA par "".
            """)

    if st.toggle(":green[Afficher le code]", key= 839):
        code ="""
            # Il y a 7 avis (sur 100k+) sans "pays". On supprime ces lignes
            df = df.dropna(subset= "pays")
            # On remplace les noms, commentaires et commentaire_vinted en NA par ""
            df["nom"] = df["nom"].fillna("")
            df["commentaire"] = df["commentaire"].fillna("")
            df["commentaire_vinted"] = df["commentaire_vinted"].fillna("")
            """
        st.code(code)
    
    # --------------- Numérisation de la colonne avis_verified
    st.subheader("3. Numérisation de la colonne avis_verified")
    st.write("""
            Si un avis est vérifié cet attribut prendra la valeur 1, sinon 0.
            """)

    if st.toggle(":green[Afficher le code]", key= 8):
        code ="""
            # Numérisation de la colonne "avis_verified" (1 si l'avis est "Vérifié", 0 sinon)
            df["avis_verified"] = df["avis_verified"].apply(lambda x: 1 if x == "Vérifié" else 0)
            """
        st.code(code)
    
    # --------------- Ajout d'une colonne qui donne une valeur numérique au pays
    st.subheader("4. Numérisation de la colonne pays")
    st.write("""
            Ajout d'une colonne qui donne une valeur numérique au pays.
            """)

    if st.toggle(":green[Afficher le code]", key= 83):
        code ="""
            # Ajout d'une colonne qui donne une valeur numérique au pays
            liste = df["pays"].unique()
            liste = np.sort(liste)
            def func_get_pays_num(_pays):
                global liste
                pays_num = 0
                for i in range(0,len(liste)):
                    if _pays == liste[i]:
                        pays_num = i+1
                        break
                return pays_num
            pays_num = df["pays"].apply(func_get_pays_num)
            df.insert(4, "pays_num", pays_num)
            """
        st.code(code)

    # Mise à jour des attributs
    st.subheader("4. Mise à jour des attributs")
    # *** Mise à jour de la date d'expérience
    st.write("#### a. Mise à jour de l'attribut de la date d'expérience")
    st.write("""
            Les modalités de la colonne :orange[**date_experience**] sont déjà sous la forme **%Y-%m-%d**, 
            mais en format string. On leur applique simplement une conversion en date.
            """)

    if st.toggle(":green[Afficher le code]", key= 9):
        code = """
            # Mise à jour du type pour l'attribut "date_experience"
            df["date_experience"] = pd.to_datetime(df["date_experience"])
            """
        st.code(code)

    # *** Mise à jour de la date de l'avis    
    st.write("#### b. Mise à jour de l'attribut de la date de l'avis")
    st.write("""
            Les modalités de la colonne :orange[**date_heure_avis**] doivent être sous la forme **%Y-%m-%d %H:%M:%S**. 
            Cependant, certaines colonnes apparaissent sous une forme différente.\n
            Création et application d'une fonction pour ajuster les string problématiques, avant de les formatter en datetime sur les deux df
            """)
    
    if st.toggle(":green[Afficher le code]", key= 10):
        code = """
            # Fonction pour convertir en datetime
            def func_to_datetime(_date):
                _date = _date.split(".")[0]
                _date = _date.replace("T", " ")
                _date = dt.datetime.strptime(_date, "%Y-%m-%d %H:%M:%S")
                return _date

            # Mise à jour du type pour l'attribut "date_heure_avis"
            df["date_heure_avis"] = df["date_heure_avis"].apply(func_to_datetime)
            """
        st.code(code)

        # *** Mise à jour des autres attributs    
    st.write("#### c. Mise à jour des autres attributs")
    st.write("""
            Utilisation d'un dictionnaire pour mettre à jour les types des autres attributs.
            """)
    
    if st.toggle(":green[Afficher le code]", key= 11):
        code = """
            # Mise à jour du type pour les autres attributs
            dict_types = {"entreprise": "str", "nom": "str", "pays": "str", "note": "int", "nombre_total_avis": "int", "titre": "str", 
                          "commentaire": "str", "commentaire_vinted": "str"}
            df = df.astype(dict_types)
            """
        st.code(code)

    # --------------- Ajout de la colonne vinted_commented    
    st.subheader("5. Ajout de la colonne vinted_commented")
    st.write("""
            Ajout d'une colonne, permettant de savoir si vinted a répondu à un avis (valeur 1) ou pas (valeur 0)
            """)

    if st.toggle(":green[Afficher le code]", key= 13):
        code = """
            # Ajout d'une colonne pour indiquer si vinted a répondu à l'avis ou pas
            df["vinted_commented"] = df["commentaire_vinted"].apply(lambda x: 0 if x == "" else 1)
            """
        st.code(code)

    # --------------- Ajout des colonnes longueur_titre/longueur_commentaire    
    st.subheader("6. Ajout des colonnes longueur_titre/longueur_commentaire")
    st.write("""
            Rajout d'une colonne :orange[**longueur_titre**] pour compter le nombre de mots qu'il y a dans le titre de l'avis.\n
            Idem pour le rajout de la colonne :orange[**longueur_commentaire**].
            """)

    if st.toggle(":green[Afficher le code]", key= 153):
        code = """
            # Récupération du nombre de mots dans un texte
            def func_count_words(_text):
                _car = ("’", "'", "!", ",", "?", ";", ".", ":", "/", "+", "=", "\\n", "- ", " -", "(", ")", "[", "]", "{", "}", "*", "<", ">")
                for car in _car:
                    _text = _text.replace(car, " ")
                return len(_text.split())

            # Ajout de la colonne longueur du titre
            df["longueur_titre"] = df["titre"].apply(func_count_words)

            # Ajout de la colonne longueur du commentaire
            df["longueur_commentaire"] = df["commentaire"].apply(func_count_words)
            """
        st.code(code)

    # --------------- Ajout des colonnes semaine/jour/mois/année pour les dates de l'expérience
    ajout_col_date_exp = """
    Décomposition de l'attribut :orange[**date_experience**] afin de récupérer la semaine, le jour de semaine,
    le jour de mois, le mois et l'année. 
    """
    
    st.subheader("7. Ajout des colonnes semaine/jour/mois/année à partir de la date de l'expérience")
    st.write(ajout_col_date_exp)

    if st.toggle(":green[Afficher le code]", key= 14):
        code = """
        # Ajout de la colonne semaine_experience
        df["semaine_experience"] = df["date_experience"].apply(lambda date: date.week)

        # Ajout de la colonne jour de semaine de l'expérience
        df["jour_semaine_experience"] = df["date_experience"].apply(lambda date: date.day_of_week)

        # Ajout de la colonne jour du mois de l'expérience
        df["jour_experience"] = df["date_experience"].apply(lambda date: date.day)

        # Ajout de la colonne mois de l'expérience
        df["mois_experience"] = df["date_experience"].apply(lambda date: date.month)

        # Ajout de la colonne année de l'expérience
        df["annee_experience"] = df["date_experience"].apply(lambda date: date.year)
        """
        st.code(code)

    # --------------- Ajout des colonnes semaine/heure/jour/mois/année pour les dates de l'avis
    st.subheader("7. Ajout des colonnes semaine/heure/jour/mois/année pour les dates de l'avis")
    st.write("""
             Décomposition de l'attribut :orange[**date_heure_avis**] afin de récupérer l'heure, la semaine, le jour de semaine, 
             le jour de mois, le mois et l'année.
            """)

    if st.toggle(":green[Afficher le code]", key= 15):
        code = """
    # Ajout de la colonne semaine de l'avis
    df["semaine_avis"] = df["date_heure_avis"].apply(lambda date: date.week)

    # Ajout de la colonne jour de semaine de l'avis
    df["jour_semaine_avis"] = df["date_heure_avis"].apply(lambda date: date.day_of_week)

    # Ajout de la colonne jour du mois de l'avis
    df["jour_mois_avis"] = df["date_heure_avis"].apply(lambda date: date.day)

    # Ajout de la colonne mois de l'avis
    df["mois_avis"] = df["date_heure_avis"].apply(lambda date: date.month)

    # Ajout de la colonne année de l'avis
    df["annee_avis"] = df["date_heure_avis"].apply(lambda date: date.year)

    # Ajout de la colonne heure de l'avis
    df["heure_avis"] = df["date_heure_avis"].apply(lambda date: date.hour)
    """
        st.code(code)

    # --------------- Ajout des colonnes langue de l'avis/titre fr/commentaire fr
    st.subheader("9. Ajout des colonnes langue de l'avis/titre fr/commentaire fr")
    st.write("""
             La plupart des avis ont été rédigés par des clients dans une autre langue. La dernière étape dans ce 1er travail 
             de revue de dataset consiste à rajouter :  
             - Une colonne avec la valeur de la langue dans laquelle l'avis a été rédigé
             - Une colonne avec la traduction du commentaire en français\n
             
             Pour récupérer les traductions, nous nous servons de la bibliothèque :orange[**Selenium**] de python pour nous connecter
             à Google translate afin de :  
             - Insérer chaque titre/commentaire de la table dans le champ du texte à traduire de la page Google  
             - Récupérer la valeur de la "Langue Détectée" par Google et la rajouter dans la ligne correspondante du dataset  
             - Récupérer la valeur de la traduction et la rajouter dans la ligne correspondante du dataset (si le titre/commentaire
             n'est pas rédigé en français)
             """)
    
    # --------------- Conclusion de la page
    st.subheader("10. État des lieux")
    st.write(f"Le jeu de données nettoyé et réajusté comporte {df_cleaned.shape[0]} entrées et {df_cleaned.shape[1]} attributs.")
    st.write("""
             Ces premières transformations ont permis d'obtenir un dataset de départ propre, et exploitable pour commencer 
             à faire quelques visualisations.\n
             Dans la suite du projet, de nouvelles transformations seront apportées pour discrétiser et dichotomiser certains attributs 
             afin de créer des features adaptées aux calculs de Machine Learning.
             """)
    st.write("---")

    if st.toggle("Infos du dataset après nettoyage et réajustement", key= 17):
        # Affichage des infos
        buffer = io.StringIO()
        df_cleaned.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    if st.toggle("Lignes du dataset après nettoyage et réajustement", key= 18):
        number = st.number_input(":green[Nombre de lignes à afficher :]", min_value=1, max_value= len(df_cleaned), value= 5, key= 21)
        st.dataframe(df_cleaned.head(number))

# ----------------- Page 3 "Quelques datavisualisations" -----------------
if sidebar == pages[3]:
    st.header(pages[3])
    # --------------- Affichage de l'introduction
    st.write("""
    Les étapes précédentes ont permis d'avoir un premier dataset nettoyé, que nous pouvons utiliser pour effectuer quelques visualisations.
    """)
    st.write("---")

    # --------------------------------------------------------------------
    # ********************* Table du jeu de données **********************
    # --------------------------------------------------------------------

    # Texte d'aide sur le filtre
    h = """
    :red-background[- FILTRE QUI PERMET D'AFFICHER UN NOMBRE MAXIMAL D'AVIS.]  
    :red-background[- Utilisez la loupe :mag: sur le tableau pour rechercher une valeur particulière dans le dataset.]  
    :red-background[- Cliquez sur l'entête du tableau pour trier en fonction d'un attribut.]  
    :red-background[- Il est également possible d'élargir les colonnes du tableau.]
        """
    
    # Séparation en colonne pour les filtres
    st.subheader("1. Table du jeu de données")
    st.write("Filtrez la table pour sélectionner les attributs, le nombre de lignes à afficher, les notes, la date de publication, ou la période de publication.")
    

    # Création des filtres
    with st.expander("###### **Filtrer par :**", expanded= True):
        colonnes = st.multiselect(":green[Attributs =]", df_cleaned.columns, placeholder= "")
        attributs = list(df_cleaned.columns) if colonnes == list() else colonnes

        col1, col2, col3 = st.columns(3)
        nb_avis = col1.number_input(":green[Nombre d'avis max à afficher =]", min_value= 1,
                                    max_value= len(df_cleaned), value= len(df_cleaned), help= h)
        notes = col2.multiselect(":green[Note =]", sorted(df_cleaned["note"].unique(), reverse= True), placeholder= "...")
        notes = list(df_cleaned["note"].unique()) if notes == list() else notes

        min_date = df_cleaned["date_heure_avis"].min()
        max_date = df_cleaned["date_heure_avis"].max()
        dates = col3.date_input(":green[Date ou période des avis =]",
                        value= (min_date, max_date),
                        min_value= min_date, max_value= max_date,
                        format="YYYY-MM-DD")
    try:
        date_debut = dt.datetime.combine(min(dates), dt.datetime.min.time())
        date_fin = dt.datetime.combine(max(dates), dt.datetime.max.time())
    except:
        date_debut = dt.datetime.combine(min_date, dt.datetime.min.time())
        date_fin = dt.datetime.combine(max_date, dt.datetime.max.time())

    # Création de la table
    df_shown = df_cleaned.copy()
    df_shown = df_shown[(df_shown["date_heure_avis"].between(date_debut, date_fin, inclusive= "both"))]
    df_shown = df_shown[(df_shown["note"].isin(notes))]
    df_shown = df_shown[attributs]
    df_shown = df_shown.head(nb_avis)

    # Impression des résultats
    kol1, kol2 = st.columns([2/3, 1/3])
    texte1 = f"**{len(df_shown)}** avis récupérés à partir des plus récents."
    if str(date_fin).split(' ')[0] != str(date_debut).split(' ')[0]:
        texte2 = f"Publiés entre le **{str(date_fin).split(' ')[0]}** et le **{str(date_debut).split(' ')[0]}**."
    else:
        texte2 = f"Publiés le **{str(date_fin).split(' ')[0]}**."
    texte3 = f"Avec une note de **{str(sorted(notes, reverse= True)).replace('[', '').replace(']', '').replace(',', ' ou')}** étoiles.  "
    kol1.write(f"{texte1}  \n{texte2}  \n{texte3}")

    # Bouton de téléchargement
    @st.cache_data # IMPORTANT: mise en cache de la conversion pour éviter l'exécution à chaque rerun
    def convert_df(_df):
        return _df.to_csv().encode("utf-8")

    csv = convert_df(df_shown)

    kol2.download_button(
        label= "Télécharger le résultat (.csv)",
        type= "primary",
        data= csv,
        file_name= f"{len(df_shown)}_avis_dataset.csv",
        mime= "text/csv",
        use_container_width= True
    )
    
    # Affichage de la table
    st.dataframe(df_shown, column_config= 
                 {"_index": st.column_config.NumberColumn(format= "%d"),
                  "annee_experience": st.column_config.NumberColumn(format= "%d"),
                  "annee_avis": st.column_config.NumberColumn(format= "%d"),
                  "date expérience": st.column_config.DateColumn()})
    
    st.write("---")

    # --------------------------------------------------------------------
    # *************** Répartition du nombre total des notes **************
    # --------------------------------------------------------------------
    st.subheader("2. Répartition du nombre total d'avis")
    st.write("""
             Afficher la répartition du nombre d'avis en fonction de l'entreprise, de la note, de si l'avis est vérifié,  
             Ou encore en fonction de l'heure, du jour, du mois et de l'année de publication.
             """)

    # Création des filtres
    liste_filtres = ["Entreprise", "Note", "Avis vérifié", "Heure de publication", "Jour de publication en semaine", 
                     "Jour de publication dans le mois", "Mois de publication", "Année de publication"]
    dict_note = {"Entreprise": "entreprise", "Note": "note", "Avis vérifié": "avis_verified", 
                 "Heure de publication": "heure_avis", "Jour de publication en semaine": "jour_semaine_avis", 
                 "Jour de publication dans le mois": "jour_mois_avis", "Mois de publication": "mois_avis", "Année de publication": "anne_avis"}
    
    with st.container(border= True):
        type_filtre = st.selectbox("###### **Nombre d'avis en fonction de :**", options= liste_filtres)
    filtre = dict_note[type_filtre]

    # Nombre d'avis en fonction de la valeur de "valeur de la note", "heure_avis", "jour_semaine_avis", "jour_mois_avis", "mois_avis", "annee_avis"
    if filtre in ["note", "heure_avis", "jour_semaine_avis", "jour_mois_avis", "mois_avis", "annee_avis"]:
        fig, ax = plt.subplots(figsize= (10, 5))
        sns.countplot(x= df_cleaned[filtre], hue= df_cleaned["entreprise"], ax= ax)
        plt.title(f"Nombre d'avis par : {type_filtre}")
        plt.xlabel(type_filtre.capitalize() + (f" (nombre d'étoiles)" if filtre == "note" else ""))
        plt.ylabel("Nombre d'avis")
        plt.legend(title= "Entreprise")

        # Affichage des totaux sur les barres
        for container in ax.containers:
            if filtre not in ["jour_mois_avis", "mois_avis", "heure_avis"]:
                ax.bar_label(container)

        # Affichage des noms de jours si filtre sur le jour
        if (filtre == "jour_semaine_avis") :
            dict_jour = {0: "Dimanche", 1: "Lundi", 2: "Mardi", 3: "Mercredi", 4: "Jeudi", 5: "Vendredi", 6: "Samedi"}
            plt.xticks([0, 1, 2, 3, 4, 5, 6], ["Dimanche", "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"])

        st.pyplot(fig)

    st.write("---")

    # --------------------------------------------------------------------
    # ************* Corrélation entre les variables numériques ***********
    # --------------------------------------------------------------------
    st.subheader("3. Corrélation entre les variables numériques")
    st.write("""  
             Filtrer la matrice en choisissant des variables numériques et/ou temporelles.  
             Il possible d'afficher la corrélation entre les variables d'une seule entreprise, ou des deux.  
             Il est également possible de n'affichez que les variables fortement corrélées entre elles.  
             La corrélation est calculée avec la 
             [*méthode de Bravais-Pearson*](https://fr.wikipedia.org/wiki/Corr%C3%A9lation_(statistiques)#D%C3%A9finition).
             """)

    # Création des filtres
    quant_vars = ["note", "nombre_total_avis", "longueur_titre", "longueur_commentaire"]     # Variables quantitatives
    temp_vars = ["jour_semaine_experience", "semaine_experience", "jour_experience", "mois_experience", "annee_experience", 
                 "heure_avis", "jour_semaine_avis", "semaine_avis", "jour_mois_avis", "mois_avis", "annee_avis"]  # Variables temporelles
    
    with st.expander("###### **Filtrer par :**", expanded= False):
        sel1 = st.multiselect(":green[Variables quantitatives =]", quant_vars, default= quant_vars, placeholder= "...")
        sel2 = st.multiselect(":green[Variables temporelles =]", temp_vars, default= temp_vars, placeholder= "...")

    # Affichage du graphique
    df_corr = df_cleaned.copy()
    df_corr = df_corr[["pays_num"] + sel1 + sel2]
    df_corr = df_corr.corr()
    try:
        fig, ax = plt.subplots(figsize= (11, 5))
        if st.checkbox("**Montrer les variables en corrélation forte (|corr| > 0.5)**"):
            df_corr = df_corr[(df_corr <= -0.5) | (df_corr >= 0.5)]
            sns.heatmap(df_corr, cmap = "seismic", ax= ax, annot= True, fmt=".2f", vmin= -1, vmax= 1, linewidths= 0.1, linecolor= "lightgrey")
        else:
            sns.heatmap(df_corr, cmap = "seismic", ax= ax, annot= True, fmt=".2f", vmin= -1, vmax= 1, linewidths= 0.1, linecolor= "lightgrey")
        st.pyplot(fig)
    except:
        st.write(":red[Oops... :sweat: veuillez choisir au moins une variable pour afficher la matrice de corrélation.]")
        
    st.write("---")

    # --------------------------------------------------------------------
    # ********************** Distribution des données ********************
    # --------------------------------------------------------------------
    st.subheader("4. Distribution des données")
    st.write("""  
             Filtrer le graphique en choisissant deux variables pour lesquelles afficher la distribution.  
             Il possible d'afficher la distribution pour une seule entreprise, ou les deux.  
             Il est également possible de masquer les valeurs extrêmes.
             """)

    # Création des filtres
    quant_vars = ["note", "nombre_total_avis", "longueur_titre", "longueur_commentaire"]     # Variables quantitatives
    temp_vars = ["jour_semaine_experience", "semaine_experience", "jour_experience", "mois_experience", "annee_experience", 
                 "heure_avis", "jour_semaine_avis", "semaine_avis", "jour_mois_avis", "mois_avis", "annee_avis"]  # Variables temporelles
    
    sel1 = st.selectbox(":green[x =]", options= ["note", "jour_semaine_experience", "annee_experience",
                                            "jour_semaine_avis", "annee_avis"], index= 0, placeholder= "")
    sel2 = st.selectbox(":green[y =]", options= quant_vars + temp_vars, index= 3, placeholder= "")

    # Affichage de la distribution des avis
    df_cleaned_pays_dist = df_cleaned.copy()
    if (sel1 in (quant_vars + temp_vars)) and (sel2 in (quant_vars + temp_vars)):
        fig, ax = plt.subplots(figsize= (10, 5))

        # Masque des valeurs extrêmes
        if not st.checkbox("**Afficher les valeurs extrêmes**") :
            sns.boxplot(x= df_cleaned_pays_dist[sel1], y= df_cleaned_pays_dist[sel2], showfliers = False,
                        hue= df_cleaned_pays_dist["entreprise"])
        else:
            sns.boxplot(x= df_cleaned_pays_dist[sel1], y= df_cleaned_pays_dist[sel2], showfliers = True,
                        hue= df_cleaned_pays_dist["entreprise"])
        plt.title(f"Distribution entre : {sel1.upper()} et {sel2.upper()}")
        plt.xlabel(sel1)
        plt.ylabel(sel2)
        plt.legend(title= "Entreprise")

        # Affichage des noms de jours si filtre sur le jour
        if (sel1 == "jour_semaine_avis") or (sel1 == "jour_semaine_experience"):
            dict_jour = {0: "Dimanche", 1: "Lundi", 2: "Mardi", 3: "Mercredi", 4: "Jeudi", 5: "Vendredi", 6: "Samedi"}
            plt.xticks([0, 1, 2, 3, 4, 5, 6], ["Dimanche", "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"])
        if (sel2 == "jour_semaine_avis") or (sel2 == "jour_semaine_experience"):
            dict_jour = {0: "Dimanche", 1: "Lundi", 2: "Mardi", 3: "Mercredi", 4: "Jeudi", 5: "Vendredi", 6: "Samedi"}
            plt.yticks([0, 1, 2, 3, 4, 5, 6], ["Dimanche", "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"])

        st.pyplot(fig)
    st.write("---")

    # --------------------------------------------------------------------
    # ********************* Répartition géographique *********************
    # --------------------------------------------------------------------
    st.subheader("5. Répartition géographique")
    h = """
    (Déplacez la souris sur la carte pour voir le nombre d'avis pour un pays.  
    Zoom+ pour les détails sur une région.  
    Agrandir pour afficher la carte en plein écran.)"""
    
    st.write(f"La carte suivante affiche la répartion de certaines données en fonction des pays.")

    # Dictionnaire de correspondance
    dict_continent = {"Monde": "world", "Afrique": "africa", "Amérique du Nord": "north america", 
                   "Amérique du Sud": "south america", "Asie": "asia", "Europe": "europe"}
    
    # Création des filtres
    continent = st.selectbox(":green[Continent =]",
                        options= list(dict_continent.keys()),
                        index= 0, placeholder= "")
    


    metrique = kol2.selectbox(":green[Métrique à afficher =]", options= ["Nombre d'avis", "Moyenne"], index= 0)
    metrique = "Nombre d'avis" if len(metrique) == 0 else metrique
    my_func = {"note": "mean"}

    # ---------------------------------------------
    # ------ Nombre et moyenne d'avis par pays (global) ------
    # ---------------------------------------------
    # Préparation des données à dessiner pour les deux entreprise
    if metrique == "Moyenne":
        df_metrique = df_cleaned.groupby("pays").agg(my_func).reset_index().rename(columns= {"note": "moy"})
        df_metrique = round(df_metrique, 2)
    else:
        df_metrique = df_cleaned["pays"].value_counts().reset_index()


    df_metrique = df_metrique.merge(df_world_map, how= "inner", left_on= "pays", right_on= "iso_3166_1_")
    df_metrique = df_metrique[["pays", df_metrique.columns[1], "french_shor", "continent", "region"]].rename(columns= {"french_shor": "name_fr"})
    df_metrique["continent"] = df_metrique["continent"].apply(lambda x: x.lower())
    df_metrique["region"] = df_metrique["region"].apply(lambda x: x.lower())

    # Classement de certains pays dans les bons continents
    for i in range(0, len(df_metrique)):
        df_metrique.iloc[i, 3] = df_metrique.iloc[i, 4].lower() if df_metrique.iloc[i, 3] == "americas" else df_metrique.iloc[i, 3].lower()
        df_metrique.iloc[i, 3] = "north america" if df_metrique.iloc[i, 4] == "northern america" else df_metrique.iloc[i, 3]
        df_metrique.iloc[i, 3] = "north america" if df_metrique.iloc[i, 4] == "central america" else df_metrique.iloc[i, 3]
        df_metrique.iloc[i, 3] = "south america" if df_metrique.iloc[i, 4] == "caribbean" else df_metrique.iloc[i, 3]
        df_metrique.iloc[i, 3] = "asia" if df_metrique.iloc[i, 3] == "oceania" else df_metrique.iloc[i, 3]

    # Filtre du dataset sur la sélection utilisateur
    st.text(f'{h}')
    df_metrique = df_metrique[df_metrique["continent"] == dict_continent[continent]] if dict_continent[continent] != "world" else df_metrique


    # Tracé de la carte et tableau des résultats
    fig = px.choropleth(df_metrique, geojson= json_countries, # width= 500,
        projection= "robinson",
        locations= "pays",
        featureidkey= "properties.iso_a2_eh",
        color= df_metrique.columns[1],
        color_continuous_scale= [
        [0, 'aliceblue'],                         #0
        [1./10000, 'lightsteelblue'],             #10
        [1./1000, 'cornflowerblue'],              #100
        [1./100, 'royalblue'],                    #1000
        [1./10, 'limegreen'],                     #10000
        [1., 'green']] if metrique == "Nombre d'avis" else "Oranges",                     #100000
        scope= dict_continent[continent],
        labels= {"count": "Nombre d'avis", "moy": "Note moyenne"},
        hover_name= df_metrique["name_fr"],
        range_color=(0, df_metrique[df_metrique.columns[1]].max()),
        )
    
    if continent == "Europe" :
        fig.update_geos(resolution= 110,
                        # fitbounds= "locations",
                        # showocean = True, oceancolor= "dodgerblue",
                        visible= False,
                        showland= True, landcolor= "ghostwhite",
                        showcountries = True,
                        )
    else:
        fig.update_geos(resolution= 110,
                        # fitbounds= "locations",
                        # showocean = True, oceancolor= "dodgerblue",
                        visible= False,
                        showland= True, landcolor= "ghostwhite",
                        showcountries = True, 
                        )
    
    fig.update_layout(
        # paper_bgcolor= "LightSteelBlue",
        # title= "Nombre d'avis par pays",
        # autosize= True,
        margin= {"r": 0, "t": 0, "l": 0, "b": 0},
        )
    
    fig.update_coloraxes(
        colorbar_len= 1,
        colorbar= dict(orientation='h',
                                     y= -0.1,
                                     ticks= 'outside',
                                     ))
    
    # Affichage d'un tableau top10
    col1, col2 = st.columns([0.20, 0.80])

    col1.write("---")
    col1.write(f"***Top 5 {continent} :***")
    my_dict = {"count": "Total", "moy": "Moy."}
    t = df_metrique.head(5)[["name_fr", df_metrique.columns[1]]].rename(columns= {"name_fr": "Pays", 
                                                                                  df_metrique.columns[1]: my_dict[df_metrique.columns[1]]})
    col1.dataframe(t.sort_values(my_dict[df_metrique.columns[1]], ascending= False), use_container_width= True, hide_index= True, 
                   column_config= {"Pays": st.column_config.Column(width= "small", required=True)})

    # Affichage de la carte
    col2.plotly_chart(fig, use_container_width= True)

    st.write("---")


# *************************************************************************************************************
# ************************************* PAGE 4 : "PRÉPARATION DES DONNÉES" ************************************
# *************************************************************************************************************
if sidebar == pages[4]:
    intro = """ 
            L'objectif du projet s'apparente à un problème de regression.  
            Afin de mettre en place les algorithmes de Machine Learning associés, il est indispendable de discrétiser et de dichotomiser
            certaines variables pour construire des features et des attributs mieux adaptés pour les calculs.  
            Le modèle de données suivant détaille les relations entre les tables, les transformations appliquées sur elles à partir du
            dataset initial, ainsi que les attributs qui ont été rajoutés pour être adaptés aux algorithmes.  
            """

    modele_donnees = """
                À construire...
                """
    
    table_titres = """
                À construire...
                """
    
    table_commentaires = """
                À construire...
                """
    
    table_dates = """
                À construire...
                """
    
    st.header(pages[4])
    st.write(intro)
    st.write("---")

    st.subheader("1. Modèle de données")
    st.write(modele_donnees)
    st.subheader("2. Description des tables")
    st.write("#### a. Table des titres")
    st.write(table_titres)
    st.write("#### b. Table des commentaires")
    st.write(table_commentaires)
    st.write("#### c. Table des dates")
    st.write(table_dates)
    st.write("#### d. Autres tables")
    st.write("À construire")
    
# ----------------- Page 5 "Machine Learning" -----------------
if sidebar == pages[5]:
    st.header(pages[5])

# ----------------- Page 6 "Conclusion et perspectives" -----------------
if sidebar == pages[6]:
    st.header(pages[6])