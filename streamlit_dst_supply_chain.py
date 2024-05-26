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
file_leboncoin = "25000-reviews_leboncoin_trustpilot_scrapping.csv"
df_leboncoin = pd.read_csv(file_leboncoin, sep= ",")

file_vinted = "25009-reviews_vinted_trustpilot_scrapping.csv"
df_vinted = pd.read_csv(file_vinted, sep= ",")

file_cleaned = "draft_avis-leboncoin-vinted-truspilot.csv"
df_cleaned = pd.read_csv(file_cleaned, sep= ",")
df_cleaned["date/heure avis"] = pd.to_datetime(df_cleaned["date/heure avis"])
df_cleaned["date expérience"] = pd.to_datetime(df_cleaned["date expérience"])
df_cleaned = df_cleaned.set_index("id avis") 

file_df_world_map = "./world-administrative-boundaries/world-administrative-boundaries.shp"
df_world_map = gpd.read_file(file_df_world_map)

# Récupération de la carte
with open('./geojson/world_map.json') as response :
    json_countries = json.load(response)

# ----------------- Titre   --------------------------------------------
st.title("")
title = ":orange[Leboncoin] vs :green[Vinted]"
col1, col2, col3 = st.columns([1, 0.1, 1.9])
col2.subheader("*/*")
col1.image("./images/leboncoin-logo.svg", width= 220)
col3.image("./images/vinted-logo.svg", width= 130)


# ----------------  Sidebar   ------------------------------------------
pages = ["Le projet", "Obtention des données", "Nettoyage du jeu de données", "Quelques visualisations", 
         "Préparation des données", "Machine learning", "Conclusion"]
auteurs = """
        Auteur1  
        Auteur2  
        Auteur3  
        Auteur4
        """
st.sidebar.title("Leboncoin vs Vinted")
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
    Nous nous sommes intéressés à Leboncoin et à Vinted : deux entreprises majeures, concurrentes et spécialisées dans la 
    publication de petites annonces en ligne pour la vente de biens de particulier à particulier.\n
    En 2024, Leboncoin.fr (entreprise française) est le 2e site e-commerce le plus consulté en France avec un traffic moyen de 
    28 millions de visiteurs uniques par mois.\n
    Tandis que Vinted.fr (entreprise lithuanienne), avec 16 millions de visiteurs uniques mensuel, est le 4ème site e-commerce
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
    Les données ont été collectées à partir :
    - [Des derniers avis clients déposés sur Truspilot pour Leboncoin](https://fr.trustpilot.com/review/www.leboncoin.fr?languages=all&sort=recency)
    - [Des derniers avis clients déposés sur Truspilot pour Vinted](https://fr.trustpilot.com/review/vinted.fr?languages=all&sort=recency)

    En effet, pour prédire la note d'un client, il est nécessaire d'identifier les entités importantes d’un avis : la note, la localisation, 
    le nom de l'entreprise, la date, ....  
    Mais aussi le commentaire laissé par le client afin d'en extraire le propos global : article défectueux ou conforme? 
    livraison correcte ou problématique? sentiment? ...\n
    Ne disposant pas d'une base consolidées avec ces informations, il est apparu nécessaire d'aller collecter ces données directement depuis
    la plateforme d'avis clients Trustpilot.
    """

    webscrapping = """
    Grâce à la bibliothèque :orange[***BeautifulSoup***] de python, un programme a été rédigé afin de collecter les données 
    en "webscrappant" le site Trustpilot.
    - Afin d'avoir un jeu de données  consistant, mais aussi pour avoir des avis étalés sur plusieurs mois, le code mis en place 
    a permis de récupérer la totalité des avis publiés pour Vinted.  
    25.009 avis à la date d'exécution du code.
    - Afin d'avoir un jeu données équilibré pour les deux entreprises, à la même date, les 25.000 derniers avis publiés pour Leboncoin 
    ont également été récoltés.\n
    Un jeu de données brut totalisant 50.009 entrées a ainsi été constitué.\n
    
    ***Note** : Plus loin dans le projet, nous nous servirons de la bibliothèque :orange[**Selenium**] pour nous connecter à Google traduction.*  
    *Puis pour chaque avis : détecter, récupérer sa langue de rédaction et traduire automatiquement son titre/commentaire
    sauf s'ils sont déjà rédigés en français.*
    """
    st.subheader("1. Sources de données")
    st.write(source)
    st.subheader("2. Web scrapping")
    st.write(webscrapping)
    st.write("---")
    st.write("**Le jeu de données brut peut être visible en partie, ou en totalité ci-après.**")

    # ----------- Si affichage demandé -----------
    if st.toggle("Infos sur la table récoltée de Leboncoin", key= 1):
        # Affichage des infos
        buffer = io.StringIO()
        df_leboncoin.info(buf=buffer)
        s1 = buffer.getvalue()
        st.text(s1)

    if st.toggle("Infos sur la table récoltée de Vinted", key= 2):
        # Affichage des infos
        buffer = io.StringIO()
        df_vinted.info(buf=buffer)
        s2 = buffer.getvalue()
        st.text(s2)

    if st.toggle("Jeu de données brut de Leboncoin", key= 3):
        number1 = st.number_input(":blue[*Nombre de lignes à afficher :*]", min_value=1, max_value= len(df_leboncoin), value= 5, key= 19)
        # Affichage du df_leboncoin
        st.dataframe(df_leboncoin.head(number1))

    if st.toggle("Jeu de données brut de Vinted", key= 4):
        number2 = st.number_input(":blue[*Nombre de lignes à afficher :*]", min_value=1, max_value= len(df_vinted), value= 5, key= 20)
        # Affichage du df_vinted
        st.dataframe(df_vinted.head(number2))

# ----------------- Page 2 "Nettoyage du jeu de données" ------------------------------
if sidebar == pages[2]:
    st.header(pages[2])
    # --------------- Affichage de l'introduction
    intro = """ 
            Le jeu de données brut (la table Leboncoin et la table Vinted) récolté d'internet par le webscrapping nécessite quelques 
            transformations afin d'être exploitable pour effectuer les premières visualisations.\n
            Ces transformations impliquent notamment le renommage des colonnes, la mise à jour des types, des formattages de date,
            la suppression des doublons ou encore la gestion des NA.\n
            Les sections suivantes détaillent les opérations qui ont été effectuées sur le Leboncoin (df_leboncoin) et Vinted (df_vinted).
            Les infos sur les types et valeurs manquantes des deux tables du jeu de données sont visibles ci-après :
            """
    st.write(intro)
    st.write("---")
    col1, col2 = st.columns(2)

    if col1.toggle("Infos de la table Leboncoin", key= 5):
            # Affichage des infos de la table1
        buffer = io.StringIO()
        df_leboncoin.info(buf=buffer)
        s1 = buffer.getvalue()
        col1.text(s1)
    
    if col2.toggle("Infos de la table Vinted", key= 6):
            # Affichage des infos de la table1
        buffer = io.StringIO()
        df_vinted.info(buf=buffer)
        s2 = buffer.getvalue()
        col2.text(s2)
    st.write("---")

    # --------------- Renommage des attributs
    renommage = """
            Le renommage des colonnes se fait à l'aide d'un dictionnaire.\n
            Le tri des entrées s'effectue dans le sens des plus récentes au moins récenes en fonction de la date de l'avis.
            """
    st.subheader("1. Renommage des attributs et tri des tables")
    st.write(renommage)

    if st.toggle(":green[Afficher le code]", key= 7):
        code = """
                # Chargement des données brutes de Leboncoin dans df_leboncoin
                df_leboncoin = pd.read_csv(file_leboncoin, sep= ",")
                df_leboncoin = df_leboncoin.rename(columns= {"Unnamed: 0": "id avis", "date de visite": "date expérience",
                                        "nb total avis": "nombre total avis", "date de l'avis (GMT+0)": "date/heure avis"})
                df_leboncoin = df_leboncoin.set_index("id avis")

                # Chargement des données brutes de Vinted dans df_vinted
                df_vinted = pd.read_csv(file_vinted, sep= ",")
                df_vinted = df_vinted.rename(columns= {"Unnamed: 0": "id avis", "date de visite": "date expérience",
                                        "nb total avis": "nombre total avis", "date de l'avis (GMT+0)": "date/heure avis"})
                df_vinted = df_vinted.set_index("id avis")

                # Tri par date d'avis du plus récent au moins récent
                df_leboncoin = df_leboncoin.sort_values(by= "date/heure avis", ascending= False)
                df_vinted = df_vinted.sort_values(by= "date/heure avis", ascending= False)
                """
        st.code(code)

    # --------------- Gestion des NA
    gestion_na = """
            Les NA sont remplacés par des "" pour les attributs de type string.
            """
    st.subheader("2. Gestion des NA")
    st.write(gestion_na)

    if st.toggle(":green[Afficher le code]", key= 8):
        code ="""
            # Gestion des NA df_leboncoin
            # Il y a des NA parmi les modalités "nom" et "commentaire"
            # On les remplace par ""
            df_leboncoin["nom"] = df_leboncoin["nom"].fillna("")
            df_leboncoin["commentaire"] = df_leboncoin["commentaire"].fillna("")

            # Gestion des NA df_vinted
            # Il y a des NA parmi les modalités "pays", "titre" et "commentaire"
            # On les remplace par ""
            df_vinted["pays"] = df_vinted["pays"].fillna("")
            df_vinted["titre"] = df_vinted["titre"].fillna("")
            df_vinted["commentaire"] = df_vinted["commentaire"].fillna("")
            """
        st.code(code)

    # Mise à jour des attributs
    maj_type_attributs_date_exp = """
            Dans les deux datasets, les modalités de la colonne :orange[**date expérience**] sont déjà sous la forme **%Y-%m-%d**, 
            mais en format string. On leur applique simplement une conversion en date.
                """
    
    st.subheader("3. Mise à jour des attributs")
        # Mise à jour de la date d'expérience
    st.write("#### a. Mise à jour de l'attribut de la date d'expérience")
    st.write(maj_type_attributs_date_exp)

    if st.toggle(":green[Afficher le code]", key= 9):
        code = """
            # Mise à jour du type pour l'attribut "date expérience"
            df_leboncoin["date expérience"] = pd.to_datetime(df_leboncoin["date expérience"])
            df_vinted["date expérience"] = pd.to_datetime(df_vinted["date expérience"])
            """
        st.code(code)

        # Mise à jour de la date de l'avis
    maj_type_attributs_date_avis = """
            Dans les deux datasets, les modalités de la colonne :orange[**date/heure avis**] doivent être sous la forme **%Y-%m-%d %H:%M:%S**. 
            Cependant, certaines colonnes apparaissent sous une forme différente.\n
            Création et application d'une fonction pour ajuster les string problématiques, avant de les formatter en datetime sur les deux df
                """
    
    st.write("#### b. Mise à jour de l'attribut de la date de l'avis")
    st.write(maj_type_attributs_date_avis)
    
    if st.toggle(":green[Afficher le code]", key= 10):
        code = """
            # Fonction pour convertir en datetime
            def func_to_datetime(_date):
                _date = _date.split(".")[0]
                _date = _date.replace("T", " ")
                _date = dt.datetime.strptime(_date, "%Y-%m-%d %H:%M:%S")
                return _date

            # Application de la fonction sur l'attribut "date/heure avis"
            df_leboncoin["date/heure avis"] = df_leboncoin["date/heure avis"].apply(func_to_datetime)
            df_vinted["date/heure avis"] = df_vinted["date/heure avis"].apply(func_to_datetime)
            """
        st.code(code)

        # Mise à jour de la date de l'avis
    maj_type_autres_attributs = """
            Utilisation d'un dictionnaire pour mettre à jour les types des autres attributs.
                """
    
    st.write("#### c. Mise à jour des autres attributs")
    st.write(maj_type_autres_attributs)
    
    if st.toggle(":green[Afficher le code]", key= 11):
        code = """
            # Mise à jour du type pour les autres attributs
            dict_types = {"nom": "str", "pays": "str", "note": "int", "nombre total avis": "int", "titre": "str", "commentaire": "str"}
            df_leboncoin = df_leboncoin.astype(dict_types)
            df_vinted = df_vinted.astype(dict_types)
            """
        st.code(code)

    # --------------- Suppression des doublons
    suppression_doublons = """
            Chaque doublon est supprimé, en gardant le plus récent.
            """
    
    st.subheader("4. Suppression des doublons")
    st.write(suppression_doublons)

    if st.toggle(":green[Afficher le code]", key= 12):
        code = """
            # Suppression des doublons
            df1_duplicated = df_leboncoin.duplicated()
            df1_duplicated = df_leboncoin[df1_duplicated == True]
            df_leboncoin = df_leboncoin.drop_duplicates()
            df2_duplicated = df_vinted.duplicated()
            df2_duplicated = df_vinted[df2_duplicated == True]
            df_vinted = df_vinted.drop_duplicates()
            """
        st.code(code)

    # --------------- Ajout des colonnes longueur titre/longueur commentaire
    ajout_col_nb_mots_titre_commentaire = """
            Rajout d'une colonne :orange[**longueur titre**] pour compter le nombre de mots qu'il y a dans le titre de l'avis.\n
            Idem pour le rajout de la colonne :orange[**longueur commentaire**].
            """
    
    st.subheader("5. Ajout des colonnes longueur titre/longueur commentaire")
    st.write(ajout_col_nb_mots_titre_commentaire)

    if st.toggle(":green[Afficher le code]", key= 13):
        code = """
            # Récupération du nombre de mots dans un texte
            def func_count_words(_text):
                _car = ("’", "'", "!", ",", "?", ";", ".", ":", "/", "+", "=", "\\n", "- ", " -", "(", ")", "[", "]", "{", "}", "*", "<", ">")
                for car in _car:
                    _text = _text.replace(car, " ")
                return len(_text.split())

            # Ajout de la colonne longueur du titre
            df_leboncoin["longueur titre"] = df_leboncoin["titre"].apply(func_count_words)
            df_vinted["longueur titre"] = df_vinted["titre"].apply(func_count_words)

            # Ajout de la colonne longueur du commentaire
            df_leboncoin["longueur commentaire"] = df_leboncoin["commentaire"].apply(func_count_words)
            df_vinted["longueur commentaire"] = df_vinted["commentaire"].apply(func_count_words)
            """
        st.code(code)

    # --------------- Ajout des colonnes semaine/jour/mois/année pour les dates de l'expérience
    ajout_col_date_exp = """
    Décomposition de l'attribut :orange[**date de l'expérience**] afin de récupérer la semaine, le jour de semaine,
    le jour de mois, le mois et l'année. 
    """
    
    st.subheader("6. Ajout des colonnes semaine/jour/mois/année à partir de la date de l'expérience")
    st.write(ajout_col_date_exp)

    if st.toggle(":green[Afficher le code]", key= 14):
        code = """
        # Ajout de la colonne semaine expérience
        df_leboncoin["semaine expérience"] = df_leboncoin["date expérience"].apply(lambda date: date.week)
        df_vinted["semaine expérience"] = df_vinted["date expérience"].apply(lambda date: date.week)

        # Ajout de la colonne jour de semaine de l'expérience
        df_leboncoin["jour semaine expérience"] = df_leboncoin["date expérience"].apply(lambda date: date.day_of_week)
        df_vinted["jour semaine expérience"] = df_vinted["date expérience"].apply(lambda date: date.day_of_week)

        # Ajout de la colonne jour du mois de l'expérience
        df_leboncoin["jour expérience"] = df_leboncoin["date expérience"].apply(lambda date: date.day)
        df_vinted["jour expérience"] = df_vinted["date expérience"].apply(lambda date: date.day)

        # Ajout de la colonne mois de l'expérience
        df_leboncoin["mois expérience"] = df_leboncoin["date expérience"].apply(lambda date: date.month)
        df_vinted["mois expérience"] = df_vinted["date expérience"].apply(lambda date: date.month)

        # Ajout de la colonne année de l'expérience
        df_leboncoin["année expérience"] = df_leboncoin["date expérience"].apply(lambda date: date.year)
        df_vinted["année expérience"] = df_vinted["date expérience"].apply(lambda date: date.year)
        """
        st.code(code)

    # --------------- Ajout des colonnes semaine/heure/jour/mois/année pour les dates de l'avis
    ajout_col_date_avis = """
    Décomposition de l'attribut :orange[**date/heure avis**] afin de récupérer l'heure, la semaine, le jour de semaine, 
    le jour de mois, le mois et l'année.
    """
    
    st.subheader("7. Ajout des colonnes semaine/heure/jour/mois/année pour les dates de l'avis")
    st.write(ajout_col_date_avis)

    if st.toggle(":green[Afficher le code]", key= 15):
        code = """
    # Ajout de la colonne semaine de l'avis
    df_leboncoin["semaine avis"] = df_leboncoin["date/heure avis"].apply(lambda date: date.week)
    df_vinted["semaine avis"] = df_vinted["date/heure avis"].apply(lambda date: date.week)

    # Ajout de la colonne jour de semaine de l'avis
    df_leboncoin["jour semaine avis"] = df_leboncoin["date/heure avis"].apply(func_get_weekday)
    df_vinted["jour semaine avis"] = df_vinted["date/heure avis"].apply(func_get_weekday)

    # Ajout de la colonne jour du mois de l'avis
    df_leboncoin["jour avis"] = df_leboncoin["date/heure avis"].apply(lambda date: date.day)
    df_vinted["jour avis"] = df_vinted["date/heure avis"].apply(lambda date: date.day)

    # Ajout de la colonne mois de l'avis
    df_leboncoin["mois avis"] = df_leboncoin["date/heure avis"].apply(func_get_month)
    df_vinted["mois avis"] = df_vinted["date/heure avis"].apply(func_get_month)

    # Ajout de la colonne année de l'avis
    df_leboncoin["année avis"] = df_leboncoin["date/heure avis"].apply(lambda date: date.year)
    df_vinted["année avis"] = df_vinted["date/heure avis"].apply(lambda date: date.year)

    # Ajout de la colonne heure de l'avis
    df_leboncoin["heure avis"] = df_leboncoin["date/heure avis"].apply(lambda date: date.hour)
    df_vinted["heure avis"] = df_vinted["date/heure avis"].apply(lambda date: date.hour)
    """
        st.code(code)

    # --------------- Concaténation des tables Leboncoin et Vinted
    select_concat_24500_avis = """
    À l'issue de toutes ces étapes de nettoyage des deux df, la longueur de la table Leboncoin s'élève 24992 entrées, 
    et celle de la table Vinted à 24867 entréees.\n
    Afin d'harmoniser le jeu de données en taille, les 24500 premières entrées des deux tables sont concaténéees verticalement 
    pour former un dataset unique.
    """
    
    st.subheader("8. Concaténation des tables Leboncoin et Vinted")
    st.write(select_concat_24500_avis)
    
    if st.toggle(":green[Afficher le code]", key= 16):
        code = """
        # Sélection de 24500 entrées les plus récentes pour chaque dataframe
        df_leboncoin = df_leboncoin.head(24500)
        df_vinted = df_vinted.head(24500)

        # Insertion d'une colonne indiquant le nom de l'entreprise pour l'avis concerné
        list1 = ["Leboncoin"] * len(df_leboncoin)
        list2 = ["Vinted"] * len(df_vinted)
        df_leboncoin.insert(0, "entreprise", list1)
        df_vinted.insert(0, "entreprise", list2)

        # Concaténation des deux dataframe
        df_cleaned = pd.concat([df_leboncoin, df_vinted], axis= 0, ignore_index= True)

        # Tri par date d'avis du plus récent au moins récent
        df_cleaned = df_cleaned.sort_values(by= "date/heure avis", ascending= False)

        # Mise à jour de l'index
        df_cleaned = df_cleaned.reset_index(drop= True)
        index_df = range(0, len(df_cleaned))
        df_cleaned.insert(0, "id avis", index_df)
        df_cleaned = df_cleaned.set_index("id avis")
        """
        st.code(code)


    # --------------- Ajout des colonnes langue de l'avis/titre fr/commentaire fr
    traduction = """
    Certains avis ont été rédigés par des clients dans une autre langue. La dernière étape dans ce 1er travail 
    de revue de dataset consiste à rajouter :
    - Une colonne avec la valeur de la langue dans laquelle l'avis a été rédigé
    - Une colonne avec la traduction du titre en français
    - Une colonne avec la traduction du commentaire en français\n
    Pour récupérer les traductions, nous nous servons de la bibliothèque :orange[**Selenium**] de python pour nous connecter 
    à Google translate afin de :  
    - Insérer chaque titre/commentaire de la table dans le champ du texte à traduire de la page Google  
    - Récupérer la valeur de la "Langue Détectée" par Google et la rajouter dans la ligne correspondante du dataset  
    - Récupérer la valeur de la traduction et la rajouter dans la ligne correspondante du dataset (si le titre/commentaire
    n'est pas rédigé en français)
    """
    
    st.subheader("9. Ajout des colonnes langue de l'avis/titre fr/commentaire fr")
    st.write(traduction)
    
    # --------------- Conclusion de la page
    conclusion  = """
    Ces premières transformations ont permis d'obtenir un dataset de départ propre, et exploitable pour commencer 
    à faire quelques visualisations.\n
    Dans la suite du projet, de nouvelles transformations seront apportées pour discrétiser et dichotomiser certains attributs 
    afin de créer des features adaptées aux calculs de Machine Learning.
    """
    st.subheader("10. État des lieux")
    st.write(f"Le jeu de données nettoyé et réajusté comporte {df_cleaned.shape[0]} entrées et {df_cleaned.shape[1]} attributs.")
    st.write(conclusion)
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
        colonnes = st.multiselect(":green[Attributs =]", df_cleaned.columns, placeholder= "...")
        attributs = list(df_cleaned.columns) if colonnes == list() else colonnes

        col1, col2, col3 = st.columns(3)
        nb_avis = col1.number_input(":green[Nombre d'avis max à afficher =]", min_value= 1, max_value= len(df_cleaned), value= len(df_cleaned), help= h)
        notes = col2.multiselect(":green[Note =]", sorted(df_cleaned["note"].unique(), reverse= True), placeholder= "...")
        notes = list(df_cleaned["note"].unique()) if notes == list() else notes

        min_date = df_cleaned["date/heure avis"].min()
        max_date = df_cleaned["date/heure avis"].max()
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

    # Affichage de la table
    df_shown = df_cleaned.copy()
    df_shown = df_shown[(df_shown["date/heure avis"].between(date_debut, date_fin, inclusive= "both"))]
    df_shown = df_shown[(df_shown["note"].isin(notes))]
    df_shown = df_shown[attributs]
    df_shown = df_shown.head(nb_avis)

    st.dataframe(df_shown, column_config= 
                 {"_index": st.column_config.NumberColumn(format= "%d"),
                  "année expérience": st.column_config.NumberColumn(format= "%d"),
                  "année avis": st.column_config.NumberColumn(format= "%d"),
                  "date expérience": st.column_config.DateColumn()})

    # Impression des résultats
    st.write(f"**{len(df_shown)}** avis récupérés à partir des plus récents")
    if str(date_fin).split(' ')[0] != str(date_debut).split(' ')[0]:
        st.write(f"Publiés entre le **{str(date_fin).split(' ')[0]}** et le **{str(date_debut).split(' ')[0]}**.")
    else:
        st.write(f"Publiés le **{str(date_fin).split(' ')[0]}**.")
    st.write(f"Avec une note de **{str(sorted(notes, reverse= True)).replace('[', '').replace(']', '').replace(',', ' ou')}** étoiles")

    st.write("---")

    # --------------------------------------------------------------------
    # *************** Répartition du nombre total des notes **************
    # --------------------------------------------------------------------
    st.subheader("2. Répartition du nombre total d'avis")
    st.write("""
             Afficher la répartition du nombre d'avis en fonction de la note, du pays.  
             Ou encore en fonction de l'heure, du jour, du mois et de l'année de publication.
             """)
    
    # Code couleur pour Leboncoin et Vinted
    color_l = "#E67333"
    color_v = "#337680"

    # Création des filtres
    liste_filtres = ["Valeur de la note", "Pays du client", "Heure de publication", "Jour de publication en semaine", 
                     "Jour de publication dans le mois", "Mois de publication", "Année de publication"]
    dict_note = {"Valeur de la note": "note", "Pays du client": "pays", "Heure de publication": "heure avis", 
               "Jour de publication en semaine": "jour semaine avis", "Jour de publication dans le mois": "jour avis",
               "Mois de publication": "mois avis", "Année de publication": "année avis"}
    
    with st.container(border= True):
        type_filtre = st.selectbox("###### **Nombre d'avis en fonction de :**", options= liste_filtres)
    filtre = dict_note[type_filtre]

    # Nombre d'avis en fonction de la valeur de "valeur de la note", "heure avis", "jour semaine avis", "jour avis", "mois avis", "année avis"
    if filtre in ["note", "heure avis", "jour semaine avis", "jour avis", "mois avis", "année avis"]:
        fig, ax = plt.subplots(figsize= (10, 5))
        sns.countplot(x= df_cleaned[filtre], hue= df_cleaned["entreprise"], palette= [color_l, color_v], ax= ax)
        plt.title(f"Nombre d'avis par : {type_filtre}")
        plt.xlabel(type_filtre.capitalize() + (f" (nombre d'étoiles)" if filtre == "note" else ""))
        plt.ylabel("Nombre d'avis")
        plt.legend(title= "Entreprise")

        # Affichage des noms de jours si filtre sur le jour
        if (filtre == "jour semaine avis") :
            dict_jour = {0: "Dimanche", 1: "Lundi", 2: "Mardi", 3: "Mercredi", 4: "Jeudi", 5: "Vendredi", 6: "Samedi"}
            plt.xticks([0, 1, 2, 3, 4, 5, 6], ["Dimanche", "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"])
        st.pyplot(fig)
    
    # Nombre d'avis en fonction du "pays du client"
    if filtre == "pays":
        # Affichage du graphique de Leboncoin
        df_cleaned_pays = df_cleaned.copy()
        df_cleaned_pays["pays"] = df_cleaned_pays["pays"].apply(lambda x: "Autres pays" if x not in ["FR"] else x)
        fig1, ax = plt.subplots(figsize= (10, 5))
        sns.countplot(x= df_cleaned_pays["pays"], hue= df_cleaned_pays["entreprise"], legend= "full", palette= [color_l, color_v], ax= ax)
        plt.title(f"Nombre d'avis par : {type_filtre}")
        plt.xlabel(type_filtre.capitalize())        
        plt.ylabel("Nombre d'avis")
        #plt.legend(title= "Entreprise")
        st.pyplot(fig1)

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
    quant_vars = ["note", "nombre total avis", "longueur titre", "longueur commentaire"]     # Variables quantitatives
    temp_vars = ["jour semaine expérience", "semaine expérience", "jour expérience", "mois expérience", "année expérience", 
                 "heure avis", "jour semaine avis", "semaine avis", "jour avis", "mois avis", "année avis"]  # Variables temporelles
    entreprises = ["Leboncoin", "Vinted"]
    
    with st.expander("###### **Filtrer par :**", expanded= False):
        sel1 = st.multiselect(":green[Variables quantitatives =]", quant_vars, default= quant_vars, placeholder= "...")
        sel2 = st.multiselect(":green[Variables temporelles =]", temp_vars, default= temp_vars, placeholder= "...")
        entreprises = st.multiselect(":green[Entreprises =]", entreprises, default= entreprises, placeholder= "...")

    # Affichage du graphique
    df_corr = df_cleaned[df_cleaned["entreprise"] == entreprises[0]] if len(entreprises) == 1 else df_cleaned.copy()
    df_corr = df_corr[sel1 + sel2]
    df_corr = df_corr.corr()
    try:
        fig, ax = plt.subplots(figsize= (11, 5))
        if st.checkbox("**Montrer les variables en corrélation forte**"):
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
    quant_vars = ["note", "nombre total avis", "longueur titre", "longueur commentaire"]     # Variables quantitatives
    temp_vars = ["jour semaine expérience", "semaine expérience", "jour expérience", "mois expérience", "année expérience", 
                 "heure avis", "jour semaine avis", "semaine avis", "jour avis", "mois avis", "année avis"]  # Variables temporelles
    entreprises = ["Leboncoin", "Vinted"]
    
    sel1 = st.selectbox(":green[x =]", options= ["note", "jour semaine expérience", "année expérience",
                                            "jour semaine avis", "année avis"], index= 0, placeholder= "")
    sel2 = st.selectbox(":green[y =]", options= quant_vars + temp_vars, index= 3, placeholder= "")
    entreprises = st.multiselect(":green[Entreprises =]", entreprises, default= entreprises, placeholder= "")

    # Affichage de la distribution des avis
    df_cleaned_pays_dist = df_cleaned[df_cleaned["entreprise"] == entreprises[0]] if len(entreprises) == 1 else df_cleaned
    if (sel1 in (quant_vars + temp_vars)) and (sel2 in (quant_vars + temp_vars)):
        fig, ax = plt.subplots(figsize= (10, 5))

        # Masque des valeurs extrêmes
        if not st.checkbox("**Afficher les valeurs extrêmes**") :
            sns.boxplot(x= df_cleaned_pays_dist[sel1], y= df_cleaned_pays_dist[sel2], showfliers = False,
                        hue= df_cleaned_pays_dist["entreprise"], palette= [color_l, color_v])
        else:
            sns.boxplot(x= df_cleaned_pays_dist[sel1], y= df_cleaned_pays_dist[sel2], showfliers = True,
                        hue= df_cleaned_pays_dist["entreprise"], palette= [color_l, color_v])
        plt.title(f"Distribution entre : {sel1.upper()} et {sel2.upper()}")
        plt.xlabel(sel1)
        plt.ylabel(sel2)
        plt.legend(title= "Entreprise")

        # Affichage des noms de jours si filtre sur le jour
        if (sel1 == "jour semaine avis") or (sel1 == "jour semaine expérience"):
            dict_jour = {0: "Dimanche", 1: "Lundi", 2: "Mardi", 3: "Mercredi", 4: "Jeudi", 5: "Vendredi", 6: "Samedi"}
            plt.xticks([0, 1, 2, 3, 4, 5, 6], ["Dimanche", "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"])
        if (sel2 == "jour semaine avis") or (sel2 == "jour semaine expérience"):
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
                   "Amérique du Sud": "south america","Europe": "europe", "Asie": "asia"}
    
    # Création des filtres
    continent = st.selectbox(":green[Continent =]",
                        options= list(dict_continent.keys()),
                        index= 0, placeholder= "")
    entreprises = st.multiselect(":green[Entreprise =]", entreprises, default= entreprises, placeholder= "", key= 23)

    # ---------------------------------------------
    # ------ Nombre d'avis par pays (global) ------
    # ---------------------------------------------
    if len(entreprises) == 1:
        # Préparation des données à dessiner l'entreprise
        df_nb_avis = df_cleaned.loc[df_cleaned["entreprise"] == entreprises[0], "pays"].value_counts().reset_index()
    else:
        # Préparation des données à dessiner pour les deux entreprise
        df_nb_avis = df_cleaned["pays"].value_counts().reset_index()

    df_nb_avis = df_nb_avis.merge(df_world_map, how= "inner", left_on= "pays", right_on= "iso_3166_1_")
    df_nb_avis = df_nb_avis[["pays", "count", "french_shor", "continent", "region"]].rename(columns= {"french_shor": "name_fr"})
    df_nb_avis["continent"] = df_nb_avis["continent"].apply(lambda x: x.lower())
    df_nb_avis["region"] = df_nb_avis["region"].apply(lambda x: x.lower())

    # Classement de certains pays dans les bons continents
    for i in range(0, len(df_nb_avis)):
        df_nb_avis.iloc[i, 3] = df_nb_avis.iloc[i, 4].lower() if df_nb_avis.iloc[i, 3] == "americas" else df_nb_avis.iloc[i, 3].lower()
        df_nb_avis.iloc[i, 3] = "north america" if df_nb_avis.iloc[i, 4] == "northern america" else df_nb_avis.iloc[i, 3]
        df_nb_avis.iloc[i, 3] = "north america" if df_nb_avis.iloc[i, 4] == "central america" else df_nb_avis.iloc[i, 3]
        df_nb_avis.iloc[i, 3] = "south america" if df_nb_avis.iloc[i, 4] == "caribbean" else df_nb_avis.iloc[i, 3]
        df_nb_avis.iloc[i, 3] = "asia" if df_nb_avis.iloc[i, 3] == "oceania" else df_nb_avis.iloc[i, 3]

    # Filtre du dataset sur la sélection utilisateur
    st.text(f'{h}')
    df_nb_avis = df_nb_avis[df_nb_avis["continent"] == dict_continent[continent]] if dict_continent[continent] != "world" else df_nb_avis
    # df_nb_avis["count"] = np.log10(df_nb_avis["count"])

    # Tracé de la carte et tableau des résultats
    fig = px.choropleth(df_nb_avis, geojson= json_countries,
        locations= "pays",
        color= 'count',
        featureidkey= "properties.iso_a2_eh",
        color_continuous_scale= [
        [0, 'powderblue'],                  #0
        [1./10000, 'powderblue'],           #10
        [1./1000, 'lightskyblue'],          #100
        [1./100, 'cornflowerblue'],         #1000
        [1./10, 'chartreuse'],                   #10000
        [1., 'darkred']],                   #100000
        scope= dict_continent[continent],
        labels= {"count":"Nombre d'avis"},
        hover_name= df_nb_avis["name_fr"],
        range_color=(0, df_nb_avis["count"].max()),
        )
    
    if continent == "Europe" :
        fig.update_geos(resolution= 110, 
            showland= True, landcolor= "whitesmoke",
            showcountries = True,
                    )
    else:
        fig.update_geos(resolution= 110, 
            fitbounds= "locations", visible= True,
            showland= True, landcolor= "whitesmoke",
            showcountries = True,
                    )
    
    fig.update_layout(
        paper_bgcolor="white",
        # title= "Nombre d'avis par pays",
        # autosize= True,
        # width= 200,
        # height=2500,
        margin={"r":0,"t":0,"l":0,"b":0},
        coloraxis=dict(colorbar=dict(orientation='h', y= -0.05,
                                     #tickvals = [0, 10, 100, 1000, 10000, 40000],
                                     ticks= 'outside')
        )
    
    # fig.update_yaxes(automargin= "left+top")

    # Affichage d'un tableau top5
    texte = f'{entreprises[0]}' if len(entreprises) == 1 else f'Leboncoin\nVinted'
    col1, col2 = st.columns([0.20, 0.80])
    col1.write(f"***Top 10 {continent} :***")
    col1.text(texte)
    t = df_nb_avis.head(10)[["name_fr", "count"]].rename(columns= {"name_fr": "Pays", "count": "Total"})
    col1.dataframe(t, hide_index= True, column_config= {"Pays": st.column_config.Column(width= "small", required=True)})

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