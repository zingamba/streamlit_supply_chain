# --------------- CHARGEMENT DES BIBLIOTHÈQUES ---------------
# from bs4 import BeautifulSoup as bs
import time
import datetime as dt
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import io
# ---------------------------------------------------------------

# ****************************************************************************************************************
# CODE DE BASE POUR LE NETTOYAGE ET RÉUNIFICATION DES TABLES BRUTES LEBONCOIN ET VINTED
# ****************************************************************************************************************

# ------------ INITIALISATION DES VARIABLES GLOBALES ------------
file1 = "25000-reviews_leboncoin_trustpilot_scrapping.csv"
file2 = "25009-reviews_vinted_trustpilot_scrapping.csv"
df = None
df1 = None
df2 = None

# -------------------------- FONCTIONS --------------------------
# Mise à jour du type pour l'attribut "date/heure avis"
def func_to_datetime(_date):
    _date = _date.split(".")[0]
    _date = _date.replace("T", " ")
    _date = dt.datetime.strptime(_date, "%Y-%m-%d %H:%M:%S")
    return _date

# Récupération du jour de semaine
def func_get_weekday(_date):
    _dict = {0: "lundi", 1: "mardi", 2: "mercredi", 3: "jeudi", 4: "vendredi", 5: "samedi", 6: "dimanche"}
    _weekday = _date.day_of_week
    for key in _dict.keys():
        if _weekday == key:
            _weekday = _dict[key]
            break
    return _weekday

# Récupération du mois de l'expérience
def func_get_month(_date):
    _dict = {1: "janvier", 2: "février", 3: "mars", 4: "avril", 5: "mai", 6: "juin", 7: "juillet",
             8: "août", 9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre"}
    _month = _date.month
    for key in _dict.keys():
        if _month == key:
            _month = _dict[key]
            break
    return _month

# Récupération du nombre de mots dans un texte
def func_count_words(_text):
    _car = ("’", "'", "!", ",", "?", ";", ".", ":", "/", "+", "=", "\n", "- ", " -", "(", ")", "[", "]", "{", "}", "*", "<", ">")
    for car in _car:
        _text = _text.replace(car, " ")
    return len(_text.split())
# ---------------------------------------------------------------

# -------------------------- TRAITEMENT -------------------------
# Chargement des données brutes df1
df1 = pd.read_csv(file1, sep= ",")
df1 = df1.rename(columns= {"Unnamed: 0": "id avis", "date de visite": "date expérience",
                        "nb total avis": "nombre total avis", "date de l'avis (GMT+0)": "date/heure avis"})
df1 = df1.set_index("id avis")

# Chargement des données brutes df2
df2 = pd.read_csv(file2, sep= ",")
df2 = df2.rename(columns= {"Unnamed: 0": "id avis", "date de visite": "date expérience",
                        "nb total avis": "nombre total avis", "date de l'avis (GMT+0)": "date/heure avis"})
df2 = df2.set_index("id avis")

# Tri par date d'avis du plus récent au moins récent
df1 = df1.sort_values(by= "date/heure avis", ascending= False)
df2 = df2.sort_values(by= "date/heure avis", ascending= False)

# Copies des df
df1_old = df1.copy()
df2_old = df2.copy()

# Gestion des NA df1
# Il y a des NA parmi les modalités "nom" et "commentaire"
# On les remplace par ""
df1["nom"] = df1["nom"].fillna("")
df1["commentaire"] = df1["commentaire"].fillna("")

# Gestion des NA df2
# Il y a des NA parmi les modalités "pays", "titre" et "commentaire"
# On les remplace par ""
df2["pays"] = df2["pays"].fillna("")
df2["titre"] = df2["titre"].fillna("")
df2["commentaire"] = df2["commentaire"].fillna("")

# Mise à jour du type pour l'attribut "date expérience"
df1["date expérience"] = pd.to_datetime(df1["date expérience"])
df2["date expérience"] = pd.to_datetime(df2["date expérience"])

# Mise à jour du type pour l'attribut "date/heure avis"
df1["date/heure avis"] = df1["date/heure avis"].apply(func_to_datetime)
df2["date/heure avis"] = df2["date/heure avis"].apply(func_to_datetime)

# Mise à jour du type pour les autres attributs
dict_types = {"nom": "str", "pays": "str", "note": "int", "nombre total avis": "int", "titre": "str", "commentaire": "str"}
df1 = df1.astype(dict_types)
df2 = df2.astype(dict_types)

# Suppression des doublons
df1_duplicated = df1.duplicated()
df1_duplicated = df1[df1_duplicated == True]
df1 = df1.drop_duplicates()
df2_duplicated = df2.duplicated()
df2_duplicated = df2[df2_duplicated == True]
df2 = df2.drop_duplicates()

# Ajout de la colonne longueur du titre
df1["longueur titre"] = df1["titre"].apply(func_count_words)
df2["longueur titre"] = df2["titre"].apply(func_count_words)

# Ajout de la colonne longueur du commentaire
df1["longueur commentaire"] = df1["commentaire"].apply(func_count_words)
df2["longueur commentaire"] = df2["commentaire"].apply(func_count_words)

# Ajout de la colonne semaine expérience
df1["semaine expérience"] = df1["date expérience"].apply(lambda date: date.week)
df2["semaine expérience"] = df2["date expérience"].apply(lambda date: date.week)

# Ajout de la colonne jour de semaine de l'expérience
df1["jour semaine expérience"] = df1["date expérience"].apply(func_get_weekday)
df2["jour semaine expérience"] = df2["date expérience"].apply(func_get_weekday)

# Ajout de la colonne jour du mois de l'expérience
df1["jour expérience"] = df1["date expérience"].apply(lambda date: date.day)
df2["jour expérience"] = df2["date expérience"].apply(lambda date: date.day)

# Ajout de la colonne mois de l'expérience
df1["mois expérience"] = df1["date expérience"].apply(func_get_month)
df2["mois expérience"] = df2["date expérience"].apply(func_get_month)

# Ajout de la colonne année de l'expérience
df1["année expérience"] = df1["date expérience"].apply(lambda date: date.year)
df2["année expérience"] = df2["date expérience"].apply(lambda date: date.year)

# Ajout de la colonne semaine de l'avis
df1["semaine avis"] = df1["date/heure avis"].apply(lambda date: date.week)
df2["semaine avis"] = df2["date/heure avis"].apply(lambda date: date.week)

# Ajout de la colonne jour de semaine de l'avis
df1["jour semaine avis"] = df1["date/heure avis"].apply(func_get_weekday)
df2["jour semaine avis"] = df2["date/heure avis"].apply(func_get_weekday)

# Ajout de la colonne jour du mois de l'avis
df1["jour avis"] = df1["date/heure avis"].apply(lambda date: date.day)
df2["jour avis"] = df2["date/heure avis"].apply(lambda date: date.day)

# Ajout de la colonne mois de l'avis
df1["mois avis"] = df1["date/heure avis"].apply(func_get_month)
df2["mois avis"] = df2["date/heure avis"].apply(func_get_month)

# Ajout de la colonne année de l'avis
df1["année avis"] = df1["date/heure avis"].apply(lambda date: date.year)
df2["année avis"] = df2["date/heure avis"].apply(lambda date: date.year)

# Ajout de la colonne heure de l'avis
df1["heure avis"] = df1["date/heure avis"].apply(lambda date: date.hour)
df2["heure avis"] = df2["date/heure avis"].apply(lambda date: date.hour)

# Sélection de 24500 entrées les plus récentes pour chaque dataframe
df1 = df1.head(24500)
df2 = df2.head(24500)

# ------------------------ FINALISATION --------------------------
# Insertion d'une colonne indiquant le nom de l'entreprise pour l'avis concerné
list1 = ["Leboncoin"] * len(df1)
list2 = ["Vinted"] * len(df2)
df1.insert(0, "entreprise", list1)
df2.insert(0, "entreprise", list2)

# Concaténation des deux dataframe
df = pd.concat([df1, df2], axis= 0, ignore_index= True)

# Tri par date d'avis du plus récent au moins récent
df = df.sort_values(by= "date/heure avis", ascending= False)

# Mise à jour de l'index
df = df.reset_index(drop= True)
index_df = range(0, len(df))
df.insert(0, "id avis", index_df)
df = df.set_index("id avis")

# ****************************************************************************************************************
# FIN CODE DE BASE POUR LE NETTOYAGE ET RÉUNIFICATION DES TABLES BRUTES LEBONCOIN ET VINTED
# ****************************************************************************************************************


# ----------------- Titre   --------------------------------------------
title = ":orange[Leboncoin] vs :green[Vinted]"
col1, col2, col3 = st.columns([1, 0.1, 1.9])
col1.image("./images/leboncoin-logo.svg", width= 220)
col2.subheader("*/*")
col3.image("./images/vinted-logo.svg", width= 130)


# ----------------  Sidebar   ------------------------------------------
pages = ["Le projet", "Obtention des données", "Nettoyage du jeu de données", "Quelques visualisations", 
         "Préparation des données", "Machine learning", "Conclusion et perspectives"]
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
                Nous nous sommes intéressés à Leboncoin et Vinted : deux entreprises majeures, concurrentes et spécialisées dans la 
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
                **Note :** Plus loin dans le projet, nous nous servirons de la bibliothèque :orange[***Selenium***] pour nous connecter à Google
                traduction. Puis pour chaque avis : détecter et récupérer sa langue de rédaction, traduire automatiquement son titre/commentaire
                sauf si c'est déjà rédigés en français.
                """
    st.subheader("1. Sources de données")
    st.write(source)
    st.subheader("2. Web scrapping")
    st.write(webscrapping)
    st.write("---")
    st.write("**Le jeu de données brut peut être visible en partie, ou en totalité ci-après.**")

    # ----------- Si affichage demandé -----------
    if st.toggle("Infos sur la table récoltée de Leboncoin"):
        # Affichage des infos
        buffer = io.StringIO()
        df1_old.info(buf=buffer)
        s1 = buffer.getvalue()
        st.text(s1)

    if st.toggle("Infos sur la table récoltée de Vinted"):
        # Affichage des infos
        buffer = io.StringIO()
        df2_old.info(buf=buffer)
        s2 = buffer.getvalue()
        st.text(s2)

    if st.toggle("Jeu de données brut de Leboncoin"):
        number1 = st.number_input(":blue[*Nombre de lignes à afficher :*]", min_value=1, max_value= len(df1), value= 5, key= 1)
        # Affichage du df1
        st.dataframe(df1.head(number1))

    if st.toggle("Jeu de données brut de Vinted"):
        number2 = st.number_input(":blue[*Nombre de lignes à afficher :*]", min_value=1, max_value= len(df2), value= 5, key= 2)
        # Affichage du df2
        st.dataframe(df2.head(number2))

# ----------------- Page 2 "Nettoyage du jeu de données" ------------------------------
if sidebar == pages[2]:
    st.header(pages[2])
    # --------------- Affichage de l'introduction
    intro = """ 
            Le jeu de données brut (la table Leboncoin et la table Vinted) récolté d'internet par le webscrapping nécessite quelques transformations afin d'être exploitable pour effectuer
            les premières visualisations.\n
            Ces transformations impliquent notamment la mise à jour des types, des formattages de date,
            la suppression des doublons ou encore la gestion des NA.\n
            Les sections suivantes détaillent les opérations qui ont été effectuées sur le Leboncoin (df1) et Vinted (df2).
            Les infos sur les types et valeurs manquantes des deux tables du jeu de données sont visibles ci-après :
            """
    st.write(intro)
    st.write("---")
    col1, col2 = st.columns(2)
    if col1.toggle("Infos de la table Leboncoin"):
            # Affichage des infos de la table1
        buffer = io.StringIO()
        df1_old.info(buf=buffer)
        s1 = buffer.getvalue()
        col1.text(s1)
    if col2.toggle("Infos de la table Vinted"):
            # Affichage des infos de la table1
        buffer = io.StringIO()
        df2_old.info(buf=buffer)
        s2 = buffer.getvalue()
        col2.text(s2)
    st.write("---")

    # --------------- Gestion des NA
    gestion_na = """
            Les NA sont remplacés par des "" pour les attributs de type string.
            """
    st.subheader("1. Gestion des NA")
    st.write(gestion_na)
    code ="""
        # Gestion des NA df1
        # Il y a des NA parmi les modalités "nom" et "commentaire"
        # On les remplace par ""
        df1["nom"] = df1["nom"].fillna("")
        df1["commentaire"] = df1["commentaire"].fillna("")

        # Gestion des NA df2
        # Il y a des NA parmi les modalités "pays", "titre" et "commentaire"
        # On les remplace par ""
        df2["pays"] = df2["pays"].fillna("")
        df2["titre"] = df2["titre"].fillna("")
        df2["commentaire"] = df2["commentaire"].fillna("")
        """
    st.code(code)

    # Mise à jour des attributs
    maj_type_attributs_date_exp = """
            Dans les deux datasets, les modalités de la colonne :orange[**date expérience**] sont déjà sous la forme **%Y-%m-%d**, 
            mais en format string. On leur applique simplement une conversion en date.
                """
    
    st.subheader("2. Mise à jour des attributs")
        # Mise à jour de la date d'expérience
    st.write("#### a. Mise à jour de l'attribut de la date d'expérience")
    st.write(maj_type_attributs_date_exp)
    code = """
        # Mise à jour du type pour l'attribut "date expérience"
        df1["date expérience"] = pd.to_datetime(df1["date expérience"])
        df2["date expérience"] = pd.to_datetime(df2["date expérience"])
        """
    st.code(code)

        # Mise à jour de la date de l'avis
    maj_type_attributs_date_avis = """
            Dans les deux datasets, les modalités de la colonne :orange[**date/heure avis**] doivent être sous la forme **%Y-%m-%d %H:%M:%S**. 
            Cependant, certaines colonnes apparaissent sous une forme différente. Par exemple :
                """
    
    st.write("#### b. Mise à jour de l'attribut de la date de l'avis")
    st.write(maj_type_attributs_date_avis)
    n = len(df2_old.loc[df2_old["date/heure avis"].str.contains('T', regex= False), "date/heure avis"])
    st.write(f"{n} modalités de la colonne :orange[**date/heure avis**] de Vinted n'ont pas le bon format de date :")
    st.write(df2_old.loc[df2_old["date/heure avis"].str.contains('T', regex= False), "date/heure avis"])
    st.write(f"Création et application d'une fonction pour ajuster les string problématiques, avant de les formatter en datetime sur les deux df")
    code = """
        # Fonction pour convertir en datetime
        def func_to_datetime(_date):
            _date = _date.split(".")[0]
            _date = _date.replace("T", " ")
            _date = dt.datetime.strptime(_date, "%Y-%m-%d %H:%M:%S")
            return _date

        # Application de la fonction sur l'attribut "date/heure avis"
        df1["date/heure avis"] = df1["date/heure avis"].apply(func_to_datetime)
        df2["date/heure avis"] = df2["date/heure avis"].apply(func_to_datetime)
        """
    st.code(code)

        # Mise à jour de la date de l'avis
    maj_type_autres_attributs = """
            Utilisation d'un dictionnaire pour mettre à jour les types des autres attributs.
                """
    
    st.write("#### c. Mise à jour des autres attributs")
    st.write(maj_type_autres_attributs)
    code = """
        # Mise à jour du type pour les autres attributs
        dict_types = {"nom": "str", "pays": "str", "note": "int", "nombre total avis": "int", "titre": "str", "commentaire": "str"}
        df1 = df1.astype(dict_types)
        df2 = df2.astype(dict_types)
        """
    st.code(code)

    # --------------- Suppression des doublons
    suppression_doublons = """
            Chaque doublon est supprimé, en gardant le plus récent.
            """
    
    st.subheader("3. Suppression des doublons")
    st.write(suppression_doublons)
    code = """
        # Suppression des doublons
        df1_duplicated = df1.duplicated()
        df1_duplicated = df1[df1_duplicated == True]
        df1 = df1.drop_duplicates()
        df2_duplicated = df2.duplicated()
        df2_duplicated = df2[df2_duplicated == True]
        df2 = df2.drop_duplicates()
        """
    st.code(code)

    # --------------- Ajout des colonnes longueur titre/longueur commentaire
    ajout_col_nb_mots_titre_commentaire = """
            Rajout d'une colonne :orange[**longueur titre**] pour compter le nombre de mots qu'il y a dans le titre de l'avis.\n
            Idem pour le rajout de la colonne :orange[**longueur commentaire**].
            """
    
    st.subheader("4. Ajout des colonnes longueur titre/longueur commentaire")
    st.write(ajout_col_nb_mots_titre_commentaire)
    code = """
        # Récupération du nombre de mots dans un texte
        def func_count_words(_text):
            _car = ("’", "'", "!", ",", "?", ";", ".", ":", "/", "+", "=", "\\n", "- ", " -", "(", ")", "[", "]", "{", "}", "*", "<", ">")
            for car in _car:
                _text = _text.replace(car, " ")
            return len(_text.split())

        # Ajout de la colonne longueur du titre
        df1["longueur titre"] = df1["titre"].apply(func_count_words)
        df2["longueur titre"] = df2["titre"].apply(func_count_words)

        # Ajout de la colonne longueur du commentaire
        df1["longueur commentaire"] = df1["commentaire"].apply(func_count_words)
        df2["longueur commentaire"] = df2["commentaire"].apply(func_count_words)
        """
    st.code(code)

    # --------------- Ajout des colonnes semaine/jour/mois/année pour les dates de l'expérience
    ajout_col_date_exp = """
            Décomposition de l'attribut :orange[**date de l'expérience**] afin de récupérer la semaine, le jour de semaine,
            le jour de mois, le mois et l'année. 
            """
    
    st.subheader("5. Ajout des colonnes semaine/jour/mois/année à partir de la date de l'expérience")
    st.write(ajout_col_date_exp)
    code = """
        # Ajout de la colonne semaine expérience
        df1["semaine expérience"] = df1["date expérience"].apply(lambda date: date.week)
        df2["semaine expérience"] = df2["date expérience"].apply(lambda date: date.week)

        # Ajout de la colonne jour de semaine de l'expérience
        df1["jour semaine expérience"] = df1["date expérience"].apply(func_get_weekday)
        df2["jour semaine expérience"] = df2["date expérience"].apply(func_get_weekday)

        # Ajout de la colonne jour du mois de l'expérience
        df1["jour expérience"] = df1["date expérience"].apply(lambda date: date.day)
        df2["jour expérience"] = df2["date expérience"].apply(lambda date: date.day)

        # Ajout de la colonne mois de l'expérience
        df1["mois expérience"] = df1["date expérience"].apply(func_get_month)
        df2["mois expérience"] = df2["date expérience"].apply(func_get_month)

        # Ajout de la colonne année de l'expérience
        df1["année expérience"] = df1["date expérience"].apply(lambda date: date.year)
        df2["année expérience"] = df2["date expérience"].apply(lambda date: date.year)
        """
    st.code(code)

    # --------------- Ajout des colonnes semaine/heure/jour/mois/année pour les dates de l'avis
    ajout_col_date_avis = """
            Décomposition de l'attribut :orange[**date/heure avis**] afin de récupérer l'heure, la semaine, le jour de semaine, 
            le jour de mois, le mois et l'année.
            """
    
    st.subheader("6. Ajout des colonnes semaine/heure/jour/mois/année pour les dates de l'avis")
    st.write(ajout_col_date_avis)
    code = """
        # Ajout de la colonne semaine de l'avis
        df1["semaine avis"] = df1["date/heure avis"].apply(lambda date: date.week)
        df2["semaine avis"] = df2["date/heure avis"].apply(lambda date: date.week)

        # Ajout de la colonne jour de semaine de l'avis
        df1["jour semaine avis"] = df1["date/heure avis"].apply(func_get_weekday)
        df2["jour semaine avis"] = df2["date/heure avis"].apply(func_get_weekday)

        # Ajout de la colonne jour du mois de l'avis
        df1["jour avis"] = df1["date/heure avis"].apply(lambda date: date.day)
        df2["jour avis"] = df2["date/heure avis"].apply(lambda date: date.day)

        # Ajout de la colonne mois de l'avis
        df1["mois avis"] = df1["date/heure avis"].apply(func_get_month)
        df2["mois avis"] = df2["date/heure avis"].apply(func_get_month)

        # Ajout de la colonne année de l'avis
        df1["année avis"] = df1["date/heure avis"].apply(lambda date: date.year)
        df2["année avis"] = df2["date/heure avis"].apply(lambda date: date.year)

        # Ajout de la colonne heure de l'avis
        df1["heure avis"] = df1["date/heure avis"].apply(lambda date: date.hour)
        df2["heure avis"] = df2["date/heure avis"].apply(lambda date: date.hour)
        """
    st.code(code)

    # --------------- Concaténation des tables Leboncoin et Vinted
    select_concat_24500_avis = """
            À l'issue de toutes ces étapes de nettoyage des deux df, la longueur de la table Leboncoin s'élève 24992 entrées, 
            et celle de la table Vinted à 24867 entréees.\n
            Afin d'harmoniser le jeu de données en taille, les 24500 premières entrées des deux tables sont concaténéees verticalement 
            pour former un dataset unique.
            """
    
    st.subheader("7. Concaténation des tables Leboncoin et Vinted")
    st.write(select_concat_24500_avis)
    code = """
        # Sélection de 24500 entrées les plus récentes pour chaque dataframe
        df1 = df1.head(24500)
        df2 = df2.head(24500)

        # Insertion d'une colonne indiquant le nom de l'entreprise pour l'avis concerné
        list1 = ["Leboncoin"] * len(df1)
        list2 = ["Vinted"] * len(df2)
        df1.insert(0, "entreprise", list1)
        df2.insert(0, "entreprise", list2)

        # Concaténation des deux dataframe
        df = pd.concat([df1, df2], axis= 0, ignore_index= True)

        # Tri par date d'avis du plus récent au moins récent
        df = df.sort_values(by= "date/heure avis", ascending= False)

        # Mise à jour de l'index
        df = df.reset_index(drop= True)
        index_df = range(0, len(df))
        df.insert(0, "id avis", index_df)
        df = df.set_index("id avis")
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
    
    st.subheader("8. Ajout des colonnes langue de l'avis/titre fr/commentaire fr")
    st.write(traduction)
    
    # --------------- Conclusion de la page
    conclusion  = """
            Ces premières transformations ont permis d'obtenir un dataset de départ propre, et exploitable pour commencer 
            à faire quelques visualisations.\n
            Dans la suite du projet, de nouvelles transformations seront apportées pour discrétiser et dichotomiser certains attributs 
            pour créer des features adaptées aux calculs de Machine Learning.
            """
    st.subheader("9. État des lieux")
    st.write(f"Le jeu de données nettoyé et réajusté comporte {df.shape[0]} entrées et {df.shape[1]} attributs.")
    st.write(conclusion)
    st.write("---")
    if st.toggle("Infos du dataset après nettoyage et réajustement"):
        # Affichage des infos
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    if st.toggle("Lignes du dataset après nettoyage et réajustement"):
        number = st.number_input(":blue[*Nombre de lignes à afficher :*]", min_value=1, max_value= len(df), value= 5, key= 3)

# ----------------- Page 3 "Quelques datavisualisations" -----------------
if sidebar == pages[3]:
    st.header(pages[3])

# ----------------- Page 4 "Préparation des données" ---------------------
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