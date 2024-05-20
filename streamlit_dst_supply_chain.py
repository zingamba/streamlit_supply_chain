# --------------- CHARGEMENT DES BIBLIOTHÈQUES ---------------
from bs4 import BeautifulSoup as bs
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
import time
import datetime as dt
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import io
# ---------------------------------------------------------------

# --------------- STRATÉGIE GLOBALE DU PROGRAMME ----------------

# ------------ INITIALISATION DES VARIABLES GLOBALES ------------
file1 = "25000-reviews_leboncoin_trustpilot_scrapping.csv"
file2 = "25009-reviews_vinted_trustpilot_scrapping.csv"

# -------------------------- TRAITEMENT -------------------------
# --------------------------  Titre   -------------------------
title = ":orange[Leboncoin] vs :green[Vinted]"
st.title(title)

# --------------------------  Sidebar   -------------------------
pages = ["Le projet", "Obtention des données", "Le jeu de données", "Quelques visualisations", 
         "Pré-processing des données", "Modèle relationnel de données", "Machine learning", "Conclusion et perspectives"]
auteurs = """
        Auteur1  
        Auteur2  
        Auteur3  
        Auteur4
        """
st.sidebar.title("Leboncoin vs Vinted")
sidebar = st.sidebar.radio("Aller vers la page :", pages)
st.sidebar.write("---")
st.sidebar.subheader("Auteurs")             
st.sidebar.write(auteurs)

# ----------------------- Page "Le Projet" ---------------------
if sidebar == pages[0]:
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
                la mise en place d'algorithmes de Machine Learning pour prédire les résultats des prédictions, que nous vous inviterons à tester.
                """

    st.subheader("Présentation")
    st.write(presentation)
    st.subheader("Objectif")
    st.write(objectif)

# ----------------- Page "Obtention des données" ----------------
if sidebar == pages[1]:
    source = """ 
                Les sources de données seront ont été collectées à partir :
                - [Des derniers avis déposés sur Truspilot pour Leboncoin](https://fr.trustpilot.com/review/www.leboncoin.fr?languages=all&sort=recency)
                - [Des derniers avis déposés sur Truspilot pour Vinted](https://fr.trustpilot.com/review/vinted.fr?languages=all&sort=recency)

                En effet, pour prédire la note d'un client, il est nécessaire d'identifier les entités importantes d’un avis : la note, la localisation, 
                le nom de l'entreprise, la date, ....  
                Mais aussi le commentaire laissé par le client afin d'en extraire le propos global : article défectueux ou conforme? 
                livraison correcte ou problématique? sentiment? ...\n
                """

    webscrapping = """
                Grace à la bibliothèque :orange[*BeautifulSoup*], un programme a été rédigé afin de collecter les données 
                en "webscrappant" le site Trustpilot.
                - Afin d'avoir un jeu de données  consistant, mais aussi pour avoir des avis étalés sur plusieurs mois, le code mis en place 
                a permis de récupérer la totalité des 25009 avis publiés pour Vinted (à la date d'exécution).
                - Afin d'avoir un jeu données équilibré pour les deux entreprises, les 25.000 derniers avis publiés pour Leboncoin 
                ont également été récoltés.\n
                Un jeu de données brut totalisant 50.009 entrées a ainsi été constitué.
                """

    st.subheader("Sources de données")
    st.write(source)
    st.subheader("Web scrapping")
    st.write(webscrapping)
    st.write("---")
    st.write("Le jeu de données brut peut être visible en partie ou en totalité ci-dessous.")

    # ----------- Préparation des données si affichage demandé -----------
    df1 = pd.read_csv(file1, sep= ",")
    df2 = pd.read_csv(file2, sep= ",")
    
    # Concaténation des deux dataframe
    df = pd.concat([df1, df2], axis= 0, ignore_index= True)

    # Tri par date d'avis du plus récent au moins récent
    df = df.sort_values(by= "date de l'avis (GMT+0)", ascending= False)

    # Mise à jour de l'index
    df = df.reset_index(drop= True)
    index_df = range(0, len(df))
    df = df.drop("Unnamed: 0", axis= 1)
    df.insert(0, "id avis", index_df)
    df = df.set_index("id avis")
    
    # ----------- Affichage demandé -----------
    if st.toggle("Afficher les infos du jeu de données brut"):
        # Affichage des infos
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    if st.toggle("Afficher le jeu de données brut", value= True):
        number = st.number_input(":blue[*Nombre de lignes à afficher :*]", min_value=1, max_value= len(df), value= 5)
        # Affichage du df
        st.dataframe(df.head(number))

# ----------------- Page "Jeu de données" ----------------
if sidebar == pages[2]:
    source = """ 
                Les sources de données seront ont été collectées à partir :
                - [Des derniers avis déposés sur Truspilot pour Leboncoin](https://fr.trustpilot.com/review/www.leboncoin.fr?languages=all&sort=recency)
                - [Des derniers avis déposés sur Truspilot pour Vinted](https://fr.trustpilot.com/review/vinted.fr?languages=all&sort=recency)

                En effet, pour prédire la note d'un client, il est nécessaire d'identifier les entités importantes d’un avis : la note, la localisation, 
                le nom de l'entreprise, la date, ....  
                Mais aussi le commentaire laissé par le client afin d'en extraire le propos global : article défectueux ou conforme? 
                livraison correcte ou problématique? sentiment? ...\n
                """

    webscrapping = """
                Grace à la bibliothèque :orange[*BeautifulSoup*], un programme a été rédigé afin de collecter les données 
                en "webscrappant" le site Trustpilot.
                - Afin d'avoir un jeu de données  consistant, mais aussi pour avoir des avis étalés sur plusieurs mois, le code mis en place 
                a permis de récupérer la totalité des 25009 avis (à la date d'exécution) publiés pour Vinted.
                - Afin d'avoir un jeu données équilibré pour les deux entreprises, les 25.000 derniers avis publiés pour Leboncoin 
                ont également été récoltés.\n
                Un jeu de données brut totalisant 50.009 entrées a ainsi été constitué.
                """

    st.subheader("Sources de données")
    st.write(source)
    st.subheader("Web scrapping")
    st.write(webscrapping)
    st.write("---")
    st.write("Le jeu de données peut être visible en partie ou en totalité ci-dessous.")

    # ----------- Préparation des données si affichage demandé -----------
    df1 = pd.read_csv(file1, sep= ",")
    df2 = pd.read_csv(file2, sep= ",")
    
    # Concaténation des deux dataframe
    df = pd.concat([df1, df2], axis= 0, ignore_index= True)

    # Tri par date d'avis du plus récent au moins récent
    df = df.sort_values(by= "date de l'avis (GMT+0)", ascending= False)

    # Mise à jour de l'index
    df = df.reset_index(drop= True)
    index_df = range(0, len(df))
    df = df.drop("Unnamed: 0", axis= 1)
    df.insert(0, "id avis", index_df)
    df = df.set_index("id avis")
    
    # ----------- Affichage demandé -----------
    if st.checkbox("Afficher les infos du jeu de données brut"):
        # Affichage des infos
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    if st.checkbox("Afficher le jeu de données brut"):
        # Affichage du df
        st.dataframe(df)
