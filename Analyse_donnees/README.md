
# Analyse des données – Station de pesée chauffée

Projet GMC-3005 – Transferts thermiques

## Structure de ce dossier

```
Analyse_donnees/
│── data/                         # CSV et Excel de la campagne de mesures
│── notebooks/                    # Jupyter notebooks pour l’analyse
│── scripts/                      # Scripts Python propres et réutilisables
│── figures/                      # Images générées (stratification, heatmaps, courbes temporelles)
│── README.md                     # Documentation actuelle
```

---

# Objectif de l’analyse (Mandat 2.1)

À partir du jeu de données fourni par l’équipe de recherche (Tohidi et al.), nous devons :

1. Détecter et quantifier la stratification thermique dans le puits.
2. Analyser la température moyenne sous chaque plateau (P1–P6) au cours du temps.
3. Déduire les règles de contrôle des aérothermes basées sur les mesures observées.

L’analyse se déroule de décembre 2023 à mars 2024 et couvre 87 capteurs mesurés toutes les 2 minutes.

---

# Dataset utilisé

Le dataset contient :

* Température extérieure et humidité
* Températures dans le puits à trois hauteurs :

  * LOW – 0.40 m
  * MID – 0.80 m
  * TOP – 1.20 m
* 29 positions (P1 à P29) × 3 niveaux = 87 capteurs

Points importants à noter :

* Certains capteurs ont cessé de fonctionner en cours de route, entraînant des valeurs NaN.
* Avant le 26 janvier 2024, l’aérotherme sous P6 était hors fonction, créant une zone froid anormale.

---

# Prétraitement des données

Les données originales ont été filtrées avant publication :

* Retrait des valeurs négatives
* Retrait des valeurs inférieures à 0.1 V ou supérieures à 4.9 V
* Conversion tension → résistance → température via l’équation de Steinhart-Hart.

Ces conversions ne sont pas incluses dans ce dossier, car les données fournies sont déjà en degrés Celsius.

---

# Étapes d’analyse

## 1. Importation et préparation des données

* Chargement du CSV (datetime et champs numériques)
* Nettoyage des NaN
* Séparation des capteurs en trois niveaux : LOW, MID, TOP
* Attribution des positions (P1–P29) en fonction des en-têtes de colonnes

## 2. Détection de la stratification thermique

Méthodes utilisées :

### Méthode qualitative

Tracer la température moyenne LOW, MID et TOP en fonction du temps.
Une stratification est observée lorsque T_top > T_mid > T_low.

### Méthode quantitative

Calcul du gradient vertical pour chaque position :

* Gradient > 0 : stratification
* Gradient ≈ 0 : mélange
* Gradient < 0 : inversion thermique

Les capteurs près de P6 montrent des comportements anormaux avant le 26 janvier en raison de l’aérotherme défectueux.

## 3. Température moyenne sous chaque plateau

Pour chaque plateau, les capteurs LOW, MID et TOP associés aux positions correspondantes sont regroupés.

Cette analyse permet d’identifier les zones plus chaudes ou plus froides du puits, ainsi que l’effet des aérothermes.

## 4. Déduction des règles de contrôle des aérothermes

D’après l’énoncé, les aérothermes fonctionnent selon :

* Mise en marche lorsque la température intérieure est inférieure à 3 °C
* Arrêt lorsque la température extérieure dépasse 3 °C

L’analyse consiste à :

* Identifier les cycles ON/OFF par la dérivée temporelle des températures
* Corréler température extérieure et intérieure
* Observer les périodes de chauffage continu (surtout avant le 26 janvier)

Cette approche permet d’inférer les règles réelles de contrôle, qui peuvent différer du modèle théorique.

# Auteurs

Projet réalisé dans le cadre du TP GMC-3005 – Transferts thermiques,
Université Laval, Hiver 2025.

---




