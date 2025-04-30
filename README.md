# Détéction de Cancer du sein

Dans ce projet de machine learning, nous analysons le cancer du sein en utilisant le jeu de données Breast Cancer Wisconsin (Diagnostic).

Après un nettoyage des données (gestion des valeurs manquantes, normalisation), nous effectuons une analyse exploratoire pour identifier les caractéristiques clés des tumeurs. 

Enfin, nous entraînons differents models dont un modèle de classification (SVM) pour prédire si une tumeur est bénigne ou maligne.

## Table des matières
1. [Présentation](#présentation)
2. [Données](#données)
3. [Structure du projet](#structure-du-projet)
4. [Installation & Usage](#installation--usage)
5. [Modèles entraînés](#modèles-entraînés)
6. [Résultats](#résultats)
7. [Licence](#licence)


## Données
- **Source** : Breast Cancer Wisconsin (Diagnostic), accessible via ce lien https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

| Nom de la variable                 | Type de donnée | Unité / Mesure          | Description                                                                                           |
|------------------------------------|----------------|-------------------------|-------------------------------------------------------------------------------------------------------|
| `id`                               | Qualitatif     | —                       | Identifiant unique de l’échantillon                                                                  |
| `diagnosis`                        | Qualitatif     | —                       | Diagnostic de la tumeur (« M » = Maligne, « B » = Bénigne)                                           |
| `radius_mean`                      | Quantitatif    | Millimètres (mm)        | Moyenne des distances du centre aux points sur le périmètre                                          |
| `texture_mean`                     | Quantitatif    | Unités arbitraires      | Écart-type des valeurs de gris (texture)                                                             |
| `perimeter_mean`                   | Quantitatif    | Millimètres (mm)        | Moyenne de la longueur du contour                                                                    |
| `area_mean`                        | Quantitatif    | Millimètres carrés (mm²)| Moyenne de l’aire de la tumeur                                                                       |
| `smoothness_mean`                  | Quantitatif    | Unité (ratio)           | Variation locale des longueurs de rayon                                                               |
| `compactness_mean`                 | Quantitatif    | Unité (ratio)           | (Périmètre² / Aire) – 1, mesure de compacité                                                         |
| `concavity_mean`                   | Quantitatif    | Unité (ratio)           | Sévérité des parties concaves du contour                                                             |
| `concave points_mean`              | Quantitatif    | Unité (ratio)           | Nombre de points concaves du contour                                                                 |
| `symmetry_mean`                    | Quantitatif    | Unité (ratio)           | Mesure de la symétrie                                                                                 |
| `fractal_dimension_mean`           | Quantitatif    | Unité (ratio)           | Dimension fractale (« approximation du contour »)                                                    |
| `radius_se`                        | Quantitatif    | Millimètres (mm)        | Erreur standard de `radius_mean`                                                                      |
| `texture_se`                       | Quantitatif    | Unités arbitraires      | Erreur standard de `texture_mean`                                                                     |
| `perimeter_se`                     | Quantitatif    | Millimètres (mm)        | Erreur standard de `perimeter_mean`                                                                   |
| `area_se`                          | Quantitatif    | Millimètres carrés (mm²)| Erreur standard de `area_mean`                                                                        |
| `smoothness_se`                    | Quantitatif    | Unité (ratio)           | Erreur standard de `smoothness_mean`                                                                  |
| `compactness_se`                   | Quantitatif    | Unité (ratio)           | Erreur standard de `compactness_mean`                                                                 |
| `concavity_se`                     | Quantitatif    | Unité (ratio)           | Erreur standard de `concavity_mean`                                                                   |
| `concave points_se`                | Quantitatif    | Unité (ratio)           | Erreur standard de `concave points_mean`                                                              |
| `symmetry_se`                      | Quantitatif    | Unité (ratio)           | Erreur standard de `symmetry_mean`                                                                    |
| `fractal_dimension_se`             | Quantitatif    | Unité (ratio)           | Erreur standard de `fractal_dimension_mean`                                                           |
| `radius_worst`                     | Quantitatif    | Millimètres (mm)        | Valeur la plus élevée parmi les trois plus grandes distances (worst case)                             |
| `texture_worst`                    | Quantitatif    | Unités arbitraires      | Valeur la plus élevée parmi les trois plus grandes textures                                          |
| `perimeter_worst`                  | Quantitatif    | Millimètres (mm)        | Valeur la plus élevée parmi les trois plus grands périmètres                                         |
| `area_worst`                       | Quantitatif    | Millimètres carrés (mm²)| Valeur la plus élevée parmi les trois plus grandes aires                                            |
| `smoothness_worst`                 | Quantitatif    | Unité (ratio)           | Valeur la plus élevée parmi les trois plus grandes smoothness                                        |
| `compactness_worst`                | Quantitatif    | Unité (ratio)           | Valeur la plus élevée parmi les trois plus grandes compactness                                       |
| `concavity_worst`                  | Quantitatif    | Unité (ratio)           | Valeur la plus élevée parmi les trois plus grandes concavités                                        |
| `concave points_worst`             | Quantitatif    | Unité (ratio)           | Valeur la plus élevée parmi les trois plus grands nombres de points concaves                         |
| `symmetry_worst`                   | Quantitatif    | Unité (ratio)           | Valeur la plus élevée parmi les trois plus grandes symétries                                         |
| `fractal_dimension_worst`          | Quantitatif    | Unité (ratio)           | Valeur la plus élevée parmi les trois plus grandes dimensions fractales                              |


## Structure du projet
├── main.ipynb # EDA & Models

├── breast-cancer.csv # Données

## Installation & Usage
```bash
git clone https://…/Detection_Cancer.git
```
## Modèles entraînés
Logistic Regression

SVC

## Résultats

Logistic Regression 
---------------
```bash

Accuracy: 0.9883040935672515
Confusion Matrix:
 [[115   1]
 [  1  54]]
Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.99      0.99       116
           1       0.98      0.98      0.98        55

    accuracy                           0.99       171
   macro avg       0.99      0.99      0.99       171
weighted avg       0.99      0.99      0.99       171

ROC AUC Score: 1.00
Average Precision Score: 1.00
```

SVC
---------------------
```bash

Accuracy: 0.9766081871345029
Confusion Matrix:
 [[115   1]
 [  3  52]]
Classification Report:
               precision    recall  f1-score   support

           0       0.97      0.99      0.98       116
           1       0.98      0.95      0.96        55

    accuracy                           0.98       171
   macro avg       0.98      0.97      0.97       171
weighted avg       0.98      0.98      0.98       171

ROC AUC Score: 0.99
Average Precision Score: 0.99
```
## Licence
Ce projet est sous licence MIT.
