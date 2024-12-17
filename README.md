## Description

Ce dépôt contient le rendu du **TP/Atelier: Détection d'anomalies**, réalisé dans le cadre de l'année universitaire 2024-2025 pour le Master 2 IA à l'Université Claude Bernard Lyon 1.

L'objectif de cet atelier est de mettre en œuvre différentes techniques pour détecter les anomalies (outliers et nouveautés) dans des ensembles de données en utilisant Python. Il inclut des approches supervisées et non supervisées pour la détection d'anomalies, en tenant compte des données déséquilibrées.

## Contenu

### Notebook
1. **Détection d'anomalies avec Mouse dataset** :
   - Analyse exploratoire et visualisation.
   - Application des algorithmes :
     - `Isolation Forest`
     - `Local Outlier Factor (LOF)`
   - Détermination des seuils de contamination optimaux.
   - Visualisation des résultats.

2. **Détection de fraudes et d'intrusions réseau** :
   - Préparation et prétraitement des données :
     - Jeux de données : `creditcard.csv`, `Kddcup99`
     - Techniques : Normalisation, Encodage, etc.
   - Comparaison d'approches supervisées et non supervisées :
     - Algorithmes supervisés standards
     - Méthodes gérant les données déséquilibrées
     - `Isolation Forest` et `LOF`.
   - Évaluation : Courbes ROC, Précision/Rappel, métriques adaptées.
   - Méthodologie pour détecter les nouveautés.

### Scripts et Méthodologie
- **Fonctions automatiques** pour exécuter les traitements.
- Documentation détaillée des étapes suivies pour chaque jeu de données.

### Jeux de données
- `mouse.txt` : Analyse graphique et détection d'outliers.
- `creditcard.csv` : Détection de fraudes bancaires (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- `Kddcup99` : Détection d'intrusions réseau (https://www.kaggle.com/datasets/ericzs/kddcup99)