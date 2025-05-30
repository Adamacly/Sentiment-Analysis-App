# SENTIMENT ANALYSIS APP

Une application de classification de sentiments utilisant des modèles de traitement du langage naturel (NLP) avec une interface utilisateur Gradio.

## Description

Cette application analyse le sentiment de et les classe en trois catégories :
- **Positive** : Sentiment positif
- **Neutral** : Sentiment neutre  
- **Negative** : Sentiment négatif

L'application utilise le modèle `cardiffnlp/twitter-roberta-base-sentiment` basé sur RoBERTa.

## Fonctionnalités

- Interface web intuitive avec Gradio
- Classification de sentiments en temps réel
- Affichage du score de confiance (en pourcentage)
- Mesure du temps de prédiction
- Support de textes longs (jusqu'à 10 lignes)

## Installation

### 1. Clonez ce repository :
```bash
git clone https://github.com/Adamacly/Sentiment-Analysis-App.git
cd Sentiment-Analysis-App
```

### 2. Installez les dépendances :

```bash
pip install -r requirements.txt
```

### 3. Utilisation
```bash
python app.py
```