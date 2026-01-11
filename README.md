# Projet d'Analyse des Discours Médiatiques

## 📋 Table des Matières
1. [🎯 Présentation du Projet](#-présentation-du-projet)
2. [📊 Données et Méthodologie](#-données-et-méthodologie)
3. [🛠️ Architecture Technique](#️-architecture-technique)
4. [📈 Résultats et Analyses](#-résultats-et-analyses)
5. [🚀 Déploiement et Utilisation](#-déploiement-et-utilisation)
6. [🔮 Perspectives et Améliorations](#-perspectives-et-améliorations)

## Présentation du Projet

### Contexte
Ce projet vise à analyser les discours médiatiques à grande échelle en utilisant des techniques avancées de Natural Language Processing (NLP). L'objectif est d'identifier des patterns, des biais potentiels et des évolutions dans le traitement médiatique de l'information sur une période de 9 ans.

### Objectifs
- **Analyser** 35,468 articles du Huffington Post (2014-2022)
- **Identifier** des clusters sémantiques dans les discours
- **Détecter** des thèmes récurrents et leur évolution temporelle
- **Évaluer** le ton et le sentiment des articles
- **Créer** un dashboard interactif pour l'exploration des résultats

### Technologies Utilisées
- **NLP** : DistilBERT, LDA, TF-IDF
- **Clustering** : K-means, GMM, HDBSCAN
- **Visualisation** : Plotly, Streamlit, Matplotlib
- **Backend** : PyTorch, Scikit-learn, Pandas
- **Déploiement** : Streamlit Cloud

## 📊 Données et Méthodologie

### Sources de Données
```
Source : Huffington Post (HuffPost)
Période : Avril 2014 - Septembre 2022
Volume : 35,468 articles
Variables : Titre, contenu, date, source
```

### Pipeline d'Analyse
```
1. Collecte & Nettoyage → 2. Embeddings BERT → 3. Clustering
       ↓                         ↓                   ↓
   Texte brut           Représentation vectorielle   Groupes sémantiques
       ↓                         ↓                   ↓
4. Topic Modeling → 5. Analyse Sentiment → 6. Visualisation
       ↓                   ↓                   ↓
   Thèmes identifiés   Ton des articles   Dashboard interactif
```

### Méthodes de Clustering
- **K-means** (k=5) : Méthode principale avec silhouette score 0.035
- **Gaussian Mixture Models** : Alternative pour distributions complexes
- **HDBSCAN** : Clustering basé sur la densité
- **MiniBatch K-means** : Version optimisée pour grandes données

## 🛠️ Architecture Technique

### Structure du Projet
```
biais_mediatique/
├── data/
│   ├── raw/              # Données brutes (non versionnées)
│   └── processed/        # Données transformées
│       ├── news_clean.csv
│       ├── news_embeddings.pt
│       ├── news_with_discourse.csv
│       └── visualizations/
├── src/        
│   ├── models/          # Génération d'embeddings
│   │   └── distilbert_embeddings.py
│   ├── analysis/        # Analyses statistiques
│   │   ├── discourse_analysis.py
│   │   └── comparative_analysis.py
│   ├── app/   # Dashboard
│   │   └── streamlit_app.py
│   └── reporting/       # Génération de rapports
├── requirements.txt     # Dépendances Python
├── runtime.txt         # Version Python pour déploiement
└── README.md          # Documentation
```

### Modèle NLP : DistilBERT
- **Modèle** : `distilbert-base-uncased`
- **Embeddings** : 768 dimensions par article
- **Tokenization** : Longueur maximale = 128 tokens
- **Batch size** : 32 (optimisé pour CPU/GPU modeste)

### Métriques d'Évaluation
```python
# Scores obtenus
silhouette_score = 0.035      # Structure faible mais discernable
sentiment_mean = 0.238        # Ton globalement positif
cluster_balance = "Relativement équilibré"
```

## Résultats et Analyses

### 1. Structure Temporelle
```
📅 Distribution Annuelle (articles)
2014: 2,853    2015: 4,655    2016: 8,179
2017: 11,210   2018: 4,087    2019: 1,265
2020: 1,113    2021: 1,291    2022: 815

Pic d'activité : 2017 (11,210 articles)
Chute post-2017 : -88.8% en 2022
```

### 2. Clusters Sémantiques (5 groupes)
```
🎯 Cluster 0 : 23.0% des articles
🎯 Cluster 1 : 14.3% des articles  
🎯 Cluster 2 : 22.8% des articles
🎯 Cluster 3 : 27.2% des articles (dominant)
🎯 Cluster 4 : 12.7% des articles

📈 Évolution : Cluster 3 devient progressivement dominant à partir de 2016
```

### 3. Thèmes Identifiés (LDA - 8 topics)
```
📚 Thème 0 : Société et vie quotidienne (people, women, school, right)
📚 Thème 1 : Affaires internationales (military, united states, world)
📚 Thème 2 : Santé et politique (health care, republicans, senate)
📚 Thème 3 : Faits divers (police, city, according)
📚 Thème 4 : Justice et droits (court, justice, federal)
📚 Thème 5 : Élections (clinton, voters, democratic, election)
📚 Thème 6 : Trump - campagne (trump, donald, campaign)
📚 Thème 7 : Trump - présidence (president, white house, obama)
```

### 4. Analyse de Sentiment
```
😊 Score moyen : 0.238 (positif)
Répartition :
   • Positif (>0.1) : 56.9% des articles
   • Négatif (<-0.1) : 22.4% des articles  
   • Neutre : 20.8% des articles

Variation temporelle : Relativement stable sur la période
```

### 5. Personnalités Mentionnées
```
Top 10 des personnalités :
1. Trump : 51.9% des articles
2. Obama : 28.5%
3. Clinton : 19.9%
4. Sanders : 9.0%
5. Biden : 4.9%
6. Pence : 4.1%
7. Putin : 4.1%
8. Johnson : 3.3%
9. Harris : 2.4%
10. Pelosi : 2.1%
```

### 6. Complexité Linguistique
```
 Statistiques moyennes :
   • Mots par article : 763 mots
   • Longueur moyenne des mots : 5.0 caractères
   • Complexité par cluster : 703-824 mots/article

Cluster 3 : Articles les plus longs (824 mots en moyenne)
```

##  Déploiement et Utilisation

### Installation Locale
```bash
# 1. Cloner le repository
git clone[ https://github.com/votre-nom/biais_mediatique.git](https://github.com/BintaBall/Analyse-de-Biais-Mediatique.git)
cd biais_mediatique

# 2. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Exécuter le pipeline complet
python src/models/distilbert_embeddings.py
python src/analysis/discourse_analysis.py
python src/analysis/comparative_analysis.py

# 5. Lancer le dashboard
streamlit run src/visualization/streamlit_app.py
```

### Déploiement Streamlit Cloud
1. **Pousser le code sur GitHub**
2. **Se connecter à [share.streamlit.io](https://share.streamlit.io)**
3. **Configurer l'application :**
   - Repository : `[BintaBall/Analyse-de-Biais-Mediatique](https://github.com/BintaBall/Analyse-de-Biais-Mediatique.git)`
   - Branch : `main`
   - Main file : `src/app/streamlit_app.py`
4. **L'application est disponible à :**
   ```
    https://analyse-de-biais-mediatique-bl9pdwfkdrprvvnfdkqw7s.streamlit.app/  
    ```

### Utilisation du Dashboard
Le dashboard offre 5 onglets interactifs :

1. **📈 Distribution** : Clusters et thèmes
2. **📅 Temporel** : Évolution 2014-2022  
3. **📚 Thèmes** : Description des 8 thèmes identifiés
4. **😊 Sentiment** : Analyse du ton des articles
5. **🔍 Exploration** : Recherche par mot-clé

## Perspectives et Améliorations

### Limitations Actuelles
1. **Score silhouette faible** (0.035) : Les clusters ne sont pas bien séparés
2. **Source unique** : Seulement HuffPost analysé
3. **Période limitée** : 2014-2022
4. **Complexité computationnelle** : Génération des embeddings longue

### Améliorations Possibles

#### 1. Extension des Données
```python
# Ajouter d'autres sources
sources_additionnelles = [
    "New York Times",
    "Washington Post", 
    "Fox News",
    "CNN",
    "BBC"
]

# Étendre la période
periode_etendue = "2000-2023"
```

#### 2. Améliorations Techniques
- **Modèles avancés** : RoBERTa, DeBERTa, GPT embeddings
- **Clustering amélioré** : UMAP + HDBSCAN
- **Topic modeling** : BERTopic, Top2Vec
- **Analyse multimodale** : Images + texte

#### 3. Fonctionnalités Supplémentaires
```python
fonctionnalites_futures = [
    "Détection de fake news",
    "Analyse comparative droite/gauche", 
    "Prédiction d'engagement (likes, shares)",
    "Alertes en temps réel sur tendances",
    "API REST pour intégration"
]
```

#### 4. Déploiement Production
- **Conteneurisation** : Docker + Kubernetes
- **Base de données** : PostgreSQL/Elasticsearch
- **Orchestration** : Airflow/Prefect
- **Monitoring** : Prometheus + Grafana

### Applications Pratiques
1. **Journalisme** : Détection de biais, analyse de couverture
2. **Académique** : Études des discours médiatiques
3. **Business Intelligence** : Veille médiatique automatisée
4. **Éducation** : Outil pédagogique sur les médias

##  Conclusion

### Contributions Principales
1. **Pipeline complet** de collecte à visualisation
2. **Analyse à grande échelle** de 35k+ articles
3. **Méthodes multiples** : BERT embeddings + clustering + topic modeling
4. **Dashboard interactif** accessible en ligne
5. **Documentation technique** complète et reproductible

### Insights Clés
- **Continuité sémantique** plutôt que silos distincts
- **Évolution temporelle** marquée (pic 2017, chute postérieure)
- **Ton global positif** mais nuances selon les thèmes
- **Prédominance de Trump** dans le discours médiatique
- **Complexité variable** selon les types d'articles

---
*Ce projet a été développé dans le cadre d'une analyse NLP avancée des discours médiatiques. Les méthodes et résultats sont entièrement reproductibles avec le code fourni.*