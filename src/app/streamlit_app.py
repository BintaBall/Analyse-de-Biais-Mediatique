"""
📊 Dashboard d'Analyse des Discours Médiatiques - HuffPost
Déploiement Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import os
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyse HuffPost - Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #3B82F6;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #60A5FA;
        margin: 0.8rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 8px 12px;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">📊 Analyse des Discours Médiatiques - HuffPost</h1>', unsafe_allow_html=True)

# Fonctions de chargement avec gestion d'erreur
@st.cache_data(ttl=3600)  # Cache pour 1 heure
def load_data():
    """Charger les données avec gestion d'erreur"""
    try:
        # Essayer différents chemins (pour Streamlit Cloud)
        possible_paths = [
            "data/processed/news_with_detailed_analysis.csv",
            "../data/processed/news_with_detailed_analysis.csv",
            "./news_with_detailed_analysis.csv"
        ]
        
        df = None
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    st.sidebar.success(f"✅ Données chargées depuis: {path}")
                    break
            except:
                continue
        
        if df is None:
            # Créer des données de démo si aucun fichier trouvé
            st.sidebar.warning("⚠️ Données non trouvées. Chargement des données de démo.")
            
            # Données de démo basées sur vos résultats
            dates = pd.date_range(start='2014-01-01', end='2022-12-31', periods=1000)
            df = pd.DataFrame({
                'date': np.random.choice(dates, 1000),
                'title': [f"Article {i}" for i in range(1000)],
                'clean_text': ["Texte d'exemple " * 20 for _ in range(1000)],
                'cluster_kmeans_full': np.random.randint(0, 5, 1000),
                'topic_lda': np.random.randint(0, 8, 1000),
                'sentiment_score': np.random.uniform(-0.5, 0.5, 1000),
                'word_count': np.random.randint(500, 1000, 1000)
            })
        
        # Nettoyage et transformation
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['year_month'] = df['date'].dt.strftime('%Y-%m')
        
        # Catégoriser les sentiments
        if 'sentiment_score' in df.columns:
            def categorize_sentiment(score):
                if score > 0.1:
                    return "Positif"
                elif score < -0.1:
                    return "Négatif"
                else:
                    return "Neutre"
            
            df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)
        
        return df
    
    except Exception as e:
        st.error(f"❌ Erreur de chargement des données: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_report():
    """Charger le rapport d'analyse"""
    try:
        possible_paths = [
            "data/processed/huffpost_analysis_report.txt",
            "../data/processed/huffpost_analysis_report.txt",
            "./huffpost_analysis_report.txt"
        ]
        
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        return f.read()
            except:
                continue
        
        # Rapport de démo
        return """
        ANALYSE HUFFPOST (2014-2022)
        =============================
        
        Articles analysés: 35,468
        Période: 2014-2022
        Clusters identifiés: 5
        Thèmes principaux: 8
        
        Points clés:
        • Pic d'activité en 2017
        • Ton globalement positif
        • Trump mentionné dans 52% des articles
        • Cluster 3 dominant de 2016 à 2020
        """
    
    except:
        return "Rapport non disponible"

# Barre latérale
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2237/2237289.png", width=70)
    st.title("🔧 Paramètres")
    
    st.markdown("---")
    
    # Filtre par année
    st.subheader("📅 Période")
    year_range = st.slider(
        "Sélectionnez les années",
        2014, 2022, (2014, 2022)
    )
    
    st.markdown("---")
    
    # Informations
    st.subheader("ℹ️ À propos")
    st.write("""
    **Source:** HuffPost  
    **Période:** 2014-2022  
    **Articles:** 35,468  
    **Techniques:** NLP, Clustering, Topic Modeling
    """)
    
    # Lien vers le code
    st.markdown("---")
    st.write("**📁 Code source:**")
    st.markdown("[GitHub Repository](https://github.com)")
    
    # Statut
    st.markdown("---")
    st.write(f"**🔄 Dernière mise à jour:**")
    st.write(datetime.now().strftime("%Y-%m-%d %H:%M"))

# Chargement des données
with st.spinner("Chargement des données..."):
    df = load_data()
    analysis_report = load_report()

if df.empty:
    st.error("Aucune donnée disponible. Vérifiez les fichiers de données.")
    st.stop()

# Appliquer les filtres
filtered_df = df.copy()
if 'year' in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df['year'] >= year_range[0]) & 
        (filtered_df['year'] <= year_range[1])
    ]

# Métriques principales
st.markdown('<h2 class="section-header">📊 Vue d\'ensemble</h2>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_articles = len(filtered_df)
    st.metric("📄 Articles", f"{total_articles:,}")

with col2:
    if 'year' in filtered_df.columns:
        years = filtered_df['year'].nunique()
        st.metric("📅 Années", years)

with col3:
    if 'cluster_kmeans_full' in filtered_df.columns:
        clusters = filtered_df['cluster_kmeans_full'].nunique()
        st.metric("🎯 Clusters", clusters)

with col4:
    if 'topic_lda' in filtered_df.columns:
        themes = filtered_df['topic_lda'].nunique()
        st.metric("📚 Thèmes", themes)

with col5:
    if 'sentiment_score' in filtered_df.columns:
        avg_sentiment = filtered_df['sentiment_score'].mean()
        icon = "😊" if avg_sentiment > 0.1 else "😐" if avg_sentiment > -0.1 else "😟"
        st.metric(f"😊 Sentiment {icon}", f"{avg_sentiment:.3f}")

# Onglets
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Distribution", "📅 Temporel", "📚 Thèmes", "😊 Sentiment", "🔍 Exploration"
])

# Tab 1: Distribution
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        if 'cluster_kmeans_full' in filtered_df.columns:
            cluster_dist = filtered_df['cluster_kmeans_full'].value_counts().sort_index()
            
            fig_clusters = px.pie(
                values=cluster_dist.values,
                names=[f"Cluster {i}" for i in cluster_dist.index],
                title="Distribution des Clusters"
            )
            st.plotly_chart(fig_clusters, use_container_width=True)
    
    with col2:
        if 'topic_lda' in filtered_df.columns:
            theme_dist = filtered_df['topic_lda'].value_counts().sort_index()
            
            fig_themes = px.bar(
                x=[f"Thème {i}" for i in theme_dist.index],
                y=theme_dist.values,
                title="Distribution des Thèmes"
            )
            st.plotly_chart(fig_themes, use_container_width=True)

# Tab 2: Temporel
with tab2:
    if 'year' in filtered_df.columns:
        yearly_counts = filtered_df['year'].value_counts().sort_index()
        
        fig_volume = px.line(
            x=yearly_counts.index,
            y=yearly_counts.values,
            title="Volume d'Articles par Année"
        )
        st.plotly_chart(fig_volume, use_container_width=True)

# Tab 3: Thèmes
with tab3:
    st.write("**Thèmes identifiés:**")
    
    themes_info = [
        "0: Société et vie quotidienne",
        "1: Affaires internationales",
        "2: Santé et politique",
        "3: Faits divers",
        "4: Justice et droits",
        "5: Élections et partis",
        "6: Trump - campagne",
        "7: Trump - présidence"
    ]
    
    for theme_info in themes_info:
        st.write(f"• {theme_info}")

# Tab 4: Sentiment
with tab4:
    if 'sentiment_score' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dist = px.histogram(
                filtered_df,
                x='sentiment_score',
                title="Distribution des Scores"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            if 'sentiment_category' in filtered_df.columns:
                cat_counts = filtered_df['sentiment_category'].value_counts()
                
                fig_cat = px.pie(
                    values=cat_counts.values,
                    names=cat_counts.index,
                    title="Catégories de Sentiment"
                )
                st.plotly_chart(fig_cat, use_container_width=True)

# Tab 5: Exploration
with tab5:
    st.subheader("🔍 Exploration des Articles")
    
    if 'clean_text' in filtered_df.columns:
        search_term = st.text_input("Rechercher un mot-clé:")
        
        if search_term:
            results = filtered_df[filtered_df['clean_text'].str.contains(search_term, case=False, na=False)]
            st.write(f"**{len(results)} articles trouvés:**")
            
            if len(results) > 0:
                for idx, row in results.head(5).iterrows():
                    title = row.get('title', 'Sans titre')[:80]
                    st.write(f"• {title}...")

# Rapport d'analyse
with st.expander("📋 Voir le rapport complet"):
    st.text(analysis_report)

# Export
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Exporter les données"):
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger CSV",
            data=csv,
            file_name=f"huffpost_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🖼️ Exporter les graphiques"):
        st.info("Fonctionnalité d'export d'images à venir")

# Pied de page
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>Dashboard d'analyse HuffPost • Déployé sur Streamlit Cloud</p>
    <p>Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>
""", unsafe_allow_html=True)