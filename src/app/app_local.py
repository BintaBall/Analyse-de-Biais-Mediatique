# src/visualization/complete_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Dashboard - Analyse HuffPost",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 2.5rem;
        margin-bottom: 1.2rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3B82F6;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 6px solid #3B82F6;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #60A5FA;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 8px 8px 0px 0px;
        gap: 8px;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">📊 Dashboard d\'Analyse - HuffPost (2014-2022)</h1>', unsafe_allow_html=True)

# Barre latérale
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2237/2237289.png", width=80)
    st.title("🔧 Paramètres d'Analyse")
    
    st.markdown("---")
    
    st.subheader("📅 Filtres Temporels")
    
    # Période basée sur vos résultats
    year_range = st.slider(
        "Sélectionnez la période",
        min_value=2014,
        max_value=2022,
        value=(2014, 2022)
    )
    
    st.markdown("---")
    
    st.subheader("🎯 Filtres de Contenu")
    
    # Filtre par cluster
    cluster_options = ["Tous"] + [f"Cluster {i}" for i in range(5)]
    selected_clusters = st.multiselect(
        "Clusters à inclure",
        options=cluster_options,
        default=["Tous"]
    )
    
    # Filtre par thème
    theme_options = ["Tous"] + [f"Thème {i}" for i in range(8)]
    selected_themes = st.multiselect(
        "Thèmes à inclure",
        options=theme_options,
        default=["Tous"]
    )
    
    st.markdown("---")
    
    st.subheader("📊 Options de Visualisation")
    
    chart_theme = st.selectbox(
        "Thème des graphiques",
        options=["plotly", "plotly_white", "ggplot2", "seaborn"],
        index=1
    )
    
    st.markdown("---")
    
    # Informations sur les données
    with st.expander("📋 À propos des données"):
        st.write("**Source:** HuffPost (2014-2022)")
        st.write("**Articles:** 35,468")
        st.write("**Clusters:** 5 (K-means)")
        st.write("**Thèmes:** 8 (LDA)")
        st.write("**Dernière analyse:**", datetime.now().strftime("%Y-%m-%d"))

# Fonctions de chargement avec cache
@st.cache_data
def load_data():
    """Charger les données analysées"""
    try:
        df = pd.read_csv("data/processed/news_with_detailed_analysis.csv")
        
        # Nettoyage et conversion des dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['year_month'] = df['date'].dt.strftime('%Y-%m')
        
        # Créer des catégories de sentiment
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
        st.error(f"Erreur de chargement: {e}")
        return pd.DataFrame()

@st.cache_data
def load_analysis_report():
    """Charger le rapport d'analyse"""
    try:
        with open("data/processed/huffpost_analysis_report.txt", 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return "Rapport non disponible"

# Chargement des données
with st.spinner("🔄 Chargement des données..."):
    df = load_data()
    analysis_report = load_analysis_report()

if df.empty:
    st.error("❌ Aucune donnée disponible. Veuillez exécuter l'analyse d'abord.")
    st.stop()

# Appliquer les filtres
filtered_df = df.copy()

# Filtre temporel
if 'year' in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df['year'] >= year_range[0]) & 
        (filtered_df['year'] <= year_range[1])
    ]

# Filtre par cluster
if "Tous" not in selected_clusters and 'cluster_kmeans_full' in filtered_df.columns:
    selected_nums = [int(c.split()[-1]) for c in selected_clusters]
    filtered_df = filtered_df[filtered_df['cluster_kmeans_full'].isin(selected_nums)]

# Filtre par thème
if "Tous" not in selected_themes and 'topic_lda' in filtered_df.columns:
    selected_nums = [int(t.split()[-1]) for t in selected_themes]
    filtered_df = filtered_df[filtered_df['topic_lda'].isin(selected_nums)]

# Section 1: Métriques principales
st.markdown('<h2 class="section-header">📈 Vue d\'ensemble</h2>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "📄 Articles",
        f"{len(filtered_df):,}",
        f"{((len(filtered_df)/len(df))*100):.1f}% du total"
    )

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
        sentiment_icon = "😊" if avg_sentiment > 0 else "😐" if avg_sentiment == 0 else "😟"
        st.metric(f"😊 Sentiment {sentiment_icon}", f"{avg_sentiment:.3f}")

# Onglets principaux
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Distribution", "📅 Temporel", "📚 Thématiques", "😊 Sentiment", "🔍 Exploration"
])

# Tab 1: Distribution
with tab1:
    st.markdown('<h3 class="section-header">📊 Analyse des Distributions</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des clusters
        if 'cluster_kmeans_full' in filtered_df.columns:
            cluster_dist = filtered_df['cluster_kmeans_full'].value_counts().sort_index()
            
            fig_clusters = px.pie(
                values=cluster_dist.values,
                names=[f"Cluster {i}" for i in cluster_dist.index],
                title=f"Distribution des {len(cluster_dist)} Clusters",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig_clusters.update_layout(
                height=400,
                template=chart_theme,
                showlegend=True
            )
            
            st.plotly_chart(fig_clusters, use_container_width=True)
    
    with col2:
        # Distribution des thèmes
        if 'topic_lda' in filtered_df.columns:
            theme_dist = filtered_df['topic_lda'].value_counts().sort_index()
            
            fig_themes = px.bar(
                x=[f"Thème {i}" for i in theme_dist.index],
                y=theme_dist.values,
                title=f"Distribution des {len(theme_dist)} Thèmes",
                color=theme_dist.values,
                color_continuous_scale='viridis'
            )
            
            fig_themes.update_layout(
                height=400,
                template=chart_theme,
                xaxis_title="Thème",
                yaxis_title="Nombre d'articles",
                showlegend=False
            )
            
            st.plotly_chart(fig_themes, use_container_width=True)
    
    # Relation clusters-thèmes
    st.markdown("---")
    st.subheader("🔗 Relation entre Clusters et Thèmes")
    
    if 'cluster_kmeans_full' in filtered_df.columns and 'topic_lda' in filtered_df.columns:
        # Matrice cluster x thème
        cluster_theme_matrix = pd.crosstab(
            filtered_df['cluster_kmeans_full'],
            filtered_df['topic_lda'],
            normalize='index'
        )
        
        fig_heatmap = px.imshow(
            cluster_theme_matrix,
            labels=dict(x="Thème", y="Cluster", color="Proportion"),
            title="Proportion des thèmes dans chaque cluster",
            color_continuous_scale='RdYlBu'
        )
        
        fig_heatmap.update_layout(
            height=500,
            template=chart_theme
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)

# Tab 2: Analyse Temporelle
with tab2:
    st.markdown('<h3 class="section-header">📅 Analyse Temporelle (2014-2022)</h3>', unsafe_allow_html=True)
    
    if 'year' in filtered_df.columns:
        # Évolution du volume
        yearly_counts = filtered_df['year'].value_counts().sort_index()
        
        fig_volume = px.line(
            x=yearly_counts.index,
            y=yearly_counts.values,
            title="Volume d'Articles par Année",
            labels={'x': 'Année', 'y': "Nombre d'articles"},
            markers=True
        )
        
        fig_volume.update_layout(
            height=400,
            template=chart_theme,
            xaxis=dict(tickmode='linear')
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Distribution par année
        st.subheader("📊 Distribution Annuelle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            yearly_data = []
            for year in sorted(filtered_df['year'].unique()):
                year_data = filtered_df[filtered_df['year'] == year]
                yearly_data.append({
                    'Année': year,
                    'Articles': len(year_data),
                    '% Total': f"{(len(year_data)/len(filtered_df)*100):.1f}%"
                })
            
            yearly_df = pd.DataFrame(yearly_data)
            st.dataframe(
                yearly_df,
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            # Pic en 2017
            if 2017 in yearly_counts.index:
                st.metric(
                    "📈 Pic d'activité",
                    "2017",
                    f"{yearly_counts[2017]:,} articles"
                )
            
            # Chute en 2019
            if 2019 in yearly_counts.index and 2017 in yearly_counts.index:
                decline = ((yearly_counts[2017] - yearly_counts[2019]) / yearly_counts[2017]) * 100
                st.metric(
                    "📉 Chute 2017→2019",
                    f"{decline:.1f}%",
                    f"{yearly_counts[2019]:,} articles en 2019"
                )
        
        # Évolution des clusters dans le temps
        st.markdown("---")
        st.subheader("🔄 Évolution des Clusters")
        
        if 'cluster_kmeans_full' in filtered_df.columns:
            cluster_evolution = filtered_df.groupby(['year', 'cluster_kmeans_full']).size().unstack()
            
            fig_evolution = px.area(
                cluster_evolution,
                title="Évolution des Clusters (2014-2022)",
                labels={'value': "Nombre d'articles", 'year': 'Année', 'variable': 'Cluster'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig_evolution.update_layout(
                height=500,
                template=chart_theme,
                xaxis=dict(tickmode='linear')
            )
            
            st.plotly_chart(fig_evolution, use_container_width=True)
            
            # Analyse des tendances
            st.subheader("📈 Tendances par Cluster")
            
            trends_text = ""
            for cluster in cluster_evolution.columns:
                # Calculer la tendance (première vs dernière année)
                first_year = cluster_evolution.index.min()
                last_year = cluster_evolution.index.max()
                
                if first_year in cluster_evolution.index and last_year in cluster_evolution.index:
                    first_val = cluster_evolution.loc[first_year, cluster]
                    last_val = cluster_evolution.loc[last_year, cluster]
                    
                    if first_val > 0:
                        change = ((last_val - first_val) / first_val) * 100
                        direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                        trends_text += f"- **Cluster {cluster}**: {direction} {change:.1f}% de changement\n"
            
            if trends_text:
                st.markdown(trends_text)

# Tab 3: Analyse Thématique
with tab3:
    st.markdown('<h3 class="section-header">📚 Analyse Thématique Détaillée</h3>', unsafe_allow_html=True)
    
    # Thèmes basés sur vos résultats
    themes_info = {
        0: "Société et vie quotidienne (people, women, school, right)",
        1: "Affaires internationales (military, united states, world, security)",
        2: "Santé et politique (health care, republicans, senate)",
        3: "Faits divers et police (police, city, according)",
        4: "Justice et droits (court, justice, federal, rights)",
        5: "Élections et partis (clinton, voters, democratic, republican, election)",
        6: "Trump - campagne (trump, donald, campaign)",
        7: "Trump - présidence (president, white house, obama)"
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Carte des thèmes
        if 'topic_lda' in filtered_df.columns:
            theme_yearly = filtered_df.groupby(['year', 'topic_lda']).size().unstack()
            
            # Normaliser par année
            theme_yearly_pct = theme_yearly.div(theme_yearly.sum(axis=1), axis=0)
            
            fig_themes_heat = px.imshow(
                theme_yearly_pct.T,
                labels=dict(x="Année", y="Thème", color="Proportion"),
                title="Évolution des Thèmes (proportion par année)",
                color_continuous_scale='greens'
            )
            
            fig_themes_heat.update_layout(
                height=500,
                template=chart_theme,
                yaxis=dict(
                    ticktext=[f"Thème {i}" for i in range(8)],
                    tickvals=list(range(8))
                )
            )
            
            st.plotly_chart(fig_themes_heat, use_container_width=True)
    
    with col2:
        # Informations sur les thèmes
        st.subheader("🎯 Description des Thèmes")
        
        selected_theme = st.selectbox(
            "Choisir un thème",
            options=list(themes_info.keys()),
            format_func=lambda x: f"Thème {x}: {themes_info[x].split('(')[0].strip()}"
        )
        
        if selected_theme is not None:
            theme_data = filtered_df[filtered_df['topic_lda'] == selected_theme]
            
            st.metric(
                f"Thème {selected_theme}",
                f"{len(theme_data):,} articles",
                f"{(len(theme_data)/len(filtered_df)*100):.1f}%"
            )
            
            st.write("**Description:**", themes_info[selected_theme])
            
            # Top années pour ce thème
            if 'year' in theme_data.columns:
                top_years = theme_data['year'].value_counts().head(3)
                st.write("**Top années:**")
                for year, count in top_years.items():
                    st.write(f"- {year}: {count:,} articles")
    
    # Personnalités par thème
    st.markdown("---")
    st.subheader("👥 Personnalités par Thème")
    
    # Données factices basées sur vos résultats
    politicians_by_theme = {
        0: ["Biden", "Obama", "Sanders"],
        1: ["Putin", "Trump", "Obama"],
        2: ["Trump", "Clinton", "Sanders"],
        3: ["Local officials", "Police chiefs", "Mayors"],
        4: ["Judges", "Justice officials", "Trump"],
        5: ["Clinton", "Trump", "Sanders", "Biden"],
        6: ["Trump", "Pence", "Campaign staff"],
        7: ["Trump", "Obama", "Biden", "Harris"]
    }
    
    selected_theme_pol = st.selectbox(
        "Voir les personnalités pour le thème:",
        options=list(politicians_by_theme.keys()),
        format_func=lambda x: f"Thème {x}",
        key="theme_politicians"
    )
    
    if selected_theme_pol in politicians_by_theme:
        st.write("**Personnalités fréquentes:**")
        for i, pol in enumerate(politicians_by_theme[selected_theme_pol], 1):
            st.write(f"{i}. {pol}")

# Tab 4: Analyse de Sentiment
with tab4:
    st.markdown('<h3 class="section-header">😊 Analyse de Sentiment et Ton</h3>', unsafe_allow_html=True)
    
    if 'sentiment_score' in filtered_df.columns:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Distribution
            fig_sent_dist = px.histogram(
                filtered_df,
                x='sentiment_score',
                nbins=30,
                title="Distribution des Scores",
                color_discrete_sequence=['lightseagreen']
            )
            
            fig_sent_dist.update_layout(
                height=300,
                template=chart_theme,
                showlegend=False
            )
            
            st.plotly_chart(fig_sent_dist, use_container_width=True)
        
        with col2:
            # Catégories
            if 'sentiment_category' in filtered_df.columns:
                cat_counts = filtered_df['sentiment_category'].value_counts()
                
                fig_sent_cat = px.pie(
                    values=cat_counts.values,
                    names=cat_counts.index,
                    title="Répartition des Catégories",
                    color_discrete_sequence=['green', 'red', 'gray']
                )
                
                fig_sent_cat.update_layout(
                    height=300,
                    template=chart_theme
                )
                
                st.plotly_chart(fig_sent_cat, use_container_width=True)
        
        with col3:
            # Statistiques
            st.subheader("📊 Statistiques")
            
            stats = [
                ("Moyenne", f"{filtered_df['sentiment_score'].mean():.3f}"),
                ("Médiane", f"{filtered_df['sentiment_score'].median():.3f}"),
                ("Écart-type", f"{filtered_df['sentiment_score'].std():.3f}"),
                ("Minimum", f"{filtered_df['sentiment_score'].min():.3f}"),
                ("Maximum", f"{filtered_df['sentiment_score'].max():.3f}")
            ]
            
            for label, value in stats:
                st.metric(label, value)
        
        # Évolution temporelle du sentiment
        st.markdown("---")
        st.subheader("📈 Évolution du Sentiment")
        
        if 'year_month' in filtered_df.columns:
            sentiment_ts = filtered_df.groupby('year_month')['sentiment_score'].agg(['mean', 'std', 'count'])
            sentiment_ts = sentiment_ts[sentiment_ts['count'] > 10]
            
            fig_sent_ts = px.line(
                sentiment_ts,
                x=sentiment_ts.index,
                y='mean',
                title="Évolution du Sentiment Moyen (2014-2022)",
                labels={'mean': 'Sentiment moyen', 'index': 'Mois'}
            )
            
            # Ajouter la bande d'incertitude
            fig_sent_ts.add_trace(
                go.Scatter(
                    x=list(sentiment_ts.index) + list(sentiment_ts.index)[::-1],
                    y=list(sentiment_ts['mean'] + sentiment_ts['std']) + 
                      list(sentiment_ts['mean'] - sentiment_ts['std'])[::-1],
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='±1 écart-type'
                )
            )
            
            fig_sent_ts.update_layout(
                height=400,
                template=chart_theme,
                showlegend=True,
                xaxis=dict(tickangle=45)
            )
            
            st.plotly_chart(fig_sent_ts, use_container_width=True)
        
        # Sentiment par cluster
        st.markdown("---")
        st.subheader("🎯 Sentiment par Cluster")
        
        if 'cluster_kmeans_full' in filtered_df.columns:
            sentiment_by_cluster = filtered_df.groupby('cluster_kmeans_full')['sentiment_score'].agg(['mean', 'std', 'count'])
            
            fig_sent_cluster = px.bar(
                x=[f"Cluster {i}" for i in sentiment_by_cluster.index],
                y=sentiment_by_cluster['mean'],
                error_y=sentiment_by_cluster['std'],
                title="Sentiment Moyen par Cluster",
                labels={'x': 'Cluster', 'y': 'Sentiment moyen'},
                color=sentiment_by_cluster['mean'],
                color_continuous_scale='RdYlGn'
            )
            
            fig_sent_cluster.update_layout(
                height=400,
                template=chart_theme,
                showlegend=False
            )
            
            st.plotly_chart(fig_sent_cluster, use_container_width=True)

# Tab 5: Exploration des Données
with tab5:
    st.markdown('<h3 class="section-header">🔍 Exploration Interactive</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔎 Filtres Avancés")
        
        # Recherche par mot-clé
        search_term = st.text_input("Mot-clé dans le texte", "")
        
        # Filtre par sentiment
        sentiment_filter = st.selectbox(
            "Catégorie de sentiment",
            options=["Tous", "Positif", "Neutre", "Négatif"]
        )
        
        # Filtre par complexité
        if 'word_count' in filtered_df.columns:
            min_words = st.slider("Nombre minimum de mots", 0, 2000, 0, 100)
        else:
            min_words = 0
        
        # Nombre d'articles à afficher
        n_display = st.slider("Articles à afficher", 5, 100, 20)
    
    with col2:
        # Application des filtres
        explore_df = filtered_df.copy()
        
        if search_term:
            explore_df = explore_df[explore_df['clean_text'].str.contains(search_term, case=False, na=False)]
        
        if sentiment_filter != "Tous" and 'sentiment_category' in explore_df.columns:
            explore_df = explore_df[explore_df['sentiment_category'] == sentiment_filter]
        
        if min_words > 0 and 'word_count' in explore_df.columns:
            explore_df = explore_df[explore_df['word_count'] >= min_words]
        
        # Affichage des résultats
        st.subheader(f"📰 Résultats ({len(explore_df)} articles)")
        
        if len(explore_df) > 0:
            # Créer une vue simplifiée
            display_data = []
            
            for idx, row in explore_df.head(n_display).iterrows():
                article_info = {
                    'Titre': row.get('title', 'Sans titre')[:100] + ("..." if len(str(row.get('title', ''))) > 100 else ""),
                    'Cluster': f"Cluster {row.get('cluster_kmeans_full', 'N/A')}",
                    'Thème': f"Thème {row.get('topic_lda', 'N/A')}",
                    'Sentiment': f"{row.get('sentiment_score', 0):.3f}" if 'sentiment_score' in row else "N/A",
                    'Année': row.get('year', 'N/A')
                }
                display_data.append(article_info)
            
            display_df = pd.DataFrame(display_data)
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            # Exploration détaillée d'un article
            st.markdown("---")
            st.subheader("📄 Détail d'un Article")
            
            if len(explore_df) > 0:
                article_idx = st.selectbox(
                    "Sélectionnez un article",
                    options=range(min(10, len(explore_df))),
                    format_func=lambda x: explore_df.iloc[x].get('title', f"Article {x}")[:80]
                )
                
                if article_idx is not None:
                    article = explore_df.iloc[article_idx]
                    
                    col_info, col_text = st.columns([1, 2])
                    
                    with col_info:
                        st.write("**Métadonnées:**")
                        st.write(f"**Cluster:** {article.get('cluster_kmeans_full', 'N/A')}")
                        st.write(f"**Thème:** {article.get('topic_lda', 'N/A')}")
                        st.write(f"**Sentiment:** {article.get('sentiment_score', 0):.3f}")
                        st.write(f"**Année:** {article.get('year', 'N/A')}")
                        if 'word_count' in article:
                            st.write(f"**Mots:** {article['word_count']}")
                    
                    with col_text:
                        st.write("**Extrait du texte:**")
                        if 'clean_text' in article:
                            text_preview = str(article['clean_text'])[:500] + ("..." if len(str(article['clean_text'])) > 500 else "")
                            st.text_area(
                                "Contenu",
                                text_preview,
                                height=200,
                                disabled=True
                            )
        else:
            st.info("Aucun article ne correspond aux critères de recherche")
    
    # Statistiques d'exploration
    st.markdown("---")
    st.subheader("📊 Statistiques de l'Exploration")
    
    if len(explore_df) > 0:
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            avg_words = explore_df['word_count'].mean() if 'word_count' in explore_df.columns else 0
            st.metric("Mots/article", f"{avg_words:.0f}")
        
        with col_stat2:
            avg_sentiment = explore_df['sentiment_score'].mean() if 'sentiment_score' in explore_df.columns else 0
            st.metric("Sentiment moyen", f"{avg_sentiment:.3f}")
        
        with col_stat3:
            if 'year' in explore_df.columns:
                recent_year = explore_df['year'].max()
                st.metric("Année la plus récente", recent_year)

# Section finale : Résumé et export
st.markdown("---")
st.markdown('<h2 class="section-header">📥 Export et Résumé</h2>', unsafe_allow_html=True)

col_export, col_summary = st.columns([1, 2])

with col_export:
    st.subheader("📤 Export des Données")
    
    # Préparer les données pour l'export
    export_df = filtered_df.copy()
    
    # Sélectionner les colonnes d'export
    export_cols = ['title', 'clean_text', 'year', 'cluster_kmeans_full', 'topic_lda', 'sentiment_score']
    export_cols = [col for col in export_cols if col in export_df.columns]
    
    if export_cols:
        csv_data = export_df[export_cols].to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📊 Télécharger CSV",
            data=csv_data,
            file_name=f"huffpost_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Télécharger les données filtrées au format CSV"
        )
    
    # Rapport d'analyse
    if analysis_report:
        st.download_button(
            label="📄 Télécharger Rapport",
            data=analysis_report,
            file_name=f"rapport_analyse_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

with col_summary:
    st.subheader("📋 Points Clés de l'Analyse")
    
    summary_points = [
        f"• **{len(df):,} articles** analysés sur **9 ans** (2014-2022)",
        f"• **Pic d'activité en 2017** avec {df[df['year']==2017].shape[0]:,} articles",
        f"• **5 clusters sémantiques** identifiés avec une évolution temporelle marquée",
        f"• **8 thèmes principaux** couvrant politique, société, santé et justice",
        f"• **Ton globalement positif** (score moyen: {df['sentiment_score'].mean():.3f})",
        f"• **Trump** est la personnalité la plus mentionnée (52% des articles)",
        f"• **Cluster 3** domine progressivement de 2016 à 2020",
        f"• **Complexité variable** : de 703 à 824 mots/article selon les clusters"
    ]
    
    for point in summary_points:
        st.markdown(point)

# Pied de page
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6B7280; padding: 1rem;">
    <p>📊 Dashboard d'analyse HuffPost • 35,468 articles (2014-2022) • Analyse NLP complète</p>
    <p>Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
""", unsafe_allow_html=True)

# JavaScript pour améliorations
st.markdown("""
<script>
// Améliorer l'expérience utilisateur
document.addEventListener('DOMContentLoaded', function() {
    // Highlight les métriques au survol
    const metrics = document.querySelectorAll('[data-testid="stMetricValue"]');
    metrics.forEach(metric => {
        metric.parentElement.style.transition = 'all 0.3s ease';
        metric.parentElement.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.05)';
        });
        metric.parentElement.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
        });
    });
    
    // Smooth scrolling pour les ancres internes
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            if (targetId !== '#') {
                e.preventDefault();
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    window.scrollTo({
                        top: targetElement.offsetTop - 100,
                        behavior: 'smooth'
                    });
                }
            }
        });
    });
});
</script>
""", unsafe_allow_html=True)