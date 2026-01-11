# src/analysis/single_source_analysis.py
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, OPTICS
import warnings
from collections import Counter
from datetime import datetime
import re
warnings.filterwarnings('ignore')

print("="*70)
print("📰 ANALYSE DU DISCOURS - HUFFINGTON POST")
print("="*70)

# 1. Chargement des données
print("\n📥 Chargement des données...")

df = pd.read_csv("data/processed/news_with_discourse.csv")
embeddings = torch.load("data/processed/news_embeddings.pt").numpy()

print(f"✅ Articles analysés: {len(df):,}")
print(f"✅ Source unique: {df['source'].iloc[0] if 'source' in df.columns else 'Huffington Post'}")

# 2. Analyse temporelle
print("\n" + "="*70)
print("📅 ANALYSE 1: Évolution Temporelle")
print("="*70)

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Extraction d'informations temporelles
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['year_month'] = df['date'].dt.to_period('M')
    
    print(f"\n📊 Période couverte: {df['date'].min().strftime('%Y-%m-%d')} à {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"📈 Nombre d'années: {df['year'].nunique()}")
    
    # Distribution par année
    yearly_counts = df['year'].value_counts().sort_index()
    print("\n📅 Distribution par année:")
    for year, count in yearly_counts.items():
        print(f"   • {year}: {count:,} articles")
    
    # Évolution des clusters dans le temps
    if 'cluster_kmeans_full' in df.columns:
        cluster_evolution = df.groupby(['year', 'cluster_kmeans_full']).size().unstack()
        
        print("\n🔄 Évolution des clusters par année (%):")
        for year in cluster_evolution.index:
            year_data = cluster_evolution.loc[year]
            total = year_data.sum()
            print(f"\n   {year}:")
            for cluster in year_data.index:
                percentage = (year_data[cluster] / total * 100)
                if percentage > 5:  # Afficher seulement les clusters significatifs
                    print(f"     • Cluster {cluster}: {percentage:.1f}%")
else:
    print("⚠️ Colonne 'date' non disponible pour l'analyse temporelle")

# 3. Analyse thématique approfondie
print("\n" + "="*70)
print("📚 ANALYSE 2: Thématiques (Topic Modeling)")
print("="*70)

print("\n🔍 Application de LDA pour identifier les thèmes...")

# Préparation du texte
texts = df['clean_text'].fillna('').astype(str).tolist()

# Vectorisation
vectorizer = CountVectorizer(
    max_features=2000,
    min_df=10,
    max_df=0.8,
    stop_words='english',
    ngram_range=(1, 2)  # Unigrammes et bigrammes
)

X_counts = vectorizer.fit_transform(texts)
print(f"✅ Vocabulaire: {X_counts.shape[1]} termes")

# LDA
n_topics = 8
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    learning_method='online',
    max_iter=20,
    learning_offset=50.
)

lda.fit(X_counts)
topic_distribution = lda.transform(X_counts)
df['topic_lda'] = topic_distribution.argmax(axis=1)

print(f"\n🎯 {n_topics} thèmes identifiés:")
print("-" * 80)

# Afficher les mots-clés par thème
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[-15:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    
    # Filtrer les mots courts/inutiles
    meaningful_words = [w for w in top_words[:10] if len(w) > 3 and not w.isdigit()]
    
    print(f"\n   Thème {topic_idx} ({df['topic_lda'].value_counts().get(topic_idx, 0):,} articles):")
    print(f"   • {', '.join(meaningful_words[:7])}")

# 4. Analyse des sentiments (approximation)
print("\n" + "="*70)
print("😊 ANALYSE 3: Ton et Émotion (Lexique)")
print("="*70)

print("\n🔍 Analyse du ton des articles...")

# Charger un lexique de sentiments (simplifié)
positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'win', 'happy', 
                  'hope', 'improve', 'better', 'progress', 'support', 'love', 'peace']
negative_words = ['bad', 'terrible', 'negative', 'failure', 'lose', 'sad', 'war', 
                  'attack', 'crisis', 'problem', 'danger', 'threat', 'hate', 'violence']

def estimate_sentiment(text):
    """Estimation simple du sentiment"""
    if not isinstance(text, str):
        return 0
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count + negative_count > 0:
        return (positive_count - negative_count) / (positive_count + negative_count)
    return 0

# Appliquer à un échantillon pour éviter d'être trop lent
sample_size = min(5000, len(df))
sample_df = df.sample(sample_size, random_state=42)

sample_df['sentiment_score'] = sample_df['clean_text'].apply(estimate_sentiment)
df['sentiment_score'] = df.index.map(
    sample_df.set_index(sample_df.index)['sentiment_score'].to_dict()
).fillna(0)

print(f"\n📊 Analyse de sentiment ({sample_size} articles échantillonnés):")
print(f"   • Score moyen: {sample_df['sentiment_score'].mean():.3f}")
print(f"   • Articles positifs (>0.1): {(sample_df['sentiment_score'] > 0.1).sum()} ({(sample_df['sentiment_score'] > 0.1).sum()/sample_size*100:.1f}%)")
print(f"   • Articles négatifs (<-0.1): {(sample_df['sentiment_score'] < -0.1).sum()} ({(sample_df['sentiment_score'] < -0.1).sum()/sample_size*100:.1f}%)")
print(f"   • Articles neutres: {(abs(sample_df['sentiment_score']) <= 0.1).sum()} ({(abs(sample_df['sentiment_score']) <= 0.1).sum()/sample_size*100:.1f}%)")

# 5. Analyse des entités nommées (approximation)
print("\n" + "="*70)
print("🏛️ ANALYSE 4: Personnalités et Lieux Mentionnés")
print("="*70)

print("\n🔍 Extraction des entités fréquentes...")

# Liste simplifiée de personnalités politiques américaines
politicians = ['trump', 'obama', 'biden', 'clinton', 'sanders', 'putin', 
               'merkel', 'macron', 'johnson', 'pence', 'harris', 'pelosi']

# Mots typiquement géopolitiques
countries = ['usa', 'us', 'america', 'china', 'russia', 'uk', 'britain', 
             'france', 'germany', 'iran', 'north korea', 'israel']

def extract_mentions(text):
    """Extraire les mentions de personnalités et pays"""
    if not isinstance(text, str):
        return {'politicians': [], 'countries': []}
    
    text_lower = text.lower()
    found_politicians = [p for p in politicians if p in text_lower]
    found_countries = [c for c in countries if c in text_lower]
    
    return {
        'politicians': found_politicians,
        'countries': found_countries
    }

# Analyser un échantillon
sample_texts = sample_df['clean_text'].fillna('').tolist()
all_mentions = [extract_mentions(text) for text in sample_texts]

# Compter les occurrences
politician_counts = Counter()
country_counts = Counter()

for mentions in all_mentions:
    politician_counts.update(mentions['politicians'])
    country_counts.update(mentions['countries'])

print(f"\n👥 Personnalités les plus mentionnées:")
for politician, count in politician_counts.most_common(10):
    percentage = count / sample_size * 100
    print(f"   • {politician.title():15} : {count:4,d} mentions ({percentage:.1f}%)")

print(f"\n🌍 Pays les plus mentionnés:")
for country, count in country_counts.most_common(8):
    percentage = count / sample_size * 100
    print(f"   • {country.title():15} : {count:4,d} mentions ({percentage:.1f}%)")

# 6. Analyse de complexité linguistique
print("\n" + "="*70)
print("📝 ANALYSE 5: Complexité Linguistique")
print("="*70)

print("\n🔍 Mesure de la complexité des textes...")

def analyze_text_complexity(text):
    """Analyser la complexité d'un texte"""
    if not isinstance(text, str):
        return {'word_count': 0, 'sentence_count': 0, 'avg_word_length': 0}
    
    # Nombre de mots
    words = text.split()
    word_count = len(words)
    
    # Nombre approximatif de phrases
    sentence_count = len(re.split(r'[.!?]+', text))
    
    # Longueur moyenne des mots
    if word_count > 0:
        avg_word_length = sum(len(w) for w in words) / word_count
    else:
        avg_word_length = 0
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length
    }

# Analyser un échantillon
complexity_results = sample_df['clean_text'].apply(analyze_text_complexity)
complexity_df = pd.DataFrame(complexity_results.tolist())

print(f"\n📊 Statistiques de complexité:")
print(f"   • Mots par article (moyenne): {complexity_df['word_count'].mean():.0f}")
print(f"   • Phrases par article (moyenne): {complexity_df['sentence_count'].mean():.1f}")
print(f"   • Longueur moyenne des mots: {complexity_df['avg_word_length'].mean():.1f} caractères")

# Relation entre complexité et cluster
if 'cluster_kmeans_full' in df.columns:
    sample_df = sample_df.copy()
    sample_df['word_count'] = complexity_df['word_count']
    
    print(f"\n📈 Complexité par cluster:")
    for cluster in sorted(df['cluster_kmeans_full'].unique()):
        cluster_mask = sample_df['cluster_kmeans_full'] == cluster
        if cluster_mask.sum() > 0:
            avg_words = sample_df[cluster_mask]['word_count'].mean()
            print(f"   • Cluster {cluster}: {avg_words:.0f} mots en moyenne")

# 7. Visualisations
print("\n" + "="*70)
print("🎨 GÉNÉRATION DES VISUALISATIONS")
print("="*70)

print("\n📊 Création des graphiques...")

fig = plt.figure(figsize=(20, 15))

# Subplot 1: Évolution temporelle
ax1 = plt.subplot(2, 3, 1)
if 'year' in df.columns and 'cluster_kmeans_full' in df.columns:
    yearly_cluster = df.groupby(['year', 'cluster_kmeans_full']).size().unstack()
    yearly_cluster.plot(kind='area', stacked=True, ax=ax1, alpha=0.7)
    ax1.set_title('Évolution des Clusters (2016-2020)')
    ax1.set_xlabel('Année')
    ax1.set_ylabel("Nombre d'articles")
    ax1.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
else:
    ax1.text(0.5, 0.5, 'Données temporelles\nnon disponibles', 
             ha='center', va='center', fontsize=12)
    ax1.axis('off')

# Subplot 2: Distribution des thèmes LDA
ax2 = plt.subplot(2, 3, 2)
if 'topic_lda' in df.columns:
    topic_counts = df['topic_lda'].value_counts().sort_index()
    colors = plt.cm.Set3(np.linspace(0, 1, len(topic_counts)))
    bars = ax2.bar(range(len(topic_counts)), topic_counts.values, color=colors)
    ax2.set_title(f'Distribution des {len(topic_counts)} Thèmes (LDA)')
    ax2.set_xlabel('Thème')
    ax2.set_ylabel("Nombre d'articles")
    ax2.set_xticks(range(len(topic_counts)))
    ax2.set_xticklabels([f'T{i}' for i in range(len(topic_counts))])
    
    # Ajouter les comptes
    for bar, count in zip(bars, topic_counts.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{count:,}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)

# Subplot 3: Analyse de sentiment
ax3 = plt.subplot(2, 3, 3)
sentiment_bins = np.linspace(-1, 1, 21)
ax3.hist(sample_df['sentiment_score'], bins=sentiment_bins, 
        alpha=0.7, color='skyblue', edgecolor='black')
ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutre')
ax3.axvline(x=sample_df['sentiment_score'].mean(), color='green', 
           linestyle='-', alpha=0.7, label=f'Moyenne: {sample_df["sentiment_score"].mean():.3f}')
ax3.set_title('Distribution des Scores de Sentiment')
ax3.set_xlabel('Score de sentiment')
ax3.set_ylabel("Nombre d'articles")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Personnalités mentionnées
ax4 = plt.subplot(2, 3, 4)
if politician_counts:
    top_politicians = politician_counts.most_common(8)
    politicians_names = [p[0].title() for p in top_politicians]
    politicians_counts = [p[1] for p in top_politicians]
    
    y_pos = np.arange(len(politicians_names))
    bars = ax4.barh(y_pos, politicians_counts, color='lightcoral')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(politicians_names)
    ax4.invert_yaxis()
    ax4.set_title('Personnalités les Plus Mentionnées')
    ax4.set_xlabel("Nombre d'articles mentionnant")
    
    # Ajouter les pourcentages
    for i, (bar, count) in enumerate(zip(bars, politicians_counts)):
        percentage = count / sample_size * 100
        ax4.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                f'{percentage:.1f}%', ha='left', va='center')

# Subplot 5: Complexité par cluster
ax5 = plt.subplot(2, 3, 5)
if 'cluster_kmeans_full' in df.columns and 'word_count' in sample_df.columns:
    cluster_complexity = []
    cluster_labels = []
    
    for cluster in sorted(df['cluster_kmeans_full'].unique()):
        cluster_mask = sample_df['cluster_kmeans_full'] == cluster
        if cluster_mask.sum() > 0:
            avg_words = sample_df[cluster_mask]['word_count'].mean()
            cluster_complexity.append(avg_words)
            cluster_labels.append(f'Cluster {cluster}')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_complexity)))
    bars = ax5.bar(range(len(cluster_complexity)), cluster_complexity, color=colors)
    ax5.set_title('Complexité (mots/article) par Cluster')
    ax5.set_xlabel('Cluster')
    ax5.set_ylabel('Mots par article (moyenne)')
    ax5.set_xticks(range(len(cluster_complexity)))
    ax5.set_xticklabels(cluster_labels)
    
    for bar, value in zip(bars, cluster_complexity):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{value:.0f}', ha='center', va='bottom')

# Subplot 6: Résumé statistique
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = f"RÉSUMÉ DE L'ANALYSE\n\n"
summary_text += f"Source: Huffington Post\n"
summary_text += f"Articles: {len(df):,}\n"
summary_text += f"Période: {df['date'].min().strftime('%Y-%m') if 'date' in df.columns else 'N/A'} à {df['date'].max().strftime('%Y-%m') if 'date' in df.columns else 'N/A'}\n\n"

if 'year' in df.columns:
    summary_text += f"Distribution annuelle:\n"
    for year, count in yearly_counts.items():
        summary_text += f"• {year}: {count:,}\n"
    summary_text += "\n"

summary_text += f"Thèmes identifiés: {n_topics}\n"
summary_text += f"Sentiment moyen: {sample_df['sentiment_score'].mean():.3f}\n"
summary_text += f"Personnalité principale: {politician_counts.most_common(1)[0][0].title() if politician_counts else 'N/A'}\n\n"

if 'cluster_kmeans_full' in df.columns:
    dominant_cluster = df['cluster_kmeans_full'].value_counts().idxmax()
    dominant_percentage = df['cluster_kmeans_full'].value_counts().max() / len(df) * 100
    summary_text += f"Cluster dominant: {dominant_cluster} ({dominant_percentage:.1f}%)"

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig("data/processed/huffpost_detailed_analysis.png", dpi=150, bbox_inches='tight')
print("✅ Visualisation sauvegardée: data/processed/huffpost_detailed_analysis.png")

# 8. Sauvegarde des résultats
print("\n" + "="*70)
print("💾 SAUVEGARDE DES RÉSULTATS")
print("="*70)

# Ajouter les nouvelles colonnes au DataFrame
df['topic_lda'] = df.get('topic_lda', -1)
df['sentiment_score'] = df.get('sentiment_score', 0)

# Sauvegarde
output_path = "data/processed/news_with_detailed_analysis.csv"
df.to_csv(output_path, index=False)
print(f"✅ Données détaillées sauvegardées: {output_path}")

# Générer un rapport
report_text = f"""ANALYSE DÉTAILLÉE - HUFFINGTON POST
{'='*80}

DATE D'ANALYSE: {datetime.now().strftime('%Y-%m-%d %H:%M')}
ARTICLES ANALYSÉS: {len(df):,}

1. PÉRIODE COUVERTE:
   • Début: {df['date'].min().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A'}
   • Fin: {df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A'}
   • Années: {df['year'].nunique() if 'year' in df.columns else 'N/A'}

2. DISTRIBUTION TEMPORELLE:
"""
if 'year' in df.columns:
    for year, count in yearly_counts.items():
        report_text += f"   • {year}: {count:,} articles\n"

report_text += f"""
3. THÉMATIQUES IDENTIFIÉES (LDA):
   • {n_topics} thèmes principaux
   • Distribution:"""
if 'topic_lda' in df.columns:
    for topic in range(n_topics):
        count = (df['topic_lda'] == topic).sum()
        if count > 0:
            percentage = count / len(df) * 100
            report_text += f"\n     - Thème {topic}: {count:,} articles ({percentage:.1f}%)"

report_text += f"""
   
4. ANALYSE DE SENTIMENT:
   • Score moyen: {sample_df['sentiment_score'].mean():.3f}
   • Articles positifs: {(sample_df['sentiment_score'] > 0.1).sum():,} ({(sample_df['sentiment_score'] > 0.1).sum()/sample_size*100:.1f}%)
   • Articles négatifs: {(sample_df['sentiment_score'] < -0.1).sum():,} ({(sample_df['sentiment_score'] < -0.1).sum()/sample_size*100:.1f}%)
   • Articles neutres: {(abs(sample_df['sentiment_score']) <= 0.1).sum():,} ({(abs(sample_df['sentiment_score']) <= 0.1).sum()/sample_size*100:.1f}%)

5. PERSONNALITÉS LES PLUS MENTIONNÉES:
"""
for politician, count in politician_counts.most_common(5):
    percentage = count / sample_size * 100
    report_text += f"   • {politician.title()}: {count:,} mentions ({percentage:.1f}%)\n"

report_text += f"""
6. PAYS LES PLUS MENTIONNÉS:
"""
for country, count in country_counts.most_common(5):
    percentage = count / sample_size * 100
    report_text += f"   • {country.title()}: {count:,} mentions ({percentage:.1f}%)\n"

report_text += f"""
7. COMPLEXITÉ LINGUISTIQUE:
   • Mots par article (moyenne): {complexity_df['word_count'].mean():.0f}
   • Phrases par article (moyenne): {complexity_df['sentence_count'].mean():.1f}
   • Longueur moyenne des mots: {complexity_df['avg_word_length'].mean():.1f} caractères

8. CLUSTERING:
"""
if 'cluster_kmeans_full' in df.columns:
    cluster_dist = df['cluster_kmeans_full'].value_counts().sort_index()
    for cluster, count in cluster_dist.items():
        percentage = count / len(df) * 100
        report_text += f"   • Cluster {cluster}: {count:,} articles ({percentage:.1f}%)\n"

report_text += f"""
CONCLUSIONS PRINCIPALES:
1. Le Huffington Post couvre une période de {df['year'].nunique() if 'year' in df.columns else 'N/A'} ans
2. {n_topics} thèmes principaux structurent le discours
3. Le ton général est {'positif' if sample_df['sentiment_score'].mean() > 0 else 'négatif' if sample_df['sentiment_score'].mean() < 0 else 'neutre'}
4. Les personnalités politiques sont fréquemment mentionnées
5. La complexité linguistique varie selon les clusters

{'='*80}
FIN DU RAPPORT
"""

with open("data/processed/huffpost_analysis_report.txt", "w", encoding='utf-8') as f:
    f.write(report_text)

print(f"✅ Rapport généré: data/processed/huffpost_analysis_report.txt")

print("\n" + "="*70)
print("🎉 ANALYSE TERMINÉE !")
print("="*70)
print(f"\n📊 RÉCAPITULATIF:")
print(f"   • Articles analysés: {len(df):,}")
print(f"   • Thèmes identifiés: {n_topics}")
print(f"   • Personnalités extraites: {len(politician_counts)}")
print(f"   • Analyse de sentiment: {sample_size} articles")
print(f"   • Visualisations: 1")
print(f"   • Rapports: 1")
print(f"\n📁 FICHIERS GÉNÉRÉS:")
print(f"   • data/processed/huffpost_detailed_analysis.png")
print(f"   • data/processed/news_with_detailed_analysis.csv")
print(f"   • data/processed/huffpost_analysis_report.txt")