import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import warnings
from collections import Counter
import time
warnings.filterwarnings('ignore')

# Paramètres
EMBEDDINGS_PATH = "data/processed/news_embeddings.pt"
DATA_PATH = "data/processed/news_clean.csv"
OUTPUT_PATH = "data/processed/news_with_discourse.csv"
RANDOM_STATE = 42
SAMPLE_SIZE = 15000  # Travailler sur un échantillon pour les méthodes lentes

print("="*60)
print("🚀 ANALYSE DES DISCOURS - VERSION OPTIMISÉE")
print("="*60)

print("\n📥 Chargement des données...")
start_time = time.time()

# 1. Charger les données
embeddings = torch.load(EMBEDDINGS_PATH, map_location='cpu')
embeddings_np = embeddings.numpy()
df = pd.read_csv(DATA_PATH)

print(f"✅ Embeddings: {embeddings_np.shape}")
print(f"✅ Données: {df.shape}")
print(f"⏱️  Temps chargement: {time.time() - start_time:.1f}s")

# 2. ÉCHANTILLONNAGE INTELLIGENT pour méthodes lentes
print(f"\n🎯 Création échantillon ({SAMPLE_SIZE} articles)...")
if len(df) > SAMPLE_SIZE:
    # Échantillon stratifié si possible (par source ou date)
    sample_idx = np.random.choice(len(df), SAMPLE_SIZE, replace=False)
    embeddings_sample = embeddings_np[sample_idx]
    df_sample = df.iloc[sample_idx].copy()
else:
    embeddings_sample = embeddings_np
    df_sample = df.copy()

print(f"   Échantillon: {embeddings_sample.shape}")

# 3. MÉTHODE RAPIDE 1: MiniBatch K-means sur embeddings
print("\n" + "="*60)
print("1️⃣  MÉTHODE RAPIDE: MiniBatch K-means")
print("="*60)

start_time = time.time()

# Normalisation rapide
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings_sample)

# MiniBatch K-means (beaucoup plus rapide)
mbkmeans = MiniBatchKMeans(
    n_clusters=5,  # Commencer avec 5 clusters
    random_state=RANDOM_STATE,
    batch_size=1000,
    n_init=3,
    max_iter=100
)

mbkmeans_labels = mbkmeans.fit_predict(embeddings_scaled)
df_sample['cluster_mbkmeans'] = mbkmeans_labels

# Calcul rapide silhouette (sur sous-échantillon)
sil_score = silhouette_score(embeddings_scaled[:2000], mbkmeans_labels[:2000])

print(f"✅ MiniBatch K-means terminé")
print(f"   ⏱️  Temps: {time.time() - start_time:.1f}s")
print(f"   🎯 Silhouette (échantillon): {sil_score:.3f}")
print(f"   📊 Distribution: {dict(Counter(mbkmeans_labels))}")

# 4. MÉTHODE RAPIDE 2: Clustering hiérarchique agglomératif
print("\n" + "="*60)
print("2️⃣  MÉTHODE RAPIDE: Clustering Hiérarchique")
print("="*60)

start_time = time.time()

# Réduction de dimension pour clustering hiérarchique
pca_fast = PCA(n_components=50)
embeddings_pca = pca_fast.fit_transform(embeddings_scaled)

# Clustering hiérarchique avec linkage 'ward' (efficace)
agglo = AgglomerativeClustering(
    n_clusters=4,
    linkage='ward',
    metric='euclidean'
)

agglo_labels = agglo.fit_predict(embeddings_pca)
df_sample['cluster_agglo'] = agglo_labels

print(f"✅ Clustering hiérarchique terminé")
print(f"   ⏱️  Temps: {time.time() - start_time:.1f}s")
print(f"   📊 Distribution: {dict(Counter(agglo_labels))}")

# 5. MÉTHODE RAPIDE 3: Topic Modeling avec NMF
print("\n" + "="*60)
print("3️⃣  MÉTHODE RAPIDE: Topic Modeling (NMF)")
print("="*60)

start_time = time.time()

print("   Extraction des mots-clés...")
# Vectorizer rapide avec peu de features
vectorizer = CountVectorizer(
    max_features=1000,
    min_df=10,
    max_df=0.7,
    stop_words='english'
)

X_counts = vectorizer.fit_transform(df_sample['clean_text'].fillna(''))

# NMF rapide
from sklearn.decomposition import NMF
nmf = NMF(
    n_components=6,  # 6 topics
    random_state=RANDOM_STATE,
    max_iter=50,  # Moins d'itérations
    alpha_W=0.1
)

W = nmf.fit_transform(X_counts)
topic_labels = W.argmax(axis=1)
df_sample['topic_nmf'] = topic_labels

print(f"✅ Topic Modeling terminé")
print(f"   ⏱️  Temps: {time.time() - start_time:.1f}s")
print(f"   📊 Topics: {dict(Counter(topic_labels))}")

# Afficher les mots-clés par topic
print("\n   📝 Top mots par topic:")
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(nmf.components_):
    top_words_idx = topic.argsort()[-8:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"     Topic {topic_idx}: {', '.join(top_words[:5])}")

# 6. ANALYSE DE DENSITÉ (méthode rapide)
print("\n" + "="*60)
print("4️⃣  ANALYSE: Détection des Zones Denses")
print("="*60)

start_time = time.time()

# Utiliser Nearest Neighbors pour détecter la densité
print("   Calcul des densités locales...")
nn = NearestNeighbors(n_neighbors=50, metric='euclidean')
nn.fit(embeddings_pca)

# Distance aux k plus proches voisins
distances, _ = nn.kneighbors(embeddings_pca)
avg_distances = distances.mean(axis=1)

# Identifier les points denses (faible distance moyenne)
dense_threshold = np.percentile(avg_distances, 30)  # 30% les plus denses
is_dense = avg_distances < dense_threshold

df_sample['is_dense_region'] = is_dense

print(f"✅ Analyse densité terminée")
print(f"   ⏱️  Temps: {time.time() - start_time:.1f}s")
print(f"   📊 Zones denses: {is_dense.sum()} points ({(is_dense.sum()/len(df_sample)*100):.1f}%)")

# 7. ANALYSE DES EXTRÊMES (méthode rapide)
print("\n" + "="*60)
print("5️⃣  ANALYSE: Détection des Discours Extrêmes")
print("="*60)

start_time = time.time()

# Méthode simple: distance au centre
center = embeddings_scaled.mean(axis=0)
distances_to_center = np.linalg.norm(embeddings_scaled - center, axis=1)

# Identifier les extrêmes (loin du centre)
extreme_threshold = np.percentile(distances_to_center, 90)  # Top 10% les plus éloignés
is_extreme = distances_to_center > extreme_threshold

df_sample['is_extreme'] = is_extreme
df_sample['distance_to_center'] = distances_to_center

print(f"✅ Détection extrêmes terminée")
print(f"   ⏱️  Temps: {time.time() - start_time:.1f}s")
print(f"   📊 Discours extrêmes: {is_extreme.sum()} ({(is_extreme.sum()/len(df_sample)*100):.1f}%)")

# 8. VISUALISATIONS RAPIDES
print("\n" + "="*60)
print("🎨 VISUALISATIONS")
print("="*60)

print("   Génération des visualisations...")
start_time = time.time()

# PCA pour visualisation
pca_vis = PCA(n_components=2)
embeddings_2d = pca_vis.fit_transform(embeddings_scaled)

fig = plt.figure(figsize=(18, 12))

# Subplot 1: MiniBatch K-means
ax1 = plt.subplot(2, 3, 1)
scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                      c=df_sample['cluster_mbkmeans'], cmap='tab10',
                      alpha=0.5, s=10)
ax1.set_title(f'MiniBatch K-means (5 clusters)\nSilhouette: {sil_score:.3f}')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
plt.colorbar(scatter1, ax=ax1)

# Subplot 2: Clustering hiérarchique
ax2 = plt.subplot(2, 3, 2)
scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                      c=df_sample['cluster_agglo'], cmap='Set2',
                      alpha=0.5, s=10)
ax2.set_title('Clustering Hiérarchique\n(4 clusters)')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
plt.colorbar(scatter2, ax=ax2)

# Subplot 3: Topics NMF
ax3 = plt.subplot(2, 3, 3)
scatter3 = ax3.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                      c=df_sample['topic_nmf'], cmap='tab20',
                      alpha=0.5, s=10)
ax3.set_title(f'Topic Modeling (NMF)\n{df_sample["topic_nmf"].nunique()} topics')
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
plt.colorbar(scatter3, ax=ax3)

# Subplot 4: Zones denses
ax4 = plt.subplot(2, 3, 4)
colors = ['blue' if not dense else 'red' for dense in df_sample['is_dense_region']]
ax4.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
           c=colors, alpha=0.3, s=5)
ax4.set_title(f'Zones Denses (rouge)\n{df_sample["is_dense_region"].sum()} points')
ax4.set_xlabel('PC1')
ax4.set_ylabel('PC2')

# Subplot 5: Discours extrêmes
ax5 = plt.subplot(2, 3, 5)
colors = ['blue' if not extreme else 'red' for extreme in df_sample['is_extreme']]
ax5.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
           c=colors, alpha=0.3, s=5)
ax5.set_title(f'Discours Extrêmes (rouge)\n{df_sample["is_extreme"].sum()} points')
ax5.set_xlabel('PC1')
ax5.set_ylabel('PC2')

# Subplot 6: Résumé statistique
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = f"RÉSUMÉ DE L'ANALYSE\n\n"
summary_text += f"Échantillon: {len(df_sample):,} articles\n"
summary_text += f"Period: "
if 'date' in df_sample.columns:
    summary_text += f"{df_sample['date'].min()[:7]} à {df_sample['date'].max()[:7]}\n\n"
else:
    summary_text += "N/A\n\n"

summary_text += f"MiniBatch K-means:\n"
summary_text += f"  • 5 clusters\n"
summary_text += f"  • Silhouette: {sil_score:.3f}\n\n"

summary_text += f"Clustering Hiérarchique:\n"
summary_text += f"  • 4 clusters\n\n"

summary_text += f"Topic Modeling:\n"
summary_text += f"  • {df_sample['topic_nmf'].nunique()} topics\n\n"

summary_text += f"Zones Denses:\n"
summary_text += f"  • {df_sample['is_dense_region'].sum():,} points\n"
summary_text += f"  • {(df_sample['is_dense_region'].sum()/len(df_sample)*100):.1f}%\n\n"

summary_text += f"Discours Extrêmes:\n"
summary_text += f"  • {df_sample['is_extreme'].sum():,} points\n"
summary_text += f"  • {(df_sample['is_extreme'].sum()/len(df_sample)*100):.1f}%"

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig("data/processed/fast_analysis_results.png", dpi=120, bbox_inches='tight')
print(f"   ✅ Visualisation sauvegardée: data/processed/fast_analysis_results.png")
print(f"   ⏱️  Temps visualisation: {time.time() - start_time:.1f}s")

# 9. PROPAGATION AU DATASET COMPLET (méthode intelligente)
print("\n" + "="*60)
print("🌍 PROPAGATION AU DATASET COMPLET")
print("="*60)

print("   Propagation des clusters à tous les articles...")

# Pour K-means: entraîner sur échantillon, prédire sur tout
if len(df) > SAMPLE_SIZE:
    # Entraîner K-means sur l'échantillon
    kmeans_full = KMeans(
        n_clusters=5,
        random_state=RANDOM_STATE,
        n_init=3
    )
    kmeans_full.fit(embeddings_scaled)
    
    # Prédire sur tout le dataset
    print("   Prédiction sur les 35k articles...")
    all_embeddings_scaled = scaler.transform(embeddings_np)
    all_clusters = kmeans_full.predict(all_embeddings_scaled)
    df['cluster_kmeans_full'] = all_clusters
    
    print(f"   ✅ Clusters assignés à tous les articles")
    print(f"   📊 Distribution: {dict(Counter(all_clusters))}")

# 10. SAUVEGARDE DES RÉSULTATS
print("\n" + "="*60)
print("💾 SAUVEGARDE")
print("="*60)

# Sauvegarder l'analyse détaillée de l'échantillon
df_sample.to_csv(OUTPUT_PATH.replace('.csv', '_sample_detailed.csv'), index=False)
print(f"✅ Analyse échantillon: {OUTPUT_PATH.replace('.csv', '_sample_detailed.csv')}")

# Sauvegarder les clusters pour tout le dataset
if 'cluster_kmeans_full' in df.columns:
    df[['title', 'clean_text', 'date', 'source', 'cluster_kmeans_full']].to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Clusters dataset complet: {OUTPUT_PATH}")

# Générer un rapport rapide
report_text = f"""ANALYSE DES DISCOURS MÉDIATIQUES - RAPPORT RAPIDE
{'='*60}

DONNÉES ANALYSÉES:
• Total articles: {len(df):,}
• Échantillon analysé: {len(df_sample):,}
• Période: {df['date'].min()[:10] if 'date' in df.columns else 'N/A'} à {df['date'].max()[:10] if 'date' in df.columns else 'N/A'}

RÉSULTATS CLÉS:

1. STRUCTURE DES DISCOURS:
• MiniBatch K-means révèle {df_sample['cluster_mbkmeans'].nunique()} clusters
• Score silhouette: {sil_score:.3f} (structure faible mais discernable)
• Distribution: {dict(Counter(df_sample['cluster_mbkmeans']))}

2. TOPICS IDENTIFIÉS (NMF):
• {df_sample['topic_nmf'].nunique()} topics principaux
• Distribution: {dict(Counter(df_sample['topic_nmf']))}

3. ZONES DE CONCENTRATION:
• {(df_sample['is_dense_region'].sum()/len(df_sample)*100):.1f}% des discours dans zones denses
• Indique des thèmes récurrents/prédominants

4. DISCOURS EXTRÊMES:
• {(df_sample['is_extreme'].sum()/len(df_sample)*100):.1f}% des discours identifiés comme 'extrêmes'
• Ces articles sont sémantiquement éloignés du discours médian

INTERPRÉTATION:
Les discours médiatiques forment un continuum avec quelques pôles de concentration.
L'absence de clusters nets suggère une relative homogénéité des discours
ou la nécessité d'analyses plus fines (par source, par période).

NEXT STEPS:
1. Analyse comparative par source médiatique
2. Évolution temporelle des topics
3. Analyse sentiment par cluster
"""

with open("data/processed/quick_analysis_report.txt", "w") as f:
    f.write(report_text)

print(f"✅ Rapport généré: data/processed/quick_analysis_report.txt")

print("\n" + "="*60)
print("🎉 ANALYSE TERMINÉE!")
print("="*60)
print(f"\n📊 RÉCAPITULATIF:")
print(f"   • Articles analysés: {len(df):,}")
print(f"   • Méthodes testées: 5")
print(f"   • Clusters identifiés: {df_sample['cluster_mbkmeans'].nunique()}")
print(f"   • Topics détectés: {df_sample['topic_nmf'].nunique()}")
print(f"   • Visualisations: 1")
print(f"   • Rapports: 2")
print(f"\n📁 FICHIERS GÉNÉRÉS:")
print(f"   • data/processed/fast_analysis_results.png")
print(f"   • data/processed/news_with_discourse.csv")
print(f"   • data/processed/quick_analysis_report.txt")