# src/analysis/polarization_analysis.py

def measure_polarization(df, embeddings):
    """Mesurer le degré de polarisation des discours"""
    
    # 1. Distance entre clusters
    from sklearn.metrics.pairwise import cosine_similarity
    
    cluster_centers = []
    for cluster in sorted(df['cluster_kmeans_full'].unique()):
        cluster_emb = embeddings[df['cluster_kmeans_full'] == cluster]
        if len(cluster_emb) > 0:
            cluster_centers.append(cluster_emb.mean(axis=0))
    
    # Matrice de similarité entre clusters
    if len(cluster_centers) > 1:
        similarity_matrix = cosine_similarity(cluster_centers)
        polarization_score = 1 - similarity_matrix.mean()
        
        print(f"📏 SCORE DE POLARISATION: {polarization_score:.3f}")
        print("   (0 = discours uniforme, 1 = très polarisé)")
    
    # 2. Polarisation par source
    if 'source' in df.columns:
        source_polarization = {}
        for source in df['source'].value_counts().head(5).index:
            source_emb = embeddings[df['source'] == source]
            if len(source_emb) > 10:
                # Diversité interne
                source_similarities = cosine_similarity(source_emb)
                source_polarization[source] = 1 - source_similarities.mean()
        
        print("\n📰 POLARISATION PAR SOURCE:")
        for source, score in sorted(source_polarization.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {source}: {score:.3f}")