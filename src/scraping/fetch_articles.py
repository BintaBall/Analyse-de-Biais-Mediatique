import json
import concurrent.futures
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import os

INPUT = "data/intermediate/selected_links.json"
OUTPUT = "data/processed/news.csv"

def scrape_article(url):
    try:
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)
        return text.strip()
    except:
        return None

def main():
    # Créer le dossier de sortie
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    
    # Charger les articles déjà traités (depuis le CSV existant)
    existing_urls = set()
    if os.path.exists(OUTPUT):
        try:
            existing_df = pd.read_csv(OUTPUT)
            if 'url' in existing_df.columns:
                existing_urls = set(existing_df['url'].dropna().astype(str))
            print(f"📊 Articles déjà dans le CSV: {len(existing_urls)}")
        except:
            existing_df = pd.DataFrame()
            print("⚠️ Impossible de lire le CSV existant, création d'un nouveau")
    else:
        existing_df = pd.DataFrame()
    
    # Lire tous les articles
    all_articles = []
    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            all_articles.append(json.loads(line))
    
    # Filtrer les articles non traités
    articles_to_process = []
    for article in all_articles:
        if article["link"] not in existing_urls:
            articles_to_process.append(article)
    
    print(f"📁 Articles à traiter: {len(articles_to_process)}/{len(all_articles)}")
    
    if len(articles_to_process) == 0:
        print("✅ Tous les articles ont déjà été traités !")
        return
    
    # Paramètres
    BATCH_SIZE = 100
    MAX_WORKERS = 10
    all_new_articles = []
    
    # Traitement par lots
    for i in range(0, len(articles_to_process), BATCH_SIZE):
        batch = articles_to_process[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(articles_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"\n🔧 Lot {batch_num}/{total_batches} ({len(batch)} articles)")
        
        # Traiter le lot
        batch_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for article in batch:
                future = executor.submit(scrape_article, article["link"])
                futures.append((future, article))
            
            for future, article in tqdm(futures, desc=f"Lot {batch_num}", leave=False):
                text = future.result()
                if text and len(text) > 600:
                    batch_results.append({
                        "title": article["headline"],
                        "text": text,
                        "category": article["category"],
                        "date": article["date"],
                        "source": "HuffPost",
                        "url": article["link"]  # Important pour éviter les doublons
                    })
        
        # Ajouter au total
        all_new_articles.extend(batch_results)
        
        # Sauvegarde incrémentale
        if batch_results:
            df_batch = pd.DataFrame(batch_results)
            
            # Mode d'écriture
            mode = 'a' if i > 0 or not existing_df.empty else 'w'
            header = i == 0 and existing_df.empty
            
            df_batch.to_csv(OUTPUT, mode=mode, header=header, index=False)
            print(f"💾 {len(batch_results)} nouveaux articles sauvegardés")
        
        # Vérifier la taille actuelle
        if os.path.exists(OUTPUT):
            current_size = pd.read_csv(OUTPUT).shape[0]
            print(f"📈 Total actuel dans le CSV: {current_size} articles")
    
    # Résumé final
    print(f"\n{'='*50}")
    print(f"✅ TRAITEMENT TERMINÉ")
    print(f"📊 Nouveaux articles ajoutés: {len(all_new_articles)}")
    
    # Vérification finale des doublons
    if os.path.exists(OUTPUT):
        final_df = pd.read_csv(OUTPUT)
        
        # Vérifier et supprimer les doublons basés sur l'URL
        initial_count = len(final_df)
        final_df = final_df.drop_duplicates(subset=['url'])
        final_count = len(final_df)
        
        if initial_count != final_count:
            print(f"🧹 Nettoyage: {initial_count - final_count} doublons supprimés")
            final_df.to_csv(OUTPUT, index=False)
        
        print(f"📋 Total final (sans doublons): {final_count} articles")

if __name__ == "__main__":
    main()