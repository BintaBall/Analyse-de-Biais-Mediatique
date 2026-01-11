import json
import random

INPUT = "data/raw/News_Category_Dataset_v3.json"
OUTPUT = "data/intermediate/selected_links.json"
TARGET = 50000

CATEGORIES = {"POLITICS", "WORLD NEWS", "U.S. NEWS"}

articles = []

# 1. Lire et filtrer les articles
with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item["category"] in CATEGORIES:
            articles.append(item)

# 2. Vérifier combien d'articles sont disponibles
available_count = len(articles)
print(f"Nombre d'articles disponibles dans les catégories spécifiées : {available_count}")

# 3. Ajuster la taille de l'échantillon si nécessaire
if available_count < TARGET:
    print(f"Avertissement : Seulement {available_count} articles disponibles, échantillon réduit à cette taille.")
    sample_size = available_count
else:
    sample_size = TARGET

# 4. Prendre l'échantillon
sampled = random.sample(articles, sample_size)

# 5. Sauvegarder les résultats
with open(OUTPUT, "w", encoding="utf-8") as f:
    for a in sampled:
        f.write(json.dumps(a) + "\n")

print(f"{len(sampled)} liens sélectionnés")