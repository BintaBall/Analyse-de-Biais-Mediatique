import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import numpy as np
import os
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL = "distilbert-base-uncased"
INPUT = "data/processed/news_clean.csv"
OUTPUT = "data/processed/news_embeddings.pt"
BATCH_SIZE = 32
MAX_LENGTH = 128
SAVE_EVERY = 100  # Sauvegarde checkpoint tous les 100 batchs
MAX_EMBEDDINGS_IN_RAM = 5000  # Maximum d'embeddings à garder en RAM

print("⏳ Chargement du modèle...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL)
model = DistilBertModel.from_pretrained(MODEL)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"📱 Utilisation de : {device}")

# Charger les données
print("📥 Chargement des données...")
df = pd.read_csv(INPUT)
if "clean_text" in df.columns:
    texts = df["clean_text"].fillna("").astype(str).tolist()
else:
    # Chercher une colonne texte
    text_cols = [col for col in df.columns if df[col].dtype == 'object']
    if text_cols:
        texts = df[text_cols[0]].fillna("").astype(str).tolist()
    else:
        raise ValueError("Aucune colonne texte trouvée")

print(f"📊 {len(texts):,} textes à traiter")
del df
gc.collect()

# Gestion des checkpoints
checkpoint_file = OUTPUT.replace('.pt', '_checkpoint.pkl')
embeddings_file = OUTPUT

def save_checkpoint(current_idx, embeddings_buffer=None):
    """Sauvegarder l'état du traitement"""
    checkpoint_data = {
        'last_idx': current_idx,
        'total_texts': len(texts),
        'batch_size': BATCH_SIZE
    }
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    # Sauvegarder les embeddings intermédiaires si fournis
    if embeddings_buffer is not None and len(embeddings_buffer) > 0:
        temp_embeddings = torch.cat(embeddings_buffer, dim=0)
        temp_path = embeddings_file.replace('.pt', f'_partial_{current_idx}.pt')
        torch.save(temp_embeddings, temp_path)
        print(f"   Embeddings partiels sauvegardés: {temp_path}")
    
    return checkpoint_file

def load_checkpoint():
    """Charger le dernier checkpoint"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Vérifier la compatibilité
            if checkpoint['total_texts'] == len(texts):
                print(f"✅ Checkpoint trouvé - Reprise à l'index {checkpoint['last_idx'] + 1}")
                return checkpoint['last_idx'] + 1
            else:
                print("⚠️ Checkpoint incompatible - Nouveau démarrage")
                return 0
        except Exception as e:
            print(f"⚠️ Erreur chargement checkpoint: {e}")
            return 0
    return 0

# Fonction pour charger les embeddings partiels sauvegardés
def load_partial_embeddings():
    """Charger tous les fichiers d'embeddings partiels"""
    partial_files = []
    for f in os.listdir(os.path.dirname(OUTPUT)):
        if f.startswith(os.path.basename(OUTPUT).replace('.pt', '_partial_')) and f.endswith('.pt'):
            partial_files.append(os.path.join(os.path.dirname(OUTPUT), f))
    
    if not partial_files:
        return []
    
    # Trier par index
    partial_files.sort(key=lambda x: int(x.split('_')[-1].replace('.pt', '')))
    
    embeddings_list = []
    for file_path in partial_files:
        try:
            emb = torch.load(file_path, map_location='cpu')
            embeddings_list.append(emb)
            print(f"   Chargé: {os.path.basename(file_path)} - {emb.shape[0]} embeddings")
        except Exception as e:
            print(f"   Erreur chargement {file_path}: {e}")
    
    return embeddings_list

# Vérifier s'il existe déjà des embeddings finaux
if os.path.exists(embeddings_file):
    response = input(f"⚠️ Le fichier {embeddings_file} existe déjà. Voulez-vous le reprendre (R) ou redémarrer (N) ? [R/N]: ")
    if response.upper() == 'N':
        os.remove(embeddings_file)
        print("🗑️ Fichier existant supprimé")
        start_idx = 0
    else:
        # Essayer de charger les embeddings existants
        try:
            existing_embeddings = torch.load(embeddings_file, map_location='cpu')
            print(f"✅ Embeddings existants chargés: {existing_embeddings.shape}")
            start_idx = existing_embeddings.shape[0]
            
            if start_idx >= len(texts):
                print("🎉 Tous les textes sont déjà traités!")
                exit(0)
                
        except Exception as e:
            print(f"⚠️ Erreur chargement: {e} - Nouveau démarrage")
            start_idx = 0
else:
    # Vérifier les checkpoints
    start_idx = load_checkpoint()
    
    # Charger les embeddings partiels si existants
    partial_embeddings = load_partial_embeddings()
    if partial_embeddings:
        existing_embeddings = torch.cat(partial_embeddings, dim=0)
        start_idx = existing_embeddings.shape[0]
        print(f"📊 {start_idx} embeddings déjà générés depuis fichiers partiels")

# Fonction pour traiter un batch
@torch.no_grad()
def process_batch(batch_texts):
    """Traiter un batch de textes"""
    if not batch_texts:
        return torch.zeros(0, 768)
    
    # Tokenization
    inputs = tokenizer(
        batch_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        return_attention_mask=True
    )
    
    # Déplacer sur device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Extraire embeddings [CLS]
    embeddings = outputs.last_hidden_state[:, 0, :]
    
    return embeddings.cpu()

# Traitement principal
print(f"\n🚀 Démarrage du traitement à partir de l'index {start_idx}")
print(f"📊 Batch size: {BATCH_SIZE}")

# Buffer pour stocker les embeddings temporairement
embeddings_buffer = []
all_embeddings = []

# Si on a déjà des embeddings, les ajouter au buffer
if 'existing_embeddings' in locals():
    embeddings_buffer.append(existing_embeddings)
    print(f"📥 {existing_embeddings.shape[0]} embeddings existants chargés dans le buffer")

# Barre de progression
pbar = tqdm(total=len(texts), initial=start_idx, desc="Génération embeddings")

# Variables pour le monitoring
batch_count = 0
last_save_idx = start_idx

for i in range(start_idx, len(texts), BATCH_SIZE):
    batch_start = i
    batch_end = min(i + BATCH_SIZE, len(texts))
    batch_texts = texts[batch_start:batch_end]
    
    try:
        # Traiter le batch
        batch_embeddings = process_batch(batch_texts)
        
        # Ajouter au buffer
        embeddings_buffer.append(batch_embeddings)
        
        # Gérer la mémoire: sauvegarder si le buffer devient trop grand
        total_in_buffer = sum(emb.shape[0] for emb in embeddings_buffer)
        if total_in_buffer >= MAX_EMBEDDINGS_IN_RAM:
            # Concaténer et sauvegarder
            concat_embeddings = torch.cat(embeddings_buffer, dim=0)
            
            # Ajouter à la liste principale
            all_embeddings.append(concat_embeddings)
            
            # Vider le buffer
            embeddings_buffer = []
            
            # Sauvegarde intermédiaire
            temp_path = embeddings_file.replace('.pt', f'_temp_{batch_end}.pt')
            torch.save(concat_embeddings, temp_path)
            print(f"\n💾 Sauvegarde intermédiaire: {temp_path}")
        
        # Mettre à jour la progression
        processed = batch_end
        pbar.update(len(batch_texts))
        batch_count += 1
        
        # Sauvegarde périodique du checkpoint
        if batch_count % SAVE_EVERY == 0 or batch_end == len(texts):
            save_checkpoint(batch_end - 1, embeddings_buffer)
            
            # Afficher l'état
            pbar.set_postfix({
                'processed': f"{processed:,}/{len(texts):,}",
                'buffer': f"{total_in_buffer:,}"
            })
            
            # Nettoyer la mémoire
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"\n⚠️ Erreur batch {batch_start}: {str(e)[:100]}")
        # Ajouter des embeddings nuls pour continuer
        zero_embeddings = torch.zeros(len(batch_texts), 768)
        embeddings_buffer.append(zero_embeddings)
        pbar.update(len(batch_texts))

pbar.close()

# Finaliser: concaténer tous les embeddings
print("\n📦 Assemblage final des embeddings...")

# Ajouter les derniers embeddings du buffer
if embeddings_buffer:
    final_buffer = torch.cat(embeddings_buffer, dim=0)
    all_embeddings.append(final_buffer)

# Concaténer tout
if all_embeddings:
    final_embeddings = torch.cat(all_embeddings, dim=0)
else:
    # Charger depuis fichiers temporaires si nécessaire
    temp_files = []
    for f in os.listdir(os.path.dirname(OUTPUT)):
        if f.startswith(os.path.basename(OUTPUT).replace('.pt', '_temp_')) and f.endswith('.pt'):
            temp_files.append(os.path.join(os.path.dirname(OUTPUT), f))
    
    if temp_files:
        temp_files.sort()
        temp_embeddings = []
        for temp_file in temp_files:
            emb = torch.load(temp_file, map_location='cpu')
            temp_embeddings.append(emb)
        
        final_embeddings = torch.cat(temp_embeddings, dim=0)
    else:
        final_embeddings = torch.zeros(0, 768)

# Vérifier la taille
print(f"✅ Embeddings générés: {final_embeddings.shape}")

if final_embeddings.shape[0] > 0:
    # Sauvegarde finale
    print(f"💾 Sauvegarde finale: {embeddings_file}")
    torch.save(final_embeddings, embeddings_file)
    
    # Sauvegarde au format numpy aussi
    np.save(embeddings_file.replace('.pt', '.npy'), final_embeddings.numpy())
    
    # Statistiques
    print(f"\n📊 Statistiques:")
    print(f"   Total: {final_embeddings.shape[0]:,} embeddings")
    print(f"   Dimension: {final_embeddings.shape[1]}")
    print(f"   Moyenne: {final_embeddings.mean().item():.6f}")
    print(f"   Std: {final_embeddings.std().item():.6f}")
    
    # Vérification
    loaded = torch.load(embeddings_file, map_location='cpu')
    if loaded.shape == final_embeddings.shape:
        print(f"✅ Vérification OK: {loaded.shape}")
    else:
        print(f"❌ Vérification échouée: {loaded.shape} != {final_embeddings.shape}")

# Nettoyage
print("\n🧹 Nettoyage des fichiers temporaires...")

# Supprimer les fichiers temporaires
for pattern in ['_checkpoint.pkl', '_partial_', '_temp_']:
    for f in os.listdir(os.path.dirname(OUTPUT)):
        if pattern in f:
            try:
                os.remove(os.path.join(os.path.dirname(OUTPUT), f))
                print(f"🗑️ Supprimé: {f}")
            except:
                pass

# Libérer la mémoire
del model, tokenizer, final_embeddings
if 'loaded' in locals():
    del loaded
gc.collect()

if device.type == 'cuda':
    torch.cuda.empty_cache()

print("\n🎉 Traitement terminé avec succès!")
print(f"📁 Fichier .pt généré: {embeddings_file}")