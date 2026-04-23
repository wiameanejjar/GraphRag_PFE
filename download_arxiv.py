import json
import os
import arxiv
import time

# ================================================
# ÉTAPE 1 : Lire les métadonnées Kaggle
# et filtrer 500 articles cs.AI
# ================================================

os.makedirs("data/pdfs", exist_ok=True)
os.makedirs("data/arxiv_papers", exist_ok=True)

json_path = "data/kaggle_download/arxiv-metadata-oai-snapshot.json"

print("Lecture des métadonnées arXiv...")
print("(Fichier volumineux, patiente 1-2 minutes...)")

papers_csai = []

with open(json_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            paper = json.loads(line.strip())
            categories = paper.get("categories", "")
            if "cs.AI" in categories:
                papers_csai.append({
                    "id": paper.get("id", ""),
                    "title": paper.get("title", "").replace("\n", " ").strip(),
                    "abstract": paper.get("abstract", "").replace("\n", " ").strip(),
                    "authors": paper.get("authors", ""),
                    "categories": categories,
                    "update_date": paper.get("update_date", ""),
                    "pdf_filename": ""
                })
            if len(papers_csai) >= 500:
                break
        except:
            continue

print(f"Articles cs.AI trouvés : {len(papers_csai)}")

with open("data/arxiv_papers/arxiv_csai_500.json", "w", encoding="utf-8") as f:
    json.dump(papers_csai, f, ensure_ascii=False, indent=2)

print("Métadonnées sauvegardées.")
print(f"Exemple : {papers_csai[0]['title']}")

# ================================================
# ÉTAPE 2 : Télécharger les PDFs un par un
# ================================================

print("\nDémarrage téléchargement PDFs...")
print("Durée estimée : 2-3 heures")
print("Si ça s'arrête, relance le script — il reprend automatiquement\n")

client = arxiv.Client(
    page_size=1,
    delay_seconds=15,
    num_retries=8
)

downloaded = 0
skipped = 0
errors = 0

for i, paper_meta in enumerate(papers_csai):
    arxiv_id = paper_meta["id"]
    safe_id = arxiv_id.replace("/", "_").replace(".", "_")
    filename = f"paper_{i:04d}_{safe_id}.pdf"
    filepath = f"data/pdfs/{filename}"

    # Si déjà téléchargé on saute
    if os.path.exists(filepath):
        skipped += 1
        papers_csai[i]["pdf_filename"] = filename
        continue

    try:
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search))

        if not results:
            print(f"  [{i+1}/500] Introuvable : {arxiv_id}")
            errors += 1
            continue

        result = results[0]
        result.download_pdf(dirpath="data/pdfs", filename=filename)
        papers_csai[i]["pdf_filename"] = filename
        downloaded += 1

        print(f"  [{i+1}/500] OK : {paper_meta['title'][:55]}...")

        time.sleep(12)

        # Pause longue tous les 25 téléchargements
        if downloaded % 25 == 0:
            print(f"\n  Pause 90 secondes (anti-blocage arXiv)...")
            # Sauvegarde intermédiaire
            with open("data/arxiv_papers/arxiv_csai_500.json", "w", encoding="utf-8") as f:
                json.dump(papers_csai, f, ensure_ascii=False, indent=2)
            print(f"  Sauvegarde : {downloaded} PDFs téléchargés\n")
            time.sleep(90)

    except Exception as e:
        errors += 1
        print(f"  ERREUR [{i+1}/500] {arxiv_id} : {str(e)[:60]}")
        time.sleep(20)
        continue

# Sauvegarde finale
with open("data/arxiv_papers/arxiv_csai_500.json", "w", encoding="utf-8") as f:
    json.dump(papers_csai, f, ensure_ascii=False, indent=2)

print(f"\n{'='*50}")
print(f"TERMINÉ !")
print(f"  PDFs téléchargés : {downloaded}")
print(f"  Déjà existants   : {skipped}")
print(f"  Erreurs          : {errors}")
print(f"  Dossier PDFs     : data/pdfs/")
print(f"Exemple : {papers[0]['title']}")