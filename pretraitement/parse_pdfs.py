import json
import os
import fitz
from tqdm import tqdm

os.makedirs("data/cleaned", exist_ok=True)

print("Chargement des métadonnées...")
with open("data/arxiv_papers/arxiv_csai_500.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

print(f"Parsing de {len(papers)} PDFs...")

parsed_papers = []
errors = 0

for paper in tqdm(papers, desc="Parsing PDFs"):
    filename = paper.get("pdf_filename", "")
    pdf_path = f"data/pdfs/{filename}"

    if not filename or not os.path.exists(pdf_path):
        errors += 1
        continue

    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        full_text = full_text.strip()

        if len(full_text) < 200:
            errors += 1
            continue

        parsed_papers.append({
            "id": paper["id"],
            "title": paper["title"],
            "abstract": paper["abstract"],
            "full_text": full_text,
            "pdf_filename": filename,
            "categories": paper["categories"],
            "published": paper.get("update_date", "")
        })

    except Exception as e:
        errors += 1
        continue

with open("data/cleaned/arxiv_parsed.json", "w", encoding="utf-8") as f:
    json.dump(parsed_papers, f, ensure_ascii=False, indent=2)

print(f"\nRésultat :")
print(f"  PDFs parsés avec succès : {len(parsed_papers)}")
print(f"  Erreurs/ignorés         : {errors}")
print(f"  Sauvegardé dans         : data/cleaned/arxiv_parsed.json")