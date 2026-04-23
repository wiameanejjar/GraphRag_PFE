import json
import re
import os

def clean_text(text):
    # Supprimer l'entête arXiv (ex: "arXiv:0704.0047v1 [cs.NE] 1 Apr 2007")
    text = re.sub(r'arXiv:\S+\s*\[[\w\.]+\]\s*\d+\s+\w+\s+\d{4}', '', text)
    
    # Supprimer les numéros de page isolés
    text = re.sub(r'\n\s*\d+\s*\n', ' ', text)
    
    # Supprimer les URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Supprimer les emails
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Supprimer les caractères de contrôle
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Remplacer les sauts de ligne multiples par un espace
    text = re.sub(r'\n+', ' ', text)
    
    # Remplacer les espaces multiples par un seul
    text = re.sub(r'\s+', ' ', text)
    
    # Supprimer les caractères non imprimables
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\/\%\+\=\<\>\@\#\&\*]', ' ', text)
    
    # Nettoyage final
    text = text.strip()
    return text

print("Chargement des articles parsés...")
with open("data/cleaned/arxiv_parsed.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

print(f"Nettoyage amélioré de {len(papers)} articles...")

cleaned_papers = []
too_short = 0

for paper in papers:
    cleaned_title = clean_text(paper["title"])
    cleaned_abstract = clean_text(paper["abstract"])
    cleaned_fulltext = clean_text(paper["full_text"])
    
    # Garder seulement si le texte est assez long (au moins 300 mots)
    if len(cleaned_fulltext.split()) < 300:
        too_short += 1
        continue
    
    cleaned_papers.append({
        "id": paper["id"],
        "title": cleaned_title,
        "abstract": cleaned_abstract,
        "full_text": cleaned_fulltext,
        "pdf_filename": paper["pdf_filename"],
        "categories": paper["categories"],
        "published": paper["published"],
        "word_count": len(cleaned_fulltext.split())
    })

with open("data/cleaned/arxiv_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_papers, f, ensure_ascii=False, indent=2)

print(f"Articles nettoyés  : {len(cleaned_papers)}")
print(f"Trop courts        : {too_short}")
print(f"Sauvegardé dans    : data/cleaned/arxiv_cleaned.json")
print(f"Exemple full_text  : {cleaned_papers[0]['full_text'][:200]}")