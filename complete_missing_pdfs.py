import json
import os
import arxiv
import time

with open("data/arxiv_papers/arxiv_csai_500.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

missing = []
for i, paper in enumerate(papers):
    filename = paper.get("pdf_filename", "")
    filepath = f"data/pdfs/{filename}"
    if not filename or not os.path.exists(filepath):
        missing.append((i, paper))

print(f"PDFs manquants : {len(missing)}")

client = arxiv.Client(page_size=1, delay_seconds=15, num_retries=10)

for i, paper_meta in missing:
    arxiv_id = paper_meta["id"]
    safe_id = arxiv_id.replace("/", "_").replace(".", "_")
    filename = f"paper_{i:04d}_{safe_id}.pdf"
    filepath = f"data/pdfs/{filename}"

    try:
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search))
        if results:
            results[0].download_pdf(dirpath="data/pdfs", filename=filename)
            papers[i]["pdf_filename"] = filename
            print(f"  OK : {paper_meta['title'][:50]}")
        else:
            print(f"  Introuvable : {arxiv_id}")
        time.sleep(15)
    except Exception as e:
        print(f"  ERREUR {arxiv_id} : {e}")
        time.sleep(20)

with open("data/arxiv_papers/arxiv_csai_500.json", "w", encoding="utf-8") as f:
    json.dump(papers, f, ensure_ascii=False, indent=2)

print("Terminé !")