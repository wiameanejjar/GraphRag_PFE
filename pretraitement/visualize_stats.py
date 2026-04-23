import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import Counter
import os

matplotlib.use('Agg')
os.makedirs("data/figures", exist_ok=True)

print("Chargement des données...")
with open("data/chunks/arxiv_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open("data/cleaned/arxiv_cleaned.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

with open("data/hotpotqa/hotpotqa_500.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# ================================================
# Graphique 1 : Distribution des tokens par chunk
# ================================================
tokens_list = [c["token_count"] for c in chunks]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(tokens_list, bins=30, color='#4C72B0', edgecolor='white', linewidth=0.5)
ax.set_title("Distribution du nombre de tokens par chunk", fontsize=14, fontweight='bold')
ax.set_xlabel("Nombre de tokens", fontsize=12)
ax.set_ylabel("Nombre de chunks", fontsize=12)
ax.axvline(np.mean(tokens_list), color='red', linestyle='--', linewidth=1.5, label=f'Moyenne : {np.mean(tokens_list):.0f}')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("data/figures/distribution_tokens_chunks.png", dpi=150, bbox_inches='tight')
plt.close()
print("Graphique 1 sauvegardé : distribution_tokens_chunks.png")

# ================================================
# Graphique 2 : Distribution des thèmes (top 10)
# ================================================
all_cats = []
for p in papers:
    cats = p["categories"]
    if isinstance(cats, list):
        all_cats.extend(cats)
    else:
        all_cats.extend(cats.split())

top_themes = Counter(all_cats).most_common(10)
labels = [t[0] for t in top_themes]
values = [t[1] for t in top_themes]

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2',
          '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor='white')
ax.set_title("Distribution des thèmes arXiv CS.AI (Top 10)", fontsize=14, fontweight='bold')
ax.set_xlabel("Nombre d'articles", fontsize=12)
for bar, val in zip(bars, values[::-1]):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            str(val), va='center', fontsize=10)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("data/figures/distribution_themes.png", dpi=150, bbox_inches='tight')
plt.close()
print("Graphique 2 sauvegardé : distribution_themes.png")

# ================================================
# Graphique 3 : Chunks par document
# ================================================
chunks_per_doc = Counter(c["doc_id"] for c in chunks)
chunks_counts = list(chunks_per_doc.values())

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(chunks_counts, bins=25, color='#55A868', edgecolor='white', linewidth=0.5)
ax.set_title("Distribution du nombre de chunks par document", fontsize=14, fontweight='bold')
ax.set_xlabel("Nombre de chunks", fontsize=12)
ax.set_ylabel("Nombre de documents", fontsize=12)
ax.axvline(np.mean(chunks_counts), color='red', linestyle='--', linewidth=1.5,
           label=f'Moyenne : {np.mean(chunks_counts):.1f}')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("data/figures/chunks_par_document.png", dpi=150, bbox_inches='tight')
plt.close()
print("Graphique 3 sauvegardé : chunks_par_document.png")

# ================================================
# Graphique 4 : Types de questions HotpotQA
# ================================================
q_types = Counter(q["type"] for q in questions)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
axes[0].pie(
    q_types.values(),
    labels=[f"{k}\n({v} questions)" for k, v in q_types.items()],
    colors=['#4C72B0', '#DD8452'],
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 11}
)
axes[0].set_title("Types de questions HotpotQA", fontsize=13, fontweight='bold')

# Bar chart
axes[1].bar(q_types.keys(), q_types.values(), color=['#4C72B0', '#DD8452'], edgecolor='white')
axes[1].set_title("Nombre de questions par type", fontsize=13, fontweight='bold')
axes[1].set_ylabel("Nombre de questions")
for i, (k, v) in enumerate(q_types.items()):
    axes[1].text(i, v + 5, str(v), ha='center', fontsize=11, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("data/figures/hotpotqa_types.png", dpi=150, bbox_inches='tight')
plt.close()
print("Graphique 4 sauvegardé : hotpotqa_types.png")

# ================================================
# Graphique 5 : Résumé général (dashboard)
# ================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Dashboard — Statistiques du Dataset PFE GraphRAG\nWiame Anejjar | Master SDIA",
             fontsize=14, fontweight='bold', y=1.01)

# Panel 1 : métriques clés
axes[0, 0].axis('off')
stats_text = [
    ("Corpus arXiv CS.AI", ""),
    ("Documents", f"{len(papers)}"),
    ("Chunks totaux", f"{len(chunks):,}"),
    ("Tokens moyens/chunk", f"{int(np.mean(tokens_list))}"),
    ("", ""),
    ("Benchmark HotpotQA", ""),
    ("Questions totales", f"{len(questions)}"),
    ("Questions bridge", f"{q_types.get('bridge', 0)}"),
    ("Questions comparison", f"{q_types.get('comparison', 0)}"),
]
y_pos = 0.95
for label, value in stats_text:
    if value == "":
        axes[0, 0].text(0.05, y_pos, label, fontsize=11, fontweight='bold',
                       color='#4C72B0', transform=axes[0, 0].transAxes)
    else:
        axes[0, 0].text(0.05, y_pos, f"  {label}:", fontsize=10,
                       transform=axes[0, 0].transAxes, color='#333333')
        axes[0, 0].text(0.75, y_pos, value, fontsize=10, fontweight='bold',
                       transform=axes[0, 0].transAxes, color='#333333')
    y_pos -= 0.1
axes[0, 0].set_title("Métriques clés", fontsize=12, fontweight='bold')

# Panel 2 : distribution tokens
axes[0, 1].hist(tokens_list, bins=20, color='#4C72B0', edgecolor='white')
axes[0, 1].set_title("Tokens par chunk", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel("Tokens")
axes[0, 1].set_ylabel("Chunks")
axes[0, 1].grid(axis='y', alpha=0.3)

# Panel 3 : thèmes
top5 = Counter(all_cats).most_common(5)
axes[1, 0].barh([t[0] for t in top5][::-1], [t[1] for t in top5][::-1],
                color='#55A868', edgecolor='white')
axes[1, 0].set_title("Top 5 thèmes", fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

# Panel 4 : types HotpotQA
axes[1, 1].pie(
    q_types.values(),
    labels=list(q_types.keys()),
    colors=['#4C72B0', '#DD8452'],
    autopct='%1.1f%%',
    startangle=90
)
axes[1, 1].set_title("Types HotpotQA", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("data/figures/dashboard_complet.png", dpi=150, bbox_inches='tight')
plt.close()
print("Graphique 5 sauvegardé : dashboard_complet.png")

print("\nTous les graphiques sont dans : data/figures/")
print("Fichiers PNG créés :")
for f in os.listdir("data/figures"):
    print(f"  - data/figures/{f}")