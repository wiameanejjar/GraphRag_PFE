from datasets import load_dataset
import json
import os

os.makedirs("data/hotpotqa", exist_ok=True)

print("Téléchargement HotpotQA en cours...")
dataset = load_dataset("hotpot_qa", "fullwiki", split="train[:2000]")

questions = []
for item in dataset:
    questions.append({
        "id": item["id"],
        "question": item["question"],
        "answer": item["answer"],
        "type": item["type"],
        "supporting_facts": item["supporting_facts"]
    })

with open("data/hotpotqa/hotpotqa_2000.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

print(f"Terminé : {len(questions)} questions téléchargées")
print(f"Exemple de question : {questions[0]['question']}")
print(f"Réponse : {questions[0]['answer']}")