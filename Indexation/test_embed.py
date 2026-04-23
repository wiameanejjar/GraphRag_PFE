import asyncio
from lightrag.llm.ollama import ollama_embed

async def test():
    result = await ollama_embed(
        ["test texte"],
        embed_model="nomic-embed-text",
        host="http://localhost:11434"
    )
    print(f"Dimension réelle : {len(result[0])}")

asyncio.run(test())