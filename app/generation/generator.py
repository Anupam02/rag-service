"""
Step 5: Generation
---------------------
Goal: take the user's question + the top-k retrieved chunks, build a
prompt that grounds the model in that context, and get an answer back
from a local Ollama model (llama3.2 / qwen2.5:7b).

The core idea of "grounding": we explicitly instruct the model to
answer USING ONLY the provided context, and to say so if the answer
isn't in it. This is what makes RAG different from just chatting with
a raw LLM - without this instruction, the model will happily blend
retrieved facts with its own (possibly outdated or hallucinated)
parametric knowledge, and you lose the whole point of grounding
answers in your source documents.

Prompt structure matters a lot here. A common failure mode: dumping
retrieved chunks into the prompt with no separation or instruction,
so the model can't tell "this is reference material" from "this is
the question" from "this is a previous turn of conversation." We use
clear delimiters and an explicit instruction block.
"""

import requests


SYSTEM_INSTRUCTION = """You are a helpful assistant that answers questions using ONLY the context provided below.

Rules:
- If the answer is fully contained in the context, answer clearly and concisely.
- If the context only partially answers the question, answer what you can and say what's missing.
- If the context does not contain the answer at all, say "I don't have enough information in the provided documents to answer that" - do NOT use outside knowledge to fill the gap.
- Cite which source each fact came from when possible, using the source names given.
"""


def build_prompt(query: str, retrieved_chunks: list) -> str:
    """retrieved_chunks: list of SearchResult from the vector store
    (has .text, .metadata, .score)"""
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        source = chunk.metadata.get("source", chunk.metadata.get("section", f"chunk_{i}"))
        context_blocks.append(f"[Source: {source}]\n{chunk.text}")

    context_str = "\n\n---\n\n".join(context_blocks)

    prompt = f"""{SYSTEM_INSTRUCTION}

CONTEXT:
{context_str}

QUESTION:
{query}

ANSWER:"""
    return prompt


class Generator:
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url

    def generate(self, query: str, retrieved_chunks: list) -> str:
        prompt = build_prompt(query, retrieved_chunks)

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
