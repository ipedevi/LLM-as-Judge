import os, json, openai, functools, pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Define variables
INPUT_FILE = "../data/input/pairs.jsonl"
OUTPUT_FILE = "../data/output/judge_results.csv"
OUTPUT_FILE_JSON = "../data/output/judge_results_lite.jsonl"

# Define the evaluation prompt
PAIRWISE_TEMPLATE = """
You are an impartial evaluator.
Compare answer A and answer B to the given question.
Rate each criterion from 1‑5 and pick a winner.

Return JSON: {"winner":"A/B/tie","scores":{"relevance":...,"accuracy":...},"explanation":""}
"""

# Cargar variables del archivo .env
load_dotenv()

# Obtener clave de entorno
api_key = os.getenv("OPENAI_API_KEY")

# Validación opcional
if not api_key:
    raise ValueError("No se encontró OPENAI_API_KEY en el entorno")

# Crear cliente OpenAI
client = openai.OpenAI(api_key=api_key)

def call_llm(prompt, model="gpt-4o", max_tokens=512):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"}  # Esto ayuda a forzar respuestas en JSON (si el modelo lo soporta)
    )
    return response.choices[0].message.content

def judge_pair(example, template=PAIRWISE_TEMPLATE):
    prompt = template + f"\n\nQUESTION:\n{example['question']}\n\nA:\n{example['answer_a']}\n\nB:\n{example['answer_b']}"
    return json.loads(call_llm(prompt))

def run_eval(dataset_path):
    ds = pd.read_json(dataset_path, lines=True)
    results = []
    for row in tqdm(ds.to_dict(orient="records")):
        out = judge_pair(row)
        results.append({**row, **out})
    return pd.DataFrame(results)

def mostrar_resultado(row):
    print("═" * 60)
    print(f"🏁  Pregunta: {row['question']}")
    print(f"🅰  Respuesta A: {row['answer_a'][:100]}...")
    print(f"🅱  Respuesta B: {row['answer_b'][:100]}...\n")
    print(f"🎯  Ganador: {'🅰' if row['winner'] == 'A' else '🅱' if row['winner'] == 'B' else '🤝 Empate'}")
    print(f"📊  Relevance: {row['scores']['relevance']} | Accuracy: {row['scores']['accuracy']}")
    print(f"💬  Explicación:\n{row['explanation']}")
    print("═" * 60 + "\n")

if __name__ == "__main__":
    df = run_eval(INPUT_FILE)
    df.to_json(OUTPUT_FILE_JSON, orient="records", indent=2, force_ascii=False)
    for _, row in df.iterrows():
        mostrar_resultado(row)