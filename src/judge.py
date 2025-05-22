import os, json, openai, functools, pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai

# Define variables
QUESTIONS_FILE = "../data/input/questions.jsonl"
ANSWERS_FILE = "../data/input/pairs.jsonl"
OUTPUT_FILE_JSON = "../data/output/judge_results.jsonl"

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
gpt_api_key = os.getenv("OPENAI_API_KEY")
gem_api_key = os.getenv("GEMINI_API_KEY")

# Validación opcional
if not gpt_api_key:
    raise ValueError("No se encontró OPENAI_API_KEY en el entorno")

# Crear cliente LLMs
client = openai.OpenAI(api_key=gpt_api_key)
genai.configure(api_key=gem_api_key)

def call_gpt_llm(prompt, model="gpt-4o", max_tokens=512, format="json"):
    if format == "json":
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}  # Esto ayuda a forzar respuestas en JSON (si el modelo lo soporta)
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens
        )
    return response.choices[0].message.content

def call_gemini_llm(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error al llamar a Gemini: {e}"

def judge_pair(example, template=PAIRWISE_TEMPLATE):
    prompt = template + f"\n\nQUESTION:\n{example['question']}\n\nA:\n{example['answer_a']}\n\nB:\n{example['answer_b']}"
    return json.loads(call_gpt_llm(prompt))

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

# Simulación de llamadas a LLMs
def call_llm(model_name, question):
    model_name = model_name.lower()
    if model_name == "chatgpt":
        return call_gpt_llm(question,format="text")
    elif model_name == "gemini":
        return call_gemini_llm(question)
    else:
        return f"Respuesta simulada de {model_name} a: {question}"

# Procesamiento del archivo jsonl
def procesar_jsonl(archivo_entrada, archivo_salida):
    with open(archivo_entrada, 'r', encoding='utf-8') as infile, \
         open(archivo_salida, 'w', encoding='utf-8') as outfile:
        for linea in infile:
            entrada = json.loads(linea)
            pregunta = entrada["question"]
            modelo1 = entrada["user1"]
            modelo2 = entrada["user2"]

            respuesta_a = call_llm(modelo1, pregunta)
            respuesta_b = call_llm(modelo2, pregunta)

            salida = {
                "question": pregunta,
                "answer_a": respuesta_a,
                "answer_b": respuesta_b
            }
            outfile.write(json.dumps(salida, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # Create JSON file based on questions
    procesar_jsonl(QUESTIONS_FILE, ANSWERS_FILE)

    # judge questions
    df = run_eval(ANSWERS_FILE)
    df.to_json(OUTPUT_FILE_JSON, orient="records", indent=2, force_ascii=False)
    for _, row in df.iterrows():
        mostrar_resultado(row)