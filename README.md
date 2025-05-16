# 🧑‍⚖️ LLM-as-Judge en Python

Este proyecto implementa un evaluador automático de respuestas basado en LLM (Large Language Model), ideal para comparar respuestas A/B o puntuar salidas individuales con criterios como precisión y relevancia.

---

## ⚙️ Requisitos

- Python 3.9+
- Clave API válida de OpenAI (con acceso a GPT-4o o modelos compatibles)
- Las siguientes librerías:
  ```bash
  pip install -r requirements.txt

## 🧩 Ejecución

- Inicialmente debemos hacer la consulta "manualmente" y guardar los resultados en el archivo "data/input/pairs.jsonl" (en futuras versiones se hará automaticamente)
- Devemos añadir la api key de openAi (que utilizamos como juez) en el archivo ".env" situado en raiz (podemos usar .env.example como ejemplo)
- Con todo esto preparado podemos ejecutar nuestra prueba
  ```bash
  cd src
  
  python judge.py
 