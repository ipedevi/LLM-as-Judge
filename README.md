# 🧑‍⚖️ LLM-as-Judge en Python

Este proyecto implementa un evaluador automático de respuestas basado en LLM (Large Language Model), ideal para comparar respuestas A/B o puntuar salidas individuales con criterios como precisión y relevancia.

---

## ⚙️ Requisitos

- Python 3.9+
- Clave API válida de OpenAI y de Gemini
- Las siguientes librerías:
  ```bash
  pip install -r requirements.txt

## 🧩 Ejecución

- Debemos completar el archivo questions.jsonl con las preguntas y los LLMs a evaluar (actualmente solo Gemini vs ChatGPT)
- Devemos añadir la api key de openAi y de Gemini en el archivo ".env" situado en raiz (podemos usar .env.example como ejemplo)
- Con todo esto preparado podemos ejecutar nuestra prueba
  ```bash
  cd src
  
  python judge.py
- El resultado de esta prueba se mostrará en consola y se guardará en judge_results.jsonl 