# Hephaestus: Agentic Maintenance Assistant

Hephaestus is a RAG-powered agentic assistant designed to streamline maintenance operations by retrieving past interventions and providing concise, actionable answers.

## 🛠 Tech Stack
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Vector Database:** Qdrant
- **LLM Integration:** OpenAI
- **Observability:** LangSmith (integrated with RAGAS for evaluation)

## 🚀 Quick Start

1. **Prerequisites:** Ensure you have `uv` and `docker` installed.
2. **Environment:** Create a `.env` file in the root with your `OPENAI_API_KEY` and LangSmith credentials.
3. **Run Stack:** Start Qdrant, the API, and the Streamlit UI using:
   ```bash
   make run-docker-compose
   ```
4. **Run Evaluations:** To run the RAGAS evaluation pipeline:
   ```bash
   make run-evals
   ```

## 📂 Project Structure
- `apps/api/`: FastAPI backend with the RAG pipeline.
- `apps/chatbot_ui/`: Streamlit chat interface.
- `notebooks/`: Data exploration, pipeline development, and evaluation experiments.
- `data/`: Source maintenance datasets (gitignored).
- `Makefile`: Convenient shortcuts for common tasks.
