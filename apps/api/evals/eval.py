import asyncio
import dotenv
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample

from core.config import config
from agents.retrieval_generation import rag_pipeline

dotenv.load_dotenv()

qdrant_client = QdrantClient(url=config.QDRANT_URL)
ls_client = Client()

ragas_llm = LangchainLLMWrapper(
    ChatOpenAI(model=config.EVALUATION_MODEL, temperature=0)
)
ragas_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=config.EMBEDDING_MODEL))


# --- RAGAS Metrics ---


def _extract(run, example) -> tuple[dict, dict, dict]:
    return (example.inputs or {}), (run.outputs or {}), (example.outputs or {})


def ragas_context_precision_id(run, example) -> float:
    _, outputs, ref = _extract(run, example)
    sample = SingleTurnSample(
        retrieved_context_ids=outputs.get("retrieved_context_ids", []),
        reference_context_ids=ref.get("chunks_id", []),
    )
    return asyncio.run(IDBasedContextPrecision().single_turn_ascore(sample))


def ragas_context_recall_id(run, example) -> float:
    _, outputs, ref = _extract(run, example)
    sample = SingleTurnSample(
        retrieved_context_ids=outputs.get("retrieved_context_ids", []),
        reference_context_ids=ref.get("chunks_id", []),
    )
    return asyncio.run(IDBasedContextRecall().single_turn_ascore(sample))


def ragas_faithfulness(run, example) -> float:
    inputs, outputs, _ = _extract(run, example)
    sample = SingleTurnSample(
        user_input=inputs.get("question", ""),
        response=outputs.get("answer", ""),
        retrieved_contexts=[inputs.get("context_text", "")],
    )
    return asyncio.run(Faithfulness(llm=ragas_llm).single_turn_ascore(sample))


def ragas_answer_relevancy(run, example) -> float:
    inputs, outputs, _ = _extract(run, example)
    sample = SingleTurnSample(
        user_input=inputs.get("question", ""),
        response=outputs.get("answer", ""),
        retrieved_contexts=[inputs.get("context_text", "")],
    )
    return asyncio.run(
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb).single_turn_ascore(sample)
    )


# --- Run + Score Pipeline ---


def run_and_score_pipeline_on_dataset():
    return ls_client.evaluate(
        lambda x: rag_pipeline(
            client=qdrant_client,
            collection_name=config.QDRANT_COLLECTION,
            query=x["question"],
            embedding_model=config.EMBEDDING_MODEL,
            generation_model=config.GENERATION_MODEL,
        ),
        evaluators=[
            ragas_context_precision_id,
            ragas_context_recall_id,
            ragas_faithfulness,
            ragas_answer_relevancy,
        ],
        data=config.DATASET_NAME,
        experiment_prefix="retriever",
    )


# --- Main ---


def main() -> None:
    run_and_score_pipeline_on_dataset()


if __name__ == "__main__":
    main()
