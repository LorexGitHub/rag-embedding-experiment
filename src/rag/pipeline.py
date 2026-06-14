from .retriever import Retriever
from .generator import get_generator
from .evaluator import Evaluator
from .schemas import RAGResult, GenerationResult


class RAGPipeline:
    def __init__(self):
        self.generator = get_generator()
        self.evaluator = Evaluator()

    def run(
        self,
        query: str,
        documents: list[str],
        ground_truth: str,
        dataset_name: str,
        embedding_model: str,
        top_k: int = 5,
        on_stage: callable = None,
    ) -> RAGResult:
        if on_stage:
            on_stage("Loading model...")
        retrieval = Retriever.retrieve(query, documents, embedding_model, top_k)
        if on_stage:
            on_stage("Generating answer...")
        answer = self.generator.generate(query, retrieval.documents)
        if on_stage:
            on_stage("Evaluating...")
        evaluation = self.evaluator.evaluate(
            query, answer, retrieval.documents, ground_truth
        )
        return RAGResult(
            query=query,
            ground_truth=ground_truth,
            dataset_used=dataset_name,
            retrieval=retrieval,
            generation=GenerationResult(
                answer=answer,
                model_name=embedding_model,
                retrieval_model=embedding_model,
                retrieved_documents=retrieval.documents,
            ),
            evaluation=evaluation,
        )
