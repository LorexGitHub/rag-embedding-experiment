from sentence_transformers import SentenceTransformer, util
from .config import EVAL_SEMANTIC_MODEL, GENERATOR
from .schemas import EvaluationMetrics
from .generator import get_generator, _TemplateGenerator


class Evaluator:
    def __init__(self):
        self._semantic_model = None
        self._judge = None

    def _get_semantic_model(self):
        if self._semantic_model is None:
            self._semantic_model = SentenceTransformer(EVAL_SEMANTIC_MODEL)
        return self._semantic_model

    def _get_judge(self):
        if self._judge is not None:
            return self._judge
        gen = get_generator()
        if isinstance(gen, _TemplateGenerator):
            self._judge = False
            return None
        self._judge = gen
        return self._judge

    def rouge_l_f1(self, generated: str, reference: str) -> float:
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()
        if not gen_tokens or not ref_tokens:
            return 0.0
        m = len(ref_tokens)
        n = len(gen_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i - 1] == gen_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        precision = lcs / n if n > 0 else 0.0
        recall = lcs / m if m > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def semantic_similarity(self, generated: str, reference: str) -> float:
        model = self._get_semantic_model()
        emb_gen = model.encode(generated, convert_to_tensor=True)
        emb_ref = model.encode(reference, convert_to_tensor=True)
        return float(util.cos_sim(emb_gen, emb_ref)[0][0])

    def llm_quality_score(self, query: str, answer: str, context: list[str]) -> float | None:
        judge = self._get_judge()
        if not judge:
            return None
        try:
            text = judge.generate(
                f"Rate the following answer on a scale of 1-5 (one decimal allowed) "
                f"based on how well it answers the question using only the provided context.\n\n"
                f"Question: {query}\n"
                f"Context: {', '.join(context)}\n"
                f"Answer: {answer}\n\n"
                f"Only output a number between 1 and 5:",
                [],
            )
            return float(text.strip())
        except Exception:
            return None

    def evaluate(
        self,
        query: str,
        generated_answer: str,
        retrieved_context: list[str],
        ground_truth: str,
    ) -> EvaluationMetrics:
        return EvaluationMetrics(
            exact_match=ground_truth.lower() in generated_answer.lower(),
            rouge_l_f1=self.rouge_l_f1(generated_answer, ground_truth),
            semantic_similarity=self.semantic_similarity(generated_answer, ground_truth),
            llm_quality_score=self.llm_quality_score(query, generated_answer, retrieved_context),
        )
