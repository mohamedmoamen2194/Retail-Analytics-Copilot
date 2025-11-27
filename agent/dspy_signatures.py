import json
from pathlib import Path
from typing import List
from types import SimpleNamespace

import dspy

MODEL_NAME = "ollama/phi3.5:3.8b-mini-instruct-q4_K_M"
dspy.configure(lm=dspy.LM(MODEL_NAME))


class RouterSignature(dspy.Signature):
    question = dspy.InputField()
    route = dspy.OutputField(desc="One of: rag | sql | hybrid")


class PlannerSignature(dspy.Signature):
    question = dspy.InputField()
    retrieved_chunks = dspy.InputField()

    constraints = dspy.OutputField(
        desc="Structured dict including date_range, categories, kpi, fields, conditions"
    )


class NL2SQLSignature(dspy.Signature):
    question = dspy.InputField()
    constraints = dspy.InputField()
    tables = dspy.InputField()
    table_schemas = dspy.InputField()

    sql = dspy.OutputField(desc="Valid SQLite SQL query as a string.")


class SynthesizerSignature(dspy.Signature):
    question = dspy.InputField()
    format_hint = dspy.InputField()
    sql = dspy.InputField()
    sql_rows = dspy.InputField()
    rag_chunks = dspy.InputField()
    constraints = dspy.InputField()

    final_answer = dspy.OutputField(desc="Value matching format_hint")
    explanation = dspy.OutputField(desc="1-2 sentence explanation")
    citations = dspy.OutputField(desc="List of DB tables + doc chunk IDs")


ROUTER_DATASET = [
    {"question": "What is the return window for unopened Beverages per policy?", "route": "rag"},
    {"question": "List the top 3 products by revenue", "route": "sql"},
    {"question": "During Summer Beverages 1997 which category sold the most units?", "route": "hybrid"},
    {"question": "Explain the average order value definition", "route": "rag"},
    {"question": "Total revenue from Beverages in June 1997", "route": "hybrid"},
    {"question": "Best customer by gross margin in 1997", "route": "hybrid"},
    {"question": "Return policy for perishables", "route": "rag"},
    {"question": "Compute top 5 customers by revenue", "route": "sql"},
]

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
PROGRAM_PATH = ARTIFACT_DIR / "router_program.json"
METRICS_PATH = ARTIFACT_DIR / "router_metrics.json"


def _router_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    expected = example.route.strip().lower()
    actual = (pred.route or "").strip().lower()
    return 1.0 if expected == actual else 0.0


def _build_trainset() -> List[dspy.Example]:
    return [dspy.Example(**row).with_inputs("question") for row in ROUTER_DATASET]


class RouterModule(dspy.Module):
    """DSPy-powered router with optimizer + graceful degradation."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(RouterSignature)
        self.optimized = False
        self._compile_with_optimizer()

    def _compile_with_optimizer(self):
        trainset = _build_trainset()
        optimizer = dspy.BootstrapFewShot(metric=_router_metric)
        try:
            self.predictor = optimizer.compile(
                student=self.predictor,
                trainset=trainset,
            )
            self.optimized = True
            metrics = {
                "metric": "accuracy",
                "before": self._evaluate_baseline(trainset),
                "after": self._evaluate_model(trainset),
            }
            METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        except Exception:
            self.optimized = False

    def _evaluate_baseline(self, dataset: List[dspy.Example]) -> float:
        majority = "rag"
        correct = sum(1 for example in dataset if example.route == majority)
        return correct / max(len(dataset), 1)

    def _evaluate_model(self, dataset: List[dspy.Example]) -> float:
        try:
            correct = 0
            for example in dataset:
                pred = self.predictor(question=example.question)
                if (pred.route or "").strip().lower() == example.route.strip().lower():
                    correct += 1
            return correct / max(len(dataset), 1)
        except Exception:
            return 0.0

    def _heuristic_predict(self, question: str) -> str:
        q = question.lower()
        if any(keyword in q for keyword in ["return", "policy", "definition", "doc"]):
            return "rag"
        if any(keyword in q for keyword in ["top", "list", "rank"]) and "revenue" in q:
            return "sql"
        return "hybrid"

    def forward(self, question):
        try:
            return self.predictor(question=question)
        except Exception:
            heuristic_route = self._heuristic_predict(question)
            return SimpleNamespace(route=heuristic_route)