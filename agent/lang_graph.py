from dataclasses import dataclass
import json
from typing import Any

from langgraph.constants import END
from langgraph.graph import StateGraph

from agent.rag.retrieval import Retriever
from agent.graph_hybrid import HybridAgent
from agent.tools.sqlite_tool import SQLiteTool
from agent.dspy_signatures import RouterModule


router_module = RouterModule()
retriever = Retriever(docs_path="docs")
retriever.load_corpus()
sqlite_tool = SQLiteTool(db_path="data/northwind.sqlite")
agent = HybridAgent(router_module=router_module, retriever=retriever, sqlite_tool=sqlite_tool)


@dataclass
class AgentState:
    item_id: str = ""
    question: str = ""
    format_hint: str = ""
    route: str = ""
    retrieved_chunks: list = None
    constraints: dict = None
    sql: str = ""
    sql_res: dict = None
    final_answer: Any = None
    confidence: float = 0.0
    explanation: str = ""
    citations: list = None
    tables: list = None
    table_schemas: dict = None
    repairs: int = 0
    max_repairs: int = 2
    needs_repair: bool = False
    validation_error: str = ""

    def __post_init__(self):
        self.retrieved_chunks = self.retrieved_chunks or []
        self.constraints = self.constraints or {}
        self.sql_res = self.sql_res or {"success": True, "rows": []}
        self.citations = self.citations or []
        self.tables = self.tables or []
        self.table_schemas = self.table_schemas or {}


def node_router(state: AgentState) -> dict:
    route = agent.route(state.question)
    return {"route": route}


def node_retriever(state: AgentState) -> dict:
    chunks = agent.retrieve(state.question, top_k=10)
    return {
        "retrieved_chunks": chunks,
    }


def node_planner(state: AgentState) -> dict:
    constraints = agent.plan(state.question, state.retrieved_chunks)
    return {"constraints": constraints}


def node_schema(state: AgentState) -> dict:
    return {
        "tables": agent.artifacts.tables,
        "table_schemas": agent.artifacts.table_schemas,
    }


def node_nl2sql(state: AgentState) -> dict:
    sql = agent.generate_sql(state.question, state.constraints)
    return {"sql": sql}


def node_executor(state: AgentState) -> dict:
    executed_sql, sql_res = agent.execute_sql(state.sql, attempt=state.repairs)
    return {"sql": executed_sql, "sql_res": sql_res}


def node_validator(state: AgentState) -> dict:
    outcome = "synthesize"
    needs_repair = False
    validation_error = ""

    if not state.sql:
        needs_repair = state.repairs < state.max_repairs
        validation_error = "Empty SQL"
    elif not state.sql_res.get("success"):
        needs_repair = state.repairs < state.max_repairs
        validation_error = state.sql_res.get("error", "SQL error")
    elif not state.sql_res.get("rows"):
        needs_repair = state.repairs < state.max_repairs
        validation_error = "No rows"

    if needs_repair:
        outcome = "repair"

    return {
        "needs_repair": needs_repair,
        "validation_error": validation_error,
        "next_step": outcome,
    }


def node_repair(state: AgentState) -> dict:
    updated_constraints = dict(state.constraints)
    if state.repairs == 0 and updated_constraints.get("date_range"):
        updated_constraints.pop("date_range")
    elif state.repairs == 1 and updated_constraints.get("category"):
        updated_constraints.pop("category")
    sql = agent.generate_sql(state.question, updated_constraints)
    return {
        "constraints": updated_constraints,
        "sql": sql,
        "repairs": state.repairs + 1,
    }


def node_synthesizer(state: AgentState) -> dict:
    out = agent.synthesize(
        item_id=state.item_id,
        question=state.question,
        format_hint=state.format_hint,
        route=state.route,
        sql=state.sql,
        sql_res=state.sql_res,
        chunks=state.retrieved_chunks,
        constraints=state.constraints,
    )
    return out


def node_trace(state: AgentState) -> dict:
    log_entry = {
        "item_id": getattr(state, "item_id", ""),
        "question": state.question,
        "route": state.route,
        "constraints": state.constraints,
        "sql": state.sql,
        "sql_success": state.sql_res.get("success", True),
        "repairs": state.repairs,
        "needs_repair": getattr(state, "needs_repair", False),
    }
    with open("trace.jsonl", "a", encoding="utf-8") as f:
        json.dump(log_entry, f, ensure_ascii=False)
        f.write("\n")
    return {}


def planner_branch(state: AgentState) -> str:
    if state.route in {"sql", "hybrid"}:
        return "sql"
    return "rag"


def validator_branch(state: AgentState) -> str:
    return getattr(state, "next_step", "synthesize")


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("router", node_router)
    graph.add_node("retriever", node_retriever)
    graph.add_node("planner", node_planner)
    graph.add_node("schema", node_schema)
    graph.add_node("nl2sql", node_nl2sql)
    graph.add_node("executor", node_executor)
    graph.add_node("validator", node_validator)
    graph.add_node("repair", node_repair)
    graph.add_node("synthesizer", node_synthesizer)
    graph.add_node("trace", node_trace)

    graph.add_edge("router", "retriever")
    graph.add_edge("retriever", "planner")

    graph.add_conditional_edges(
        "planner",
        planner_branch,
        {
            "rag": "synthesizer",
            "sql": "schema",
        },
    )

    graph.add_edge("schema", "nl2sql")
    graph.add_edge("nl2sql", "executor")
    graph.add_edge("executor", "validator")

    graph.add_conditional_edges(
        "validator",
        validator_branch,
        {
            "repair": "repair",
            "synthesize": "synthesizer",
        },
    )

    graph.add_edge("repair", "executor")
    graph.add_edge("synthesizer", "trace")
    graph.add_edge("trace", END)

    graph.set_entry_point("router")

    return graph.compile()