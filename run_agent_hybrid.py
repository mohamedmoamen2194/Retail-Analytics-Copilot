import argparse
import json
from typing import Any, Dict

from agent.lang_graph import AgentState, build_graph


def project_contract(result: Any) -> Dict[str, Any]:
    data = result.__dict__ if hasattr(result, "__dict__") else result
    return {
        "id": data.get("id") or data.get("item_id"),
        "final_answer": data.get("final_answer"),
        "sql": data.get("sql", ""),
        "confidence": data.get("confidence", 0.0),
        "explanation": data.get("explanation", ""),
        "citations": data.get("citations", []),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--out", default="outputs_hybrid.jsonl")
    args = parser.parse_args()

    compiled = build_graph()

    outputs = []
    with open(args.batch, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            state = AgentState(
                item_id=item["id"],
                question=item["question"],
                format_hint=item["format_hint"],
            )
            result = compiled.invoke(state)
            outputs.append(project_contract(result))

    with open(args.out, "w", encoding="utf-8") as f:
        for out in outputs:
            json.dump(out, f, default=str)
            f.write("\n")

    print(f"Done. Results written to {args.out}")

if __name__ == "__main__":
    main()
