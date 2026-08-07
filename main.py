"""
main.py

Command-line interface for Agentic RAG.
"""

from dotenv import load_dotenv

from config import RAGConfig
from graph import AgenticRAGGraph
from llm import warm_up


class AgenticRAGApp:
    """
    CLI application wrapper for the Agentic RAG graph.
    """

    def __init__(self):
        load_dotenv()

        self.config = RAGConfig()
        self.graph = AgenticRAGGraph(self.config)

    def print_debug_info(self, result: dict) -> None:
        """
        Print useful debug information after each answer.
        """
        evidence = result.get("evidence", [])

        print("\n--- Debug Info ---")
        print(f"Sub-questions: {result.get('sub_questions', [])}")
        print(f"Needed research: {result.get('needs_external_research', False)}")

        if result.get("research_notes"):
            print(f"Research keywords (planning only): {result['research_notes']}")

        print(f"Retrieval passes: {result.get('iterations', 0)}")
        print(f"Steps executed: {len(result.get('executed_steps', []))}")

        for step in result.get("plan", []):
            scope = ", ".join(step["cluster_ids"] + step["parent_groups"]) or "unscoped"
            print(f"  last plan: {step['tool']} ({scope}) - {step['sub_question']}")

        tools_used = sorted({item["tool"] for item in evidence})
        sources = sorted({item["source"] for item in evidence})

        print(f"Evidence items: {len(evidence)} via {tools_used or 'none'}")
        print(f"Sources: {sources or 'none'}")
        print(f"Verdict: {result.get('verdict', '')}")
        print(f"Verdict reason: {result.get('verdict_reason', '')}")

        if result.get("gaps"):
            print(f"Gaps: {result['gaps']}")

    def run(self) -> None:
        """
        Start interactive chat loop.
        """
        # Pay the ~5.6GB model load here, where it is visibly a startup step,
        # rather than silently inside the first question.
        print("Loading model...")
        warm_up(self.graph.nodes.llm)

        print("Agentic RAG is ready.")
        print("Type 'exit' or 'quit' to stop.")

        while True:
            question = input("\nAsk a question: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break

            if not question:
                continue

            result = self.graph.ask(question)

            print("\n--- Final Answer ---")
            print(result["answer"])

            self.print_debug_info(result)


def main() -> None:
    app = AgenticRAGApp()
    app.run()


if __name__ == "__main__":
    main()