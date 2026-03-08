"""LLM-agnostic Graph QA agent with strict evidence traceability.

Default provider: Google Gemini via ``langchain-google-genai``.
The agent answers queries exclusively from the BioCypher Micro-CKG,
citing exact ``(Source)--[Edge_Type, Score=X.XX]-->(Target)`` evidence
paths. Speculation beyond graph contents is prohibited.
"""

from __future__ import annotations

import os
from typing import Any

import networkx as nx
from dotenv import load_dotenv

__all__ = [
    "get_llm",
    "serialize_graph",
    "build_traceability_prompt",
    "create_qa_agent",
    "query_graph",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SUPPORTED_PROVIDERS = ("google", "openai")

_SYSTEM_PROMPT = """\
You are a precise biomedical knowledge-graph analyst. You answer questions
using ONLY the Micro-CKG (Clinical Knowledge Graph) provided below.

## Strict Rules

1. **Evidence Only**: Every claim in your answer MUST be supported by
   explicit graph evidence. Cite each piece of evidence in this exact
   format:
   ``(SourceNode) --[EdgeType, Score=X.XXXX]--> (TargetNode)``

2. **No Speculation**: If the graph does not contain sufficient evidence
   to answer the question, respond with:
   "No evidence found in the Micro-CKG for this query."

3. **Quantitative Precision**: Always include numerical scores
   (stability_score, mean_expression, spatial_correlation,
   enrichment_score) from the edge properties. Do not round beyond
   four decimal places.

4. **Objective Language**: Use rigorous, objective terminology.
   Describe results as "high stability score", "statistically
   significant", or "objective feature convergence". Never use
   subjective adjectives such as "loose", "strong", or "impressive".

5. **Structure**: Organise multi-part answers with numbered lists.
   Group evidence by node type (Gene, CellType, AnatomicalEntity).

## Micro-CKG Data

{graph_context}
"""

_HUMAN_TEMPLATE = """\
Question: {question}

Provide your answer with full evidence traceability from the Micro-CKG.
"""


# ---------------------------------------------------------------------------
# LLM Provider Factory
# ---------------------------------------------------------------------------


def get_llm(
    provider: str = "google",
    model: str | None = None,
    temperature: float = 0.0,
) -> Any:
    """Instantiate a LangChain chat model for the requested provider.

    Loads API keys from environment variables (via ``.env`` file).

    Args:
        provider: LLM provider — ``"google"`` (default) or ``"openai"``.
        model: Model name override. Defaults to ``gemini-2.0-flash``
            for Google and ``gpt-4o-mini`` for OpenAI.
        temperature: Sampling temperature (0.0 = deterministic).

    Returns:
        A LangChain ``BaseChatModel`` instance.

    Raises:
        ValueError: If *provider* is not supported.
        EnvironmentError: If the required API key is not set.
    """
    load_dotenv()

    if provider == "google":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY not set. Add it to your .env file."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.0-flash",
            google_api_key=api_key,
            temperature=temperature,
        )

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Add it to your .env file."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            api_key=api_key,
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported provider '{provider}'. Choose from: {_SUPPORTED_PROVIDERS}"
    )


# ---------------------------------------------------------------------------
# Graph Serialiser
# ---------------------------------------------------------------------------


def serialize_graph(graph: nx.DiGraph) -> str:
    """Convert a NetworkX Micro-CKG to a structured text representation.

    The output lists all nodes with their properties, then all edges
    with their quantitative attributes. This string is injected into the
    LLM system prompt as the knowledge base.

    Args:
        graph: The Micro-CKG as a NetworkX DiGraph.

    Returns:
        A multi-line string representation of all nodes and edges.
    """
    lines: list[str] = []

    # --- Nodes ---
    lines.append("### Nodes")
    for node_id, data in sorted(graph.nodes(data=True)):
        label = data.get("label", "unknown")
        props = {k: v for k, v in data.items() if k != "label"}
        prop_str = ", ".join(f"{k}={v}" for k, v in props.items())
        lines.append(f"- [{label}] {node_id} ({prop_str})")

    # --- Edges ---
    lines.append("")
    lines.append("### Edges")
    for src, tgt, data in sorted(graph.edges(data=True)):
        label = data.get("label", "association")
        props = {k: v for k, v in data.items() if k not in ("label", "key")}
        prop_str = ", ".join(f"{k}={v}" for k, v in props.items())
        lines.append(f"- ({src}) --[{label}, {prop_str}]--> ({tgt})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt & Agent Construction
# ---------------------------------------------------------------------------


def build_traceability_prompt() -> str:
    """Return the system prompt template enforcing evidence traceability.

    Returns:
        The system prompt string with a ``{graph_context}`` placeholder.
    """
    return _SYSTEM_PROMPT


def create_qa_agent(
    graph: nx.DiGraph,
    provider: str = "google",
    model: str | None = None,
) -> Any:
    """Create a LangChain QA chain backed by the Micro-CKG.

    The chain serialises the full graph into the system prompt and
    enforces strict evidence-traceability rules.

    Args:
        graph: The Micro-CKG as a NetworkX DiGraph.
        provider: LLM provider (``"google"`` or ``"openai"``).
        model: Optional model name override.

    Returns:
        A LangChain ``RunnableSequence`` that accepts ``{"question": str}``
        and returns a string answer with evidence paths.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = get_llm(provider=provider, model=model)
    graph_context = serialize_graph(graph)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT.replace("{graph_context}", graph_context)),
        ("human", _HUMAN_TEMPLATE),
    ])

    chain = prompt | llm | StrOutputParser()
    return chain


def query_graph(
    agent: Any,
    question: str,
) -> str:
    """Submit a question to the evidence-traced QA agent.

    Args:
        agent: The LangChain ``RunnableSequence`` from
            :func:`create_qa_agent`.
        question: Natural-language question about the Micro-CKG.

    Returns:
        The LLM's answer including ``(Source)--[Edge, Score=X.XX]-->
        (Target)`` evidence citations.
    """
    print(f"  Querying agent: {question[:80]}...")
    answer = agent.invoke({"question": question})
    return answer
