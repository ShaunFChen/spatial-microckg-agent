"""LLM-agnostic Graph QA agent with strict evidence traceability.

Default provider: Ollama (local) via ``langchain-ollama``.
The agent answers queries exclusively from the BioCypher Micro-CKG,
citing exact ``(Source)--[Edge_Type, Score=X.XX]-->(Target)`` evidence
paths. Speculation beyond graph contents is prohibited.
"""

from __future__ import annotations

import os
import time
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
_SUPPORTED_PROVIDERS = ("google", "openai", "ollama")
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_INITIAL_DELAY = 30.0

_SYSTEM_PROMPT = """\
You are an objective computational biology assistant analyzing a \
Micro-Clinical Knowledge Graph (Micro-CKG) derived from Spatial \
Transcriptomics data.

Your sole function is to retrieve and report factual relationships \
explicitly defined in the provided graph context. The features in this \
graph were objectively selected using the Stabl algorithm.

**STRICT RULES - YOU MUST FOLLOW THESE:**

1. NO EXTERNAL KNOWLEDGE: You must ONLY formulate your answer based on \
the nodes and edges provided in the Context below. Do not use your \
training data, internet sources, or parameterized knowledge.

2. MISSING DATA FALLBACK: If the Context does not explicitly contain \
the answer, you must output exactly: "No evidence found in the current \
Micro-CKG." Do not attempt to guess or infer.

3. MANDATORY CITATION: Every biological claim you make MUST be followed \
by its exact structural citation from the graph, formatted exactly as: \
`[Evidence: (Source_Node) --(Edge_Type)--> (Target_Node)]`. Include \
statistical metrics if present in the edge attributes.

4. OBJECTIVE TONE: Maintain a highly professional tone. Do not use \
subjective descriptors (e.g., "loose", "strong", "impressive") to \
qualify biological relationships.

Context:
{graph_context}
"""

_HUMAN_TEMPLATE = """\
Question: {question}
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
        provider: LLM provider — ``"google"`` (default), ``"openai"``,
            or ``"ollama"`` (local, no API key needed).
        model: Model name override. Defaults to ``gemini-2.0-flash``
            for Google, ``gpt-4o-mini`` for OpenAI, and
            ``llama3.1:8b`` for Ollama.
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

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or "deepseek-r1:14b",
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
    enforces strict evidence-traceability rules designed to prevent
    hallucination in local open-source models.

    Args:
        graph: The Micro-CKG as a NetworkX DiGraph.
        provider: LLM provider (``"google"``, ``"openai"``, or
            ``"ollama"`` for local models).
        model: Optional model name override.

    Returns:
        A LangChain ``RunnableSequence`` that accepts ``{"question": str}``
        and returns a string answer with mandatory evidence citations.
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
    max_retries: int = _DEFAULT_MAX_RETRIES,
    initial_delay: float = _DEFAULT_INITIAL_DELAY,
) -> str:
    """Submit a question to the evidence-traced QA agent with retry.

    Retries with exponential backoff on rate-limit (429) errors.
    The delay doubles after each failed attempt, starting from
    *initial_delay* seconds.

    Args:
        agent: The LangChain ``RunnableSequence`` from
            :func:`create_qa_agent`.
        question: Natural-language question about the Micro-CKG.
        max_retries: Maximum number of retry attempts on rate-limit
            errors.
        initial_delay: Initial wait time in seconds before the first
            retry (doubles each attempt).

    Returns:
        The LLM's answer including ``(Source)--[Edge, Score=X.XX]-->
        (Target)`` evidence citations.

    Raises:
        RuntimeError: If all retry attempts are exhausted.
    """
    print(f"  Querying agent: {question[:80]}...")
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            answer = agent.invoke({"question": question})
            return answer
        except Exception as exc:  # noqa: BLE001
            err_str = str(exc).lower()
            is_rate_limit = "429" in err_str or "resource_exhausted" in err_str or "too many requests" in err_str
            if not is_rate_limit or attempt == max_retries:
                raise
            print(f"  Rate-limited (attempt {attempt + 1}/{max_retries + 1}). "
                  f"Retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay *= 2

    raise RuntimeError("query_graph: all retry attempts exhausted")
