# Orchestration Frameworks: LangChain, LangGraph, LlamaIndex, AutoGen, CrewAI, DSPy

## Why frameworks
They speed prototyping of RAG/agent systems by providing connectors, abstractions, tracing, and evaluation hooks.

## Framework snapshots
- LangChain: broad ecosystem, many integrations.
- LangGraph: graph/state-machine orchestration for complex flows.
- LlamaIndex: data ingestion + retrieval focused.
- AutoGen: multi-agent conversation orchestration.
- CrewAI: role-based multi-agent workflow.
- DSPy: programmatic prompt optimization.

## Comparison table

| Framework | Best for | Strength | Risk |
|---|---|---|---|
| LangChain | rapid prototyping | rich integrations | abstraction complexity |
| LangGraph | production agents | explicit state graph | steeper mental model |
| LlamaIndex | RAG-first apps | ingestion/query quality | less general agent focus |
| AutoGen | multi-agent chats | role coordination | orchestration overhead |
| CrewAI | team-like workflows | fast role/task setup | brittle prompt dependence |
| DSPy | optimization | systematic prompt tuning | setup complexity |

## Framework vs custom code
Use frameworks when:
- timeline is short
- integrations are standard

Use custom when:
- strict latency/cost/security constraints
- tight control over retrieval and orchestration internals

## Interview questions
1. What would you choose for production RAG and why?
2. Drawbacks of heavy abstraction in LangChain?
3. When is custom orchestration justified?

## Side-by-side concept
- Framework pipeline: quick assembly with built-ins.
- Raw Python: explicit control, lower hidden overhead.
