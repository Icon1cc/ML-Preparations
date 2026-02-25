# LLM Application Design Patterns

Designing an LLM application is about managing the inherent unreliability and latency of generative AI. These are the architectural patterns used by senior AI engineers.

---

## 1. The Gateway Pattern
Do not allow individual services to call OpenAI/Anthropic directly. 
*   **LLM Gateway:** A central internal service that handles:
    *   **Unified API:** Switches between GPT-4, Claude, and Llama without changing client code.
    *   **Fallback/Retries:** If OpenAI is down, automatically switch to Anthropic.
    *   **Rate Limiting:** Prevents one rogue developer from blowing the budget.
    *   **Caching:** Semantic caching to save costs on repetitive queries.

## 2. The Router Pattern
A "Manager" LLM analyzes the incoming query and decides which specialized pipeline to trigger.
*   **Example:** 
    *   *Query:* "How do I reset my password?" -> **Router** -> `Knowledge Base RAG`
    *   *Query:* "What's my account balance?" -> **Router** -> `SQL Agent`
    *   *Query:* "Translate 'Hello' to Spanish" -> **Router** -> `Direct Translation Prompt`

## 3. The Guardrail Pattern
Placing programmatic or model-based checks before and after the LLM.
*   **Input Guardrails:** Detect PII, prompt injection, or off-topic queries (e.g., asking a support bot about politics).
*   **Output Guardrails:** Detect hallucinations, toxic language, or leaks of internal system prompts.
*   **Tools:** `NeMo Guardrails`, `Guardrails AI`, or custom LLM-based classifiers.

## 4. The Orchestrator (Agent) Pattern
Moving from static prompts to dynamic loops.
*   Instead of a fixed sequence of steps, the LLM is given **Tools** and a **Goal**. It decides which tools to call, observes the results, and continues until the goal is met. (See Agent Architectures file).

## 5. The Batch Processing Pattern
LLM inference is slow and expensive.
*   **Pattern:** For tasks like document classification, data cleaning, or summarization, use **Async Batching**. Push requests to a queue (SQS/RabbitMQ), process them with workers, and store results in a DB. 
*   *Tip:* Many providers (like OpenAI) offer **Batch APIs** at a 50% discount if you can wait 24 hours for the results.

## 6. Evaluations as a First-Class Citizen
In standard software, you have unit tests. In LLM apps, you have **Evals**.
*   Every change to a prompt or model must be run through an "Eval Suite" (a set of 50-100 test cases) where a "Judge LLM" grades the output. You cannot rely on manual "vibe checks."

## 7. Context Management Strategies
*   **Window Sliding:** For long chats, only keep the last $N$ turns to save tokens.
*   **Summarization:** When the chat history gets too long, ask the LLM to summarize the past 20 messages into a single "Memory" block, then clear the history.
*   **Knowledge Graph (GraphRAG):** Instead of just vector chunks, store information as a graph of entities and relationships. This allows the LLM to answer complex, multi-hop questions like "How are the CEO of DHL and the CEO of UPS related?"

## Interview Tip: "Buy vs. Build"
If asked how to build an LLM app:
"I would start by using a managed framework like **LangChain** or **LlamaIndex** to build a prototype quickly. However, for a production-scale system, these frameworks can be 'too magical' and hard to debug. I would likely transition to a custom, lightweight implementation using a **Gateway** and an **Evaluations framework** to ensure reliability and observability."