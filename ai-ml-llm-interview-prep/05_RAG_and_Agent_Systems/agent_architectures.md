# Agent Architectures

An LLM on its own is just a text generator. An **Agent** is an LLM equipped with a framework that allows it to reason, plan, use tools, and interact with an environment.

---

## 1. Single Agent Architectures

### A. ReAct (Reason + Act)
The foundational agent framework. It forces the LLM to think before taking an action.
*   **Loop:** `Thought` -> `Action` -> `Observation` -> (Repeat)
*   **Example:**
    *   *Thought:* I need to find the CEO of Company X. I should search Google.
    *   *Action:* `Search("CEO of Company X")`
    *   *Observation:* "Jane Smith is the CEO of Company X."
    *   *Thought:* I have the answer. I will respond to the user.
    *   *Final Answer:* "The CEO is Jane Smith."
*   **Pros:** Easy to implement, highly effective for straightforward tool use.
*   **Cons:** Struggles with long, complex multi-step tasks (loses context or gets stuck in an infinite loop).

### B. Plan-and-Solve (or Plan-and-Execute)
Addresses the shortcomings of ReAct for complex tasks by separating the *planning* phase from the *execution* phase.
*   **Phase 1 (Planner):** An LLM receives the user's complex goal and outputs a step-by-step plan (e.g., "1. Look up user's tracking ID. 2. Look up weather along the route. 3. Estimate delay.").
*   **Phase 2 (Executor):** A separate loop (often a ReAct agent) executes step 1. Once complete, it moves to step 2.
*   **Pros:** Much better at long-horizon tasks. More robust to errors (if step 2 fails, the planner can be called to re-plan).

### C. Reflection / Self-Correction
Adding a critique step to the agent loop.
*   After generating a final answer, the agent (or a separate "Judge" LLM) is prompted to review its own work against a rubric (e.g., "Did you use the correct API? Is the tone polite?"). If it fails, it generates a new `Thought` to fix the error.

## 2. Multi-Agent Systems
Instead of one massive prompt trying to be a "Swiss Army Knife," we use multiple specialized agents that communicate with each other.

### Why Multi-Agent?
*   **Separation of Concerns:** A "Research Agent" has a system prompt focused entirely on finding data. A "Writer Agent" has a system prompt focused entirely on tone and formatting.
*   **Simpler Prompts:** Smaller, focused prompts hallucinate less and are easier to debug.

### Communication Patterns (e.g., AutoGen, CrewAI)
1.  **Hierarchical (Supervisor):** A "Manager" agent receives the user request and delegates sub-tasks to "Worker" agents (e.g., Coder Agent, Reviewer Agent). The workers report back to the Manager.
2.  **Sequential:** Agent A does its job, passes the output to Agent B, who passes it to Agent C. (Like an assembly line).
3.  **Conversational/Debate:** Two agents converse with each other until they reach a consensus. (e.g., A "Generator" agent creates an itinerary, a "Critic" agent points out flaws, the Generator revises it).

## 3. Core Components of an Agent
To build an agent from scratch (without LangChain), you need:
1.  **The LLM (The Brain):** Usually needs to be a highly capable model (GPT-4 class) because reasoning and strict JSON output formatting are difficult for smaller models.
2.  **Memory:**
    *   *Short-term:* The current conversation history (managed via the context window).
    *   *Long-term:* A Vector Database or SQL database to recall past user preferences.
3.  **Tools (Functions):** A JSON schema defining what external APIs the LLM can call (e.g., `get_weather(location="Berlin")`, `execute_sql(query="...")`). The LLM does not *run* the code; it outputs the JSON request, the Python backend executes the tool, and feeds the string result back to the LLM.