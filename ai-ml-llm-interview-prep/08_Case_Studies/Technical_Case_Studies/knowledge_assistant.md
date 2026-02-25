# Case Study: Enterprise Knowledge Assistant (Agent)

**Company:** Global Logistics (e.g., DHL)
**Scenario:** DHL has 100,000 employees globally. Employees waste an average of 45 minutes a day trying to find internal information across Confluence, Jira, Workday (HR), and proprietary logistics databases. Build an AI Agent (an "internal ChatGPT") that can answer any employee question securely and take basic actions.

---

## 1. Problem Formulation & Constraints
*   **Goal:** Create a unified natural language interface for all internal enterprise systems.
*   **Constraints:**
    *   *Strict Access Control:* A warehouse worker cannot be allowed to query the HR database to see the salaries of managers.
    *   *Hallucination:* The bot cannot hallucinate company policies (e.g., telling an employee they have 30 days of PTO when they only have 15).
    *   *Actionability:* The bot shouldn't just read; it should be able to *do* (e.g., "File an IT ticket for a broken scanner").

## 2. Architecture: Multi-Agent Router Pattern

A single massive prompt trying to handle HR, IT, and Logistics simultaneously will fail. We must use a Multi-Agent architecture.

### Layer 1: The Supervisor (Router)
The user types: *"I broke my scanner, how do I get a new one, and what is my current vacation balance?"*
*   A fast, cheap LLM (Llama 3 8B) acts as a Router.
*   It breaks the query into two tasks and routes them to specialized sub-agents.
    1. Route to `IT_Agent`
    2. Route to `HR_Agent`

### Layer 2: The Specialized Agents
*   **IT_Agent:** Equipped with a RAG tool connected *only* to the IT Confluence space, and an API tool connected to Jira (`create_ticket(description)`).
*   **HR_Agent:** Equipped with an API tool connected to Workday (`get_pto_balance(employee_id)`).

### Layer 3: Security & Identity (Crucial Step)
*   Before the Router even touches the prompt, the backend authenticates the user via Single Sign-On (SSO / Active Directory).
*   The `employee_id` and their `role_permissions` are implicitly injected into every downstream API call and Vector DB query. (See `rag_security.md`). The LLM never "asks" for the user's ID; the backend enforces it.

## 3. The RAG Pipeline (Knowledge Retrieval)
For the general knowledge questions ("What is the travel reimbursement policy?"):
*   **Data Ingestion:** Nightly Airflow jobs crawl Confluence and SharePoint.
*   **Chunking:** Semantic chunking by document headers to keep policy sections intact.
*   **Hybrid Search:** We must use Vector Search + BM25 keyword search, because employees often search for specific acronyms (e.g., "Form QX-99").
*   **Citation:** The final prompt explicitly instructs the Agent: `You must append a markdown link to the source document for every claim you make.`

## 4. Handling "Actions" (Function Calling)
When the user says: "File a ticket for my broken scanner."
1.  The `IT_Agent` outputs a JSON function call: `{"tool": "jira_create", "args": {"title": "Broken Scanner"}}`.
2.  **Human-in-the-Loop (Guardrail):** The backend intercepts this JSON. It does *not* execute it immediately. It renders a UI card in the chat window: *"I am about to file an IT ticket titled 'Broken Scanner'. Click Confirm to proceed."*
3.  Only upon user click does the Python backend execute the actual POST request to Jira.

## 5. Evaluation & Continuous Improvement
*   **Implicit Feedback:** Thumbs Up / Thumbs Down buttons on every response.
*   **RAGAS Evaluation:** A nightly job samples 100 queries, runs them through the pipeline, and uses GPT-4 to score the *Faithfulness* (Did it hallucinate?) and *Context Relevance* (Did the vector DB find the right policy?).
*   **Query Clustering:** We run unsupervised clustering (K-Means) on all user queries weekly to identify the "Top 10 unanswered questions" (queries where the bot returned "I don't know"), allowing the documentation team to write new wiki articles to fill the gaps.

## Interview Strategy
"The key to an enterprise assistant is **Trust**. I would never deploy an autonomous action-taking agent. I would architect a **Router-based Multi-Agent system** where data retrieval is strictly gated by **SSO metadata pre-filtering**, and any state-changing API calls require a physical 'Human-in-the-Loop' confirmation click. This provides the UX of AI with the security of traditional software."