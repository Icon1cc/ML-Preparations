# Case Study: Agent Workflow Automation (Customer Support)

**Company:** Global Logistics (e.g., DHL)
**Scenario:** When a massive storm hits an airport hub, thousands of flights are grounded. DHL receives a surge of 50,000 angry customer emails asking "Where is my package?" and "I demand a refund." Human agents cannot handle the spike. Design an autonomous AI workflow to resolve these emails end-to-end without human intervention.

---

## 1. Problem Formulation
*   **Objective:** Autonomously resolve or deflect >60% of tier-1 support emails during massive spike events to maintain SLA compliance.
*   **The Challenge:** Emails are unstructured, angry, and often contain multiple requests in one paragraph.
*   **Constraint:** Financial actions (like issuing a $500 refund) require high confidence and adherence to strict corporate logic trees.

## 2. Architecture: The AI Triage Pipeline (LangGraph)

We will use a State-Machine/Graph workflow (e.g., LangGraph) rather than a free-thinking ReAct agent to ensure deterministic, safe behavior.

### Step 1: Ingestion & Classification (The Gatekeeper)
An email arrives. It is passed to a fast, cheap model (e.g., Llama-3-8B).
*   **Task:** Classification and Entity Extraction.
*   **Output JSON:**
    ```json
    {
      "intent": "refund_request",
      "sentiment": "highly_angry",
      "tracking_number": "DHL-1234598",
      "is_urgent_medical": false
    }
    ```
*   **Routing Logic:** If `is_urgent_medical == true`, immediately route to the High-Priority Human Queue. Skip AI.

### Step 2: Data Enrichment (The API Step)
The workflow executes deterministic Python code based on the tracking number.
*   Calls internal DHL database: `SELECT status, delay_reason, item_value FROM shipments WHERE id = 'DHL-1234598'`.
*   *Result:* Status is "Delayed", Reason is "Weather (Storm XYZ)", Value is $100.

### Step 3: Business Logic Evaluation
We do *not* use the LLM to decide if the customer gets a refund. We use hardcoded Python logic to ensure 100% compliance.
*   `if delay_reason == 'Weather' and SLA_breached == True:` -> Eligible for 20% partial refund.
*   The system state is updated: `Action_Approved: Partial Refund $20`.

### Step 4: Generation (The Draft Agent)
The enriched data and the approved action are passed to a highly capable drafting model (e.g., GPT-4o or Claude 3.5 Sonnet).
*   **Prompt:** `You are a polite DHL agent. The customer's package is delayed due to a storm. Apologize empathetically. Inform them the package is safe and will arrive in 48 hours. Offer the approved $20 partial refund for the inconvenience. Do NOT promise anything else.`
*   **Output:** The model generates a highly empathetic, customized email draft.

### Step 5: The Critic (Output Guardrail)
Before sending, a final tiny LLM checks the draft against a strict rubric.
*   "Did the draft mention the $20 refund exactly?" (Yes).
*   "Did the draft promise a specific delivery time not in the context?" (No).
*   If it fails, it goes to the Human Review Queue. If it passes, it is emailed to the customer, and the API issues the refund.

## 3. Handling Ambiguity (Human-in-the-Loop)
If the tracking number is missing, or the customer asks a highly complex legal question:
*   The Classification Agent in Step 1 outputs `intent: unknown_complex`.
*   The system pauses the graph, attaches a summary of the AI's failed reasoning, and drops it into a Zendesk queue for a human agent. The human agent starts with 80% of the research already done.

## 4. Scalability (Handling the Spike)
*   During a storm, the system might receive 100 emails a second.
*   **Architecture:** We use an **Event-Driven Architecture**. Emails are dropped into an **Apache Kafka** queue. A cluster of Kubernetes pods running the LangGraph workflow pulls emails from the queue asynchronously. If the queue backs up, Kubernetes auto-scales the pods to burn through the backlog.

## Interview Strategy
"When automating customer support, the biggest risk is the LLM hallucinating a policy or giving away free money. My architecture actively prevents this by **separating cognition from execution**. I use the LLM purely for unstructured text parsing (extracting the tracking ID) and empathetic text generation (writing the final email). The actual decision to issue a refund is governed entirely by a **deterministic Python state machine** that queries our internal databases, ensuring 100% policy compliance."