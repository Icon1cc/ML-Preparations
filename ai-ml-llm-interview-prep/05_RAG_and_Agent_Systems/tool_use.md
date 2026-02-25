# Agent Tool Use (Function Calling deeply applied)

While the LLM section covers the *mechanism* of Function Calling (JSON generation), this file covers the *design* of tools for Agentic systems in production.

---

## 1. Tool Design Principles
An LLM is only as smart as the tools you give it. Bad tools lead to infinite loops and hallucinations.

### A. Principle of Least Privilege
*   Agents are prone to prompt injection. Never give an Agent a `execute_sql(query=str)` tool with root database access.
*   Instead of raw SQL, give it highly parameterized endpoints: `get_user_order_history(user_id=int)`. The backend controls the SQL.

### B. Graceful Degradation (Error Handling)
*   If a tool fails (e.g., API 500 error, or the LLM passes a string instead of an int), your backend **must not crash**.
*   It must return a clear, natural language error back to the LLM.
*   *Backend Return:* `{"error": "Invalid User ID format. Please ask the user for a 5-digit number."}`
*   The LLM reads this and self-corrects its behavior.

### C. The "Idempotency" Rule
*   Because Agents can enter loops and call tools multiple times accidentally, state-changing tools (like `charge_credit_card`) must be idempotent. Calling them twice with the same transaction ID should not result in a double charge.

## 2. Common Agent Tools in Logistics/Enterprise

1.  **Vector Search (RAG Tool):** The Agent decides *when* to search the knowledge base, rather than forcing a search on every turn.
    *   `search_company_wiki(query: str)`
2.  **SQL / Analytics APIs:**
    *   `get_shipment_status(tracking_number: str)`
    *   `calculate_customs_duty(country_code: str, item_value: float)`
3.  **Action / Mutation Tools:**
    *   `create_support_ticket(summary: str, severity: str)`
    *   *Crucial:* Action tools usually require a "Human-in-the-Loop" confirmation step in the UI before execution.
4.  **Utility Tools:**
    *   `get_current_date_time()` (LLMs do not inherently know what day it is).
    *   `calculator(math_expression: str)` (LLMs are terrible at arithmetic; let a Python `eval` or math library do it).

## 3. Context Window Management with Tools
A major problem: An Agent calls `get_customer_history` and the backend returns 10,000 lines of JSON. This blows up the context window, causing the LLM to crash or forget its original instructions.

### The "Tool Proxy" Pattern
*   Instead of returning raw data to the LLM, the tool saves the massive data to a temporary file or database table.
*   The tool returns a summary or a reference ID to the LLM: `{"result": "Data saved to Table Temp_123. Use the 'query_table' tool to ask specific questions about it."}`
*   This keeps the Agent's working memory clean.

## 4. The ReAct Loop Implementation
The core code loop for an Agent using tools:

```python
# Pseudo-code
messages = [{"role": "user", "content": "Where is my package 123?"}]

while True:
    response = call_llm(messages, tools=available_tools)
    
    if response.tool_calls:
        # The LLM wants to use a tool
        for tool_call in response.tool_calls:
            # 1. Execute the Python function safely
            result = execute_tool(tool_call.name, tool_call.arguments)
            # 2. Append the result to history
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        
        # Loop continues, LLM will now read the tool result
        
    else:
        # The LLM just generated normal text, task is complete
        return response.content
```