# Prompt Engineering Techniques

Prompt engineering is the practice of designing inputs (prompts) to guide Large Language Models (LLMs) to produce desired outputs. It ranges from simple instructions to complex, programmatic frameworks.

---

## 1. Zero-Shot Prompting
Asking the model to perform a task without providing any examples.
*   **Example:** "Translate 'Hello' to French."
*   **Use Case:** Simple tasks, powerful base/instruct models.
*   **Limitation:** Fails on complex reasoning or strict output formatting constraints.

## 2. Few-Shot Prompting (In-Context Learning)
Providing a few examples (demonstrations) of the task within the prompt before asking the model to perform the final task.
*   **Example:**
    ```text
    Review: "This movie was amazing!" -> Sentiment: Positive
    Review: "Terrible plot and bad acting." -> Sentiment: Negative
    Review: "I loved the cinematography." -> Sentiment:
    ```
*   **Use Case:** Teaching the model a specific format, tone, or handling edge cases without fine-tuning.
*   **Limitation:** Consumes context window space; sensitive to the order and quality of examples.

## 3. Chain-of-Thought (CoT)
Instructing the model to break down complex problems into intermediate reasoning steps.
*   **Example:** Adding "Let's think step by step." to the end of a prompt.
*   **Mechanism:** By generating intermediate tokens, the model essentially buys itself more "compute time" to arrive at the correct answer, drastically reducing hallucination on logic/math problems.
*   **Few-Shot CoT:** Providing examples that *include* the reasoning steps, not just the final answer.

## 4. ReAct (Reason + Act)
An agentic framework that interleaves reasoning and acting. The model thinks about what to do, takes an action (e.g., searching a database, calling an API), observes the result, and then reasons again.
*   **Structure:**
    1.  **Thought:** "I need to find the current stock price of Apple."
    2.  **Action:** `SearchAPI("Apple stock price")`
    3.  **Observation:** "$150.25"
    4.  **Thought:** "Now I have the price. I can answer the user."
    5.  **Final Answer:** "Apple's stock price is $150.25."
*   **Use Case:** Building AI Agents that need to interact with external tools and environments.

## 5. Tree of Thoughts (ToT)
An advanced reasoning framework where the model explores multiple different reasoning paths (branches) simultaneously. It evaluates intermediate steps and can backtrack if a path looks unpromising, similar to search algorithms (BFS/DFS) in classical AI.
*   **Use Case:** Highly complex tasks like creative writing, strategic planning, or difficult mathematical proofs where standard CoT fails.

## 6. System Prompts & Guardrails
*   **System Prompt:** The overarching instruction given to the model (e.g., "You are a helpful customer service assistant for DHL. Never give medical advice."). It sets the persona and boundaries.
*   **Guardrails:** Techniques to ensure safety and compliance. This can include input filtering (checking if the user prompt is malicious) and output filtering (checking if the model's response violates policy).

## Best Practices for Production
*   **Be Specific and Clear:** Avoid ambiguity. Define exact input/output formats (e.g., "Output ONLY valid JSON").
*   **Delimiters:** Use Markdown, XML tags (`<context>...</context>`), or quotes to clearly separate instructions from data.
*   **Give the Model an 'Out':** Tell the model what to do if it doesn't know the answer (e.g., "If you cannot find the answer in the provided text, say 'I do not know'"). This heavily reduces hallucination.
*   **Prompt Evaluation:** Treat prompts like code. Use a test suite of inputs and evaluate prompt changes systematically using metrics or LLM-as-a-judge.