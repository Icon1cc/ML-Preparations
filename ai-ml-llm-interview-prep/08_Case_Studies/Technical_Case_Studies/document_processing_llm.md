# Case Study: Document Processing with LLMs

**Company:** Global Logistics
**Scenario:** A logistics company receives thousands of "Bill of Lading" (BoL) and Customs Declaration documents daily in various formats (PDF, JPG, Scanned faxes). These documents contain critical data like: Shipper Name, Consignee, Item Weights, and Harmonized System (HS) codes. Manual entry is slow and error-prone. Design an AI system to automate this.

---

## 1. Problem Formulation
*   **Goal:** Automate data extraction with $>98\%$ field-level accuracy.
*   **Complexity:** Documents are unstructured, semi-structured, and often of poor scan quality.
*   **Key Challenge:** Standard NLP fails on spatial layouts. (e.g., a number might mean "Weight" because it's under a specific box on the page, not because of the words surrounding it).

## 2. Technical Approach: Intelligent Document Processing (IDP)

### Step 1: Image Pre-processing & OCR
1.  **Quality Check:** Use a simple CNN to detect blurry or rotated images. Ask for a re-scan if quality is too low.
2.  **OCR (Optical Character Recognition):** Use high-quality OCR (Azure Form Recognizer, Google Document AI, or Tesseract/PaddleOCR).
    *   *Crucial:* Don't just get a text string. Get the **Bounding Boxes** (x, y coordinates) for every word.

### Step 2: Extraction Strategy

#### Option A: The "Naive" LLM Approach (Low Volume)
*   Send the raw OCR text to GPT-4o with a prompt: "Extract the Consignee Name from this text."
*   *Failure Mode:* If the OCR text is "shuffled" (as often happens with multi-column PDFs), the LLM will lose the spatial context and hallucinate.

#### Option B: Layout-Aware LLMs (The Pro Approach)
Use a model that natively understands both text and position.
1.  **LayoutLM (Microsoft):** A Transformer-based model that takes three inputs: Token Embeddings, 2D Position Embeddings (from the bounding boxes), and Image Embeddings. It "sees" that the word "Consignee" is physically above the name "John Doe."
2.  **Multi-modal LLMs (GPT-4o / Claude 3.5 Sonnet):** Instead of OCR, send the *entire image* of the document to the model. These models are exceptionally good at spatial reasoning and can extract data directly into JSON.

### Step 3: Entity Resolution & Validation
*   **HS Code Validation:** The extracted HS code (e.g., `8517.12`) must be validated against a master database of customs codes. If the code doesn't exist, flag it for human review.
*   **Fuzzy Matching:** "Express Logistics" and "Express Log." should resolve to the same internal Entity ID. Use a library like `RapidFuzz` or a small Bi-Encoder model.

## 3. The "Human-in-the-Loop" (HITL) Workflow
For high-stakes logistics, 100% automation is impossible.
1.  **Confidence Scores:** The model outputs a confidence score (0-1) for every field.
2.  **Thresholding:**
    *   *Score > 0.95:* Automate. Write directly to the DB.
    *   *Score < 0.95:* Send to a human verification UI.
3.  **Active Learning:** The human's corrections are logged and used as new training data to fine-tune the model monthly, creating a flywheel of improvement.

## 4. Scalability and Cost
*   **Batching:** Most documents don't need sub-second extraction. Process them in batches via an asynchronous worker queue.
*   **Hybrid Model:** Use a cheap LayoutLM model for common, standard forms and only call the expensive GPT-4o for complex, multi-page, or handwritten documents.

## Interview Tip: "Spatial Context"
If you mention **LayoutLM** or **Spatial Embeddings** in a document processing interview, you immediately signal seniority. Standard "text-only" NLP is no longer the state-of-the-art for this problem.