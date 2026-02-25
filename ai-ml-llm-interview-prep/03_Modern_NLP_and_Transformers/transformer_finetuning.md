# Transformer Fine-Tuning Patterns

Fine-tuning a Transformer is how you adapt a foundational model to your specific business logic. Understanding the different *types* of fine-tuning is critical for system design.

---

## 1. Sequence Classification
*   **Task:** The input is a sequence of text; the output is a single label (e.g., Sentiment Analysis, Spam Detection, Topic Categorization).
*   **Architecture (Encoder-based like BERT):** You take the output vector of the special `[CLS]` token from the final layer. You attach a standard Linear (Dense) layer on top of it, followed by a Softmax or Sigmoid activation.
*   **Training:** You freeze the bottom layers of the Transformer (to retain general language knowledge) and train the top layers plus your new classification head using Cross-Entropy Loss.

## 2. Token Classification
*   **Task:** The input is a sequence; the output is a label *for every single word* (e.g., Named Entity Recognition (NER), Part-of-Speech tagging).
*   **Architecture:** Instead of using the `[CLS]` token, you attach a Linear layer to the output vector of *every single token* in the sequence. 
*   **Warning:** Because of sub-word tokenization, the word "Washington" might be split into `["Wash", "##ington"]`. You must carefully write code to align your true labels with the sub-words (usually assigning the label to the first sub-word and ignoring the rest during loss calculation).

## 3. Extractive Question Answering
*   **Task:** Given a Context paragraph and a Question, highlight the exact span of text in the Context that contains the answer. (Standard SQuAD dataset task).
*   **Architecture:** You feed `[Question Tokens] + [SEP] + [Context Tokens]`. You attach two separate Linear heads to every token in the Context.
    *   Head 1 predicts the probability that a token is the **Start** of the answer.
    *   Head 2 predicts the probability that a token is the **End** of the answer.

## 4. Generative Fine-Tuning (Instruction Tuning)
This is how modern LLMs (Llama, GPT) are fine-tuned.
*   **Task:** The model must generate novel text based on a prompt.
*   **Architecture (Decoder-Only):** No classification heads are added. We keep the standard Next-Token Prediction language modeling head.
*   **The Trick (Masked Loss):** We format the training data as: `User: What is 2+2? Assistant: 4`. We run the forward pass on the whole string. However, during backpropagation, we **mask out the loss for the User's prompt**. The model is only penalized if it fails to predict the tokens belonging to the *Assistant's response*. This teaches the model the "behavior" of answering questions without destroying its ability to read prompts.

## 5. Parameter-Efficient Fine-Tuning (PEFT)
Full fine-tuning updates 100% of the model's weights. For a 70B parameter model, this requires multi-node GPU clusters.

### LoRA (Low-Rank Adaptation)
The industry standard.
*   Instead of updating the massive $W$ matrix directly, LoRA freezes $W$ and injects two tiny matrices ($A$ and $B$) alongside it.
*   We only calculate gradients and update $A$ and $B$.
*   We drop trainable parameters by 99%, allowing us to fine-tune massive models on a single GPU. (See `llm_finetuning.md` for deep dive).

### Prompt Tuning / Prefix Tuning
*   Instead of touching the model weights at all, we attach a sequence of "virtual, trainable tokens" to the beginning of the prompt. Backpropagation updates the embeddings of these virtual tokens (not the model weights) to act as an optimized, mathematical "system prompt" to guide the model's behavior. Less effective than LoRA, but extremely cheap.