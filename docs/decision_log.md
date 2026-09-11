# Decision Log

This document records the non-obvious technical and product decisions made during the development of the AI Customer Support Agent.

### 1. Brand Selection
**Decision**: Selected `AppleSupport` as the brand for evaluation.
**Why**: AppleSupport is the second most active brand in the dataset (106,860 agent responses). Their support interactions cover a diverse but well-defined set of issues (hardware, software, services, billing) which allows for a clean intent taxonomy. Their responses are also highly structured, making it an ideal candidate for evaluating grounded reply generation.
**Alternatives considered**: AmazonHelp, Uber_Support.
**Trade-off**: AmazonHelp has a slightly larger volume, but AppleSupport's technical troubleshooting threads offer a better testbed for the LLM's ability to stay grounded and not hallucinate tech support steps.

### 2. Intent Taxonomy Granularity
**Decision**: Defined 10 specific intents (e.g., `software_update_issue`, `battery_issue`) rather than a massive 50+ intent taxonomy.
**Why**: A small, well-defined set of intents reduces classifier confusion and keeps the evaluation manageable, while covering 80%+ of the core support volume for the brand.
**Alternatives considered**: Unsupervised clustering to auto-generate 100+ micro-intents.
**Trade-off**: Lost granularity for extremely rare issues, but gained robust baseline performance and clear human-interpretable categories.

### 3. Separation of Golden Set
**Decision**: The 200-example golden set is strictly excluded from all training and retrieval corpora.
**Why**: To prevent evaluation leakage. If the agent retrieves the exact historical answer from the golden set, it artificially inflates groundedness metrics without proving generalization.
**Alternatives considered**: Using cross-validation without a holdout set.
**Trade-off**: Reduced the total pool of training/retrieval data by 200 high-quality examples, but ensures honest headline metrics.

### 4. Simple Baseline Architecture
**Decision**: Used TF-IDF + Logistic Regression as the simple baseline for intent classification.
**Why**: It is extremely fast to train, highly interpretable, and establishes a strong "floor" metric (~76% accuracy) that any complex neural network must beat to justify its computational cost.
**Alternatives considered**: Naive Bayes, Random Forest.
**Trade-off**: TF-IDF struggles with semantic meaning (e.g., "power draining" vs "battery dying"), relying purely on exact word matches.

### 5. Dense Embedding Retrieval
**Decision**: Used `sentence-transformers` (`all-MiniLM-L6-v2`) for retrieval instead of lexical matching (BM25).
**Why**: Customer support queries often use vastly different vocabulary to describe the same issue. Dense embeddings capture semantic similarity better than keyword matching.
**Alternatives considered**: BM25, TF-IDF cosine similarity.
**Trade-off**: Higher computational cost to index and query compared to BM25.

### 6. Indexing Technology
**Decision**: Selected `FAISS` (IndexFlatIP) for the retrieval index.
**Why**: FAISS is optimized for dense vector similarity search. Using inner product with normalized vectors provides fast cosine similarity ranking.
**Alternatives considered**: Scikit-learn's NearestNeighbors, vector databases like Pinecone.
**Trade-off**: FAISS requires loading the index into RAM, which is fine for 30k vectors but might not scale to 3M vectors without quantization (IndexIVFPQ).

### 7. Historical Grounding
**Decision**: Prompted the LLM using actual historical `AppleSupport` agent replies as evidence, rather than relying on the LLM's internal knowledge.
**Why**: To prevent hallucinations. The assignment strictly forbids inventing policies or refunds. Grounding forces the LLM to mimic the brand's established tone and resolution steps.
**Alternatives considered**: Zero-shot generation.
**Trade-off**: The LLM's reply quality is entirely bottlenecked by the quality of the retrieved historical case.

### 8. Escalation Confidence Threshold
**Decision**: Set the escalation confidence threshold at 0.5 (for a 10-class problem).
**Why**: If the classifier is less than 50% sure of the intent, the risk of providing an irrelevant automated technical response is too high.
**Alternatives considered**: 0.8 strict threshold, 0.3 loose threshold.
**Trade-off**: A 0.5 threshold yields a conservative bot with a low auto-handle rate (32%), but prioritizes customer safety and brand reputation.

### 9. Ambiguous Intent Escalation
**Decision**: Hardcoded `hardware_device_issue` and `other_complaint` to always `ESCALATE`.
**Why**: Hardware issues often require physical repair or complex diagnostics that a simple text bot cannot resolve. "Other" complaints are too vague to safely auto-handle.
**Alternatives considered**: Let the LLM try to handle them anyway.
**Trade-off**: Increases human agent load, but guarantees that complex or angry customers aren't stuck in an unhelpful bot loop.

### 10. Avoidance of External APIs
**Decision**: Completely avoided paid APIs like OpenAI (GPT-4) or Anthropic.
**Why**: To ensure the project remains fully open-source, reproducible, and adheres strictly to the assignment's implicit constraints regarding external dependencies.
**Alternatives considered**: GPT-4-mini for easy LLM-as-a-judge.
**Trade-off**: Local models (Llama 3.1 8B) require significant user hardware (RAM/GPU) to run effectively.

### 11. Local Inference Engine
**Decision**: Selected `Ollama` for running the local generative LLM.
**Why**: Ollama provides a simple, Docker-like experience for managing local models with a reliable REST API that mimics standard LLM endpoints.
**Alternatives considered**: `llama.cpp` directly, `vLLM`, HuggingFace `transformers`.
**Trade-off**: Requires the evaluator to install an external binary (Ollama) rather than simply running a pip package.

### 12. Deterministic Fallback Mechanism
**Decision**: Implemented a graceful template-based fallback if Ollama is unreachable.
**Why**: If the evaluator's machine lacks the hardware to run an LLM, the evaluation pipeline would crash. The fallback ensures the pipeline can run end-to-end to prove the system architecture works, even if the generative node is mocked.
**Alternatives considered**: Crashing the pipeline with a "Hardware requirement not met" error.
**Trade-off**: Fallback replies lack empathy and synthesis, artificially lowering the qualitative score of the bot during hardware-constrained evaluations.

### 13. Fallback Sanitization
**Decision**: Applied regex to strip Twitter `@mentions` and URLs from the historical fallback replies.
**Why**: Presenting another customer's username or an irrelevant link in a fallback reply is a severe privacy/UX violation.
**Alternatives considered**: Returning the raw historical text.
**Trade-off**: The fallback reply might lose context if the URL contained the actual solution.

### 14. LLM Judge Validation Requirement
**Decision**: Mandated manual human review of 50 judge examples (Human vs LLM agreement).
**Why**: LLMs are known to have biases (e.g., preferring longer responses). We cannot trust the LLM judge's 1-5 score without proving it correlates with human judgment on this specific brand's dataset.
**Alternatives considered**: Blindly trusting the LLM judge scores.
**Trade-off**: Adds manual labor to the evaluation phase but ensures the metrics are honest.
