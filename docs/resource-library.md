# Local resource library

The local `AI-pdf-books-Master/` folder contains 71 PDFs. They are optional references, not a reading quota. Start with the relevant subtopic on the curriculum page, then use one carefully chosen book section when it helps deepen a learning unit.

The PDFs stay outside `docs/` so the static site does not accidentally publish or redistribute them. Open them from the local folder using the exact paths shown below. Copyright and license terms belong to the individual authors and publishers.

## Recommended spine

These references roughly follow the suggested topic dependencies. Use only the sections that support a learning unit you are studying; reading every book or following a fixed order is not required.

| Topic area | Primary book | Use it for |
|---|---|---|
| Python and data | `Data Science books/[O'Reilly] Python Data Science Handbook - Jake VanderPlas.pdf` | NumPy, pandas, visualization, and a gentle reference while completing the Foundations topic. |
| Data analysis | `Data Science books/[O'Reilly] Python for Data Analysis - data wrangling with Pandas NumPy and IPython.pdf` | Practical data loading, cleaning, grouping, joining, and time series. |
| Statistics | `Data Science books/[O'Reilly] Practical Statistics for Data Scientists.pdf` | Sampling, distributions, experiments, regression intuition, and uncertainty. |
| First ML models | `Machine Learning books/[O'Reilly] Introduction to Machine Learning with Python.pdf` | A gentle bridge from data preparation to supervised and unsupervised learning. |
| Broad ML reference | `Machine Learning books/[O'Reilly] Hands On Machine Learning with Scikit Learn and TensorFlow.pdf` | A large practical reference after the first applied ML subtopics; some APIs are older. |
| Deep learning | `Machine Learning books/[O'Reilly] Fundamentals of Deep Learning.pdf` | Neurons, layers, activations, losses, gradient descent, and training behavior. |
| LLM internals | `LLM and NLP books/[Manning] Build a Large Language Model (From Scratch).pdf` | Tokenization, attention, GPT-style models, pretraining, and fine-tuning by implementation. |
| Transformer practice | `LLM and NLP books/[O'Reilly] Natural Language Processing with Transformers.pdf` | Applying transformer models, text classification, generation, efficiency, and low-label settings. |
| Foundation-model engineering | `AI books/[O'Reilly] AI Engineering_Building Applications with Foundation Models_Chip Huyen.pdf` | Model selection, prompting, RAG, fine-tuning, data, evaluation, and application architecture. |
| Production systems | `AI Architecture books/[O'Reilly] Designing Machine Learning Systems.pdf` | Data distribution, training/serving, feedback loops, monitoring, and system trade-offs. |
| MLOps | `LLMOps and MLOps books/[O'Reilly] Practical MLOps_Operationalizing Machine Learning Models.pdf` | Packaging, deployment, monitoring, and operational lifecycle. |
| Local/small models | `LLM and NLP books/[Manning] Domain-Specific Small Language Models Efficient AI for local deployment.pdf` | ONNX, quantization, profiling, inference, and small-model deployment. |

## Mapped to the curriculum

### Foundations and classical machine learning

Use these alongside the corresponding subtopic when they are useful:

- `AI books/[O'Reilly] Al and Machine Learning for Coders.pdf` — useful motivation and practical coding bridge.
- `Data Science books/[O'Reilly] Data Science from Scratch.pdf` — algorithms and statistics implemented with plain Python.
- `Data Science books/[O'Reilly] Python Data Science Handbook - Jake VanderPlas.pdf` — NumPy, pandas, and plotting reference.
- `Data Science books/[O'Reilly] Python for Data Analysis - data wrangling with Pandas NumPy and IPython.pdf` — strongest practical data-wrangling reference in the folder.
- `Data Science books/[O'Reilly] Practical Statistics for Data Scientists.pdf` — best fit for the statistics part of the beginner curriculum.
- `Data Science books/The Probability and Statistics Cookbook.pdf` — compact formula reference; use it after intuition has been taught.
- `Data Science books/Data Mining Concepts and Techniques 3rd Edition.pdf` — broader data-mining algorithms and concepts; optional depth.
- `Data Science books/Exploratory Data Analysis with R.pdf` — useful EDA thinking, but examples use R rather than Python.
- `Data Science books/Storytelling with data cole nussbaumer knaflic.pdf` — chart choice, visual hierarchy, and communicating findings.
- `Machine Learning books/[O'Reilly] Introduction to Machine Learning with Python.pdf` — recommended first applied ML book.
- `Machine Learning books/[O'Reilly] Hands On Machine Learning with Scikit Learn and TensorFlow.pdf` — broad hands-on reference; verify older library APIs before copying code.
- `Machine Learning books/ML_basic_VN.pdf` and `Machine Learning books/Machine_learning_VN.pdf` — optional Vietnamese-language reinforcement if that helps comprehension.

### Deep learning

- `Machine Learning books/[O'Reilly] Fundamentals of Deep Learning.pdf` — use for the first neural-network implementation and training intuition.
- `LLM and NLP books/[Springer] Deep Learning in Natural Language Processing.pdf` — optional NLP-specific neural-network depth.
- `Paper/contrastive-learning-ECCV-tutorial_ting_Chen.pdf` — later vision/representation-learning specialization.
- `Paper/GAN Ian Goodfellow.pdf` — historical and conceptual introduction to generative adversarial networks.

### Generative AI, LLMs, and NLP

- `AI books/[O'Reilly] AI Engineering_Building Applications with Foundation Models_Chip Huyen.pdf` — the main application-engineering reference.
- `LLM and NLP books/[Manning] Build a Large Language Model (From Scratch).pdf` — best for understanding internals by building a small GPT-like model.
- `LLM and NLP books/[O'Reilly] Natural Language Processing with Transformers.pdf` — best applied transformer reference in the folder.
- `LLM and NLP books/[Stanford] Speech and Language Processing.pdf` — broad academic NLP reference; use selected chapters only.
- `LLM and NLP books/[MIT] Foundations of Statistical Natural Language Processing.pdf` — classical NLP foundations; important historically, but not a beginner-first book.
- `LLM and NLP books/[O'Reilly] Practical Natural Language Processing.pdf` — production-oriented NLP tasks and workflows.
- `LLM and NLP books/[O'Reilly] Applied Natural Language Processing in the Enterprise.pdf` — enterprise use cases and deployment concerns.
- `LLM and NLP books/[O'Reilly] Natural-Language-Annotation-for-Machine-Learning.pdf` — dataset and labeling quality.
- `LLM and NLP books/[O'Reilly] Prompt Engineering for LLMs.pdf` — prompting patterns; treat vendor-specific advice as version-sensitive.
- `LLM and NLP books/Prompt Engineering Guide VN.pdf` — optional Vietnamese-language prompt reference.
- `LLM and NLP books/[O'Reilly] Designing Large Language Model Applications (2023).pdf` — early application patterns; compare with newer material.

### AI engineering, MLOps, and agents

- `AI Architecture books/[O'Reilly] Designing Machine Learning Systems.pdf` — system-design backbone.
- `AI Architecture books/[O'Reilly] Machine Learning Design Patterns_Solutions to Common Challenges in Data Preparation Model Building and MLOps.pdf` — reusable patterns for production ML.
- `LLMOps and MLOps books/[O'Reilly] Practical MLOps_Operationalizing Machine Learning Models.pdf` — operational lifecycle and deployment.
- `LLMOps and MLOps books/[O'Reilly] What is LLMOps_Large Language Models in Production.pdf` — short overview for entering LLM operations.
- `LLMOps and MLOps books/[O'Reilly] Building Generative AI Services with FastAPI.pdf` — API-serving project reference.
- `LLMOps and MLOps books/[O'Reilly] FastAPI.pdf` — web framework reference; do not study before you need to expose a model service.
- `LLMOps and MLOps books/Generative AI with LangChain (2024).pdf` — framework examples; learn the underlying patterns before depending on a framework.
- `AI Architecture books/[O'Reilly] Building Applications with AI Agents Designing and Implementing Multiagent Systems.pdf` — tools, orchestration, memory, evaluation, monitoring, and safety.
- `AI Architecture books/[O'Reilly] Generative AI Design Patterns.pdf` — design-pattern reference for production generative systems.
- `AI Architecture books/[O'Reilly] Architecting Data and Machine Learning Platforms.pdf` — platform and data-infrastructure depth.
- `AI Architecture books/[O'Reilly] Generative AI on Kubernetes Operationalizing Large Language Models.pdf` — optional infrastructure specialization after basic serving and monitoring.
- `LLMOps and MLOps books/[O'Reilly] Python for DevOps_Learn Ruthlessly Effective Automation.pdf` — useful automation background, not an AI textbook.
- `Data Science books/[O'Reilly] Database Internals.pdf` — optional systems depth for storage, indexes, and retrieval infrastructure.
- `Data Science books/Storytelling with data cole nussbaumer knaflic.pdf` — useful for evaluation reports and model communication.

### On-device AI and specializations

The most relevant local reference is:

- `LLM and NLP books/[Manning] Domain-Specific Small Language Models Efficient AI for local deployment.pdf` — ONNX, quantization, profiling, deployment, and local inference.

However, the folder does **not** contain a dedicated Android on-device reference for LiteRT/TensorFlow Lite, LiteRT-LM, MediaPipe, Android NNAPI/GPU/NPU delegates, or Core ML. The curriculum will therefore pair this book with current official documentation and real Android benchmark projects:

- [Android AI](https://developer.android.com/ai)
- [Google AI Edge](https://ai.google.dev/edge)
- [LiteRT](https://ai.google.dev/edge/litert)
- [MediaPipe Solutions](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [ONNX Runtime mobile](https://onnxruntime.ai/docs/get-started/with-android.html)
- [Apple Core ML](https://developer.apple.com/documentation/coreml)

The platform APIs change faster than books. When using version-sensitive Android instructions, record the date, tested device, and runtime version on the relevant topic page.

### Research and career

- `Paper/HowtoReadPaper.pdf` — use from the first paper-reading exercise; its three-pass method is practical.
- `Paper/the_craft_of_reseach.pdf` — writing, claims, evidence, and research structure.
- `Paper/Paper_Writing_Part-1.pdf` and `Paper/Paper_Writing_Part-2.pdf` — short writing references.
- `Paper/stanford-cheatsheet-transformers-large-language-models.pdf` — quick transformer revision sheet, not a first explanation.
- `Paper/LLM_codegen_lecture.pdf` — code-generation overview.
- `Paper/2305.20050v1.pdf` — process supervision and step-by-step verification.
- `Paper/2306.05685.pdf` — LLM-as-a-judge evaluation and its limitations.
- `Paper/2408.08921v1.pdf` — graph retrieval-augmented generation survey.
- `Paper/2408.08435v1.pdf` — automated design of agentic systems.
- `Paper/2408.12599v1.pdf` — controllable text generation survey.
- `Paper/2406.01574v6.pdf` and `Paper/2311.12022v1.pdf` — challenging LLM evaluation benchmarks.
- `Paper/2107.03374v2.pdf` — evaluating code-generation models.
- `Paper/1910.13461v1.pdf`, `Paper/2001.08210v2.pdf`, and `Paper/2111.09543v4.pdf` — research-paper reading practice for sequence-to-sequence and pretrained language models.
- `Paper/2401.05566.pdf` — advanced safety and deceptive-behavior research; not beginner material.
- `Paper/2501.14249v5.pdf` — advanced benchmark design and evaluation.
- `Paper/glaze-usenix23.pdf` — security, privacy, and adversarial protection for generative image systems.

## Books to postpone

Do not start with `AI Engineering`, `Practical MLOps`, `Building Applications with AI Agents`, `Database Internals`, Kubernetes, or research papers. They are valuable after the basic vocabulary, Python, data, ML, and evaluation foundations are in place. Starting there creates the illusion of progress while leaving the core concepts unstable.

## How the curriculum will use a book

For a subtopic you are studying, choose resources by purpose:

1. **Primary reading:** one short section that directly supports a learning unit.
2. **Reference reading:** a deeper chapter for later review.
3. **Practice source:** a dataset, example, or project to reproduce.
4. **Caution:** old APIs, assumptions, missing mobile context, or advanced prerequisites.

The topic page and its learning units define the curriculum. Books deepen a unit; they do not replace its beginner explanation, worked examples, exercises, practice project, or mastery checkpoint.

