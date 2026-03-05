# Prompt Recovery for LLM in Telugu (DravidianLangTech @ ACL 2026)

**Repository Name:** `Vakya-Telugu-Prompt-Style-Recovery`

## Overview

The **Telugu Prompt-Style Recovery task** focuses on automatically identifying the communicative style expressed in a Telugu transcript excerpt. Given an input text, systems must classify it into one of **nine stylistic categories**.

### Label Definitions and Annotation Cues
Each of the nine style categories is characterised by distinct linguistic and pragmatic features:

*   **Formal**: Polite, structured language; professional register; complete sentences; minimal slang.
*   **Informal**: Conversational and colloquial; frequent use of contractions, slang, and emojis; second-person direct address; short phrases.
*   **Optimistic**: Positive outlook and encouragement; future-success oriented language.
*   **Pessimistic**: Negative or bleak outlook; cautionary and doubtful tone.
*   **Humorous**: Jokes, playful exaggeration, comedic timing, irony, and light-hearted metaphors.
*   **Serious**: Sober and factual tone; warnings or grave subject-matter; deliberate absence of humour.
*   **Inspiring**: Motivational language; calls-to-action; uplifting metaphors; "you can do it" framing.
*   **Authoritative**: Directive or commanding voice; expert-like claims; high certainty; imperative guidance.
*   **Persuasive**: Language intended to convince or convert, including sales-like appeals, lobbying rhetoric, and appeals to personal benefit.

### Data Statistics
The test partition used in evaluation comprises **301 instances** distributed across the nine categories. The distribution reflects moderate class imbalance, with *Serious* being the most frequent category (15.6%) and *Pessimistic* the least frequent (7.3%).

| Style | Count | % |
| :--- | :---: | :---: |
| **Authoritative** | 30 | 10.0 |
| **Formal** | 38 | 12.6 |
| **Humorous** | 31 | 10.3 |
| **Informal** | 35 | 11.6 |
| **Inspiring** | 37 | 12.3 |
| **Optimistic** | 33 | 11.0 |
| **Persuasive** | 28 | 9.3 |
| **Pessimistic** | 22 | 7.3 |
| **Serious** | 47 | 15.6 |
| **Total** | **301** | **100.0** |

### Goals & Impact
The goal is to encourage the development of models that can reliably capture subtleties of Telugu stylistics, including register variation, emotional tone, motivational framing, rhetorical patterns, and humour.

This task plays an important role in advancing style-aware natural language understanding for low-resource Indian languages, supporting applications such as:
*   Tone-consistent prompt generation
*   Style-controlled rewriting
*   Personalized conversational agents
*   Content moderation
*   Sentiment-rich dialogue systems

By providing a standardized dataset, evaluation protocol, and benchmark scores, the shared task aims to foster reproducible research, create comparative baselines, and accelerate progress in Telugu style classification within the broader multilingual NLP landscape.

---

## Paper Code: Style-Aware Cross-Encoder Implementation

This folder contains the implementation code corresponding to the methodology described in the paper.

### Directory Structure

*   `scripts/`: Training and utility scripts.
    *   `train_dapt.py`: **Phase 1** - Domain Adaptive Pretraining (MLM).
    *   `train_contrastive.py`: **Phase 2** - Style-Aware Contrastive Learning.
    *   `train_transfer_detector.py`: **Phase 3** - Cross-Encoder Classification (Original + Changed pairs).
    *   `train_refinement.py`: **Phase 4** - Hard Confusion Refinement.
*   `scripts/models/`: Model architectures and custom loss functions.
*   `scripts/data/`: Dataset loading and Style Graph definitions.
*   `RESULTS.md`: Detailed performance metrics and classification report.

### How to Run the Pipeline

#### 1. Domain-Adaptive Pretraining (Phase 1)
Adapts the base model (e.g., Muril/IndicBERT) to the Telugu corpus.
```bash
python scripts/train_dapt.py --config configs/config.yaml
```

#### 2. Contrastive Learning (Phase 2)
Trains the encoder to learn style-aware embeddings using weighted supervised contrastive loss.
```bash
python scripts/train_contrastive.py --config configs/config.yaml
```

#### 3. Classification (Phase 3)
Trains the final Cross-Encoder to classify the style change between Original and Changed text pairs.
```bash
python scripts/train_transfer_detector.py --config configs/config.yaml
```

#### 4. Hard Confusion Refinement (Phase 4)
Retrains the model on misclassified pairs using a pairwise penalty loss.
```bash
python scripts/train_refinement.py --config configs/config.yaml --model_path outputs/style_transfer_detector/best
```

### Note on "Cross-Encoder" Terminology
In this implementation, the "Cross-Encoder" refers to the `StyleTransferDetector` model (in `train_transfer_detector.py`) which cross-encodes the **Original Text** and **Changed Text** jointly to predict the style transformation. This differs from some NLI-based definitions but serves the specific classification goal of this project.
