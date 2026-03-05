# Prompt Recovery for LLM in Telugu (DravidianLangTech @ ACL 2026)

**Repository Name:** `Vakya-Telugu-Prompt-Style-Recovery`

## Overview

The **Telugu Prompt-Style Recovery task** focuses on automatically identifying the communicative style expressed in a Telugu transcript excerpt. Given an input text, systems must classify it into one of **nine stylistic categories**:

*   Formal
*   Informal
*   Optimistic
*   Pessimistic
*   Humorous
*   Serious
*   Inspiring
*   Authoritative
*   Persuasive

Each category represents a distinct tonal or pragmatic intention of the speaker or writer.

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
