# Detailed Methodology: Style-Aware Cross-Encoder for Telugu Prompt Style Recovery

This document provides a comprehensive explanation of the codebase structure and the methodological decisions implemented in the `paper_code/` directory. It maps the theoretical concepts from the research paper to their practical implementation in Python.

## 1. System Overview
The system is a four-phase pipeline designed to recover and classify the style transformation between an **Original Text** and a **Changed Text**. Unlike standard text classification which looks at one sentence, this model acts as a "Style Transfer Detector," analyzing the *relationship* and *delta* between two texts to identify which of the 9 target styles was applied.

### The 4 Phases:
1.  **Domain-Adaptive Pretraining (DAPT):** Adapts a multilingual model to Telugu.
2.  **Style-Aware Contrastive Learning:** Learns a metric space where similar styles are close.
3.  **Cross-Encoder Classification:** The core classifier for (Original, Changed) pairs.
4.  **Hard Confusion Refinement:** Fine-tuning on persistent errors.

---

## 2. Phase 1: Domain-Adaptive Pretraining (DAPT)
**Script:** `scripts/train_dapt.py`

### Goal
To ground the base model (MuRIL or IndicBERT) in the linguistic patterns of the Telugu language before teaching it stylistic nuances.

### Implementation Details
*   **Model:** Uses `AutoModelForMaskedLM`.
*   **Objective:** Masked Language Modeling (MLM). The model randomly masks 15% of tokens in the Telugu corpus and tries to predict them.
*   **Input:** Raw Telugu text (News, Literature, Social Media).
*   **Outcome:** A saved model checkpoint (encoder) that understands Telugu syntax and morphology better than the generic base model.

---

## 3. Phase 2: Style-Aware Contrastive Learning
**Script:** `scripts/train_contrastive.py`

### Goal
To learn a high-quality embedding space where text representations are clustered not just by exact label, but by **stylistic similarity**.

### Key Component: Style Similarity Graph
**File:** `scripts/data/style_graph.py`
The project defines a weighted graph of 9 styles.
*   **Concept:** Some styles are naturally compatible (e.g., *Inspirational* & *Motivational*), while others are distinct.
*   **Matrix:** A 9x9 compatibility matrix is pre-computed where:
    *   Diagonal (Same Style) = 1.0
    *   Compatible Styles (e.g., Informal ↔ Humorous) = 0.4 (Adjustable `weak_positive_weight`)
    *   Distant Styles = 0.0

### Loss Function: Weighted Supervised Contrastive Loss
**File:** `scripts/models/losses.py` (Class `WeightedSupConLoss`)
Instead of a binary "Positive/Negative" contrastive loss, the code implements Equation (2) from the paper:
*   **Anchor:** The current sample.
*   **Positives:** Samples with the *same* label (Weight 1.0) AND samples with *compatible* labels (Weight 0.4).
*   **Negatives:** Samples with incompatible labels.
*   **Effect:** This forces the model to pull compatible styles closer in the embedding space, creating a "smooth" manifold rather than rigid, isolated clusters.

---

## 4. Phase 3: Cross-Encoder Classification (The Core)
**Script:** `scripts/train_transfer_detector.py`

### Goal
To explicitly classify the style change. This is the main "Cross-Encoder" task.

### Input Format
Unlike NLI models that use `[CLS] Text [SEP] Hypothesis`, this model is specialized for Style Transfer:
> **Input:** `[CLS] Original Text [SEP] Changed Text [SEP]`

This allows the model to "see" the transformation—what words were added, removed, or changed—to determine the style.

### Data Augmentation: Multi-Crop
**File:** `scripts/data/dataset.py`
*   **Method:** During training, long text pairs are randomly cropped to different windows.
*   **Purpose:** Prevents overfitting to specific sentence starts/ends and forces the model to recognize style markers (e.g., "Macha", "Garu") anywhere in the text.

### Loss Function: Overlap-Aware Cross-Entropy
**File:** `scripts/models/losses.py` (Class `OverlapAwareCrossEntropyLoss`)
*   **Standard Cross-Entropy:** Forces the model to predict *only* the Gold Label (1.0) and zero for everything else.
*   **Overlap-Aware (Soft Targets):** The model is trained against a **Soft Target Distribution**.
    *   Gold Label: High probability (e.g., 0.8)
    *   Compatible Styles: Low non-zero probability (e.g., 0.1 each)
    *   Incompatible Styles: Zero probability.
*   **Benefit:** The model is not penalized for being "slightly confused" between highly similar styles (like *Formal* vs. *Authoritative*), which leads to more robust training.

---

## 5. Phase 4: Hard Confusion Refinement
**Script:** `scripts/train_refinement.py`

### Goal
To fix specific, persistent errors where the model consistently confuses two overlapping styles (e.g., *Inspirational* vs. *Motivational*) with high confidence.

### Methodology
1.  **Identification:** The script runs inference on the training set. It flags samples where the model predicts the wrong class with **>70% confidence**. These are "Hard Confusions."
2.  **Dataset Construction:** A `Subset` dataset is created containing *only* these difficult examples.
3.  **Refinement Training:** The model is fine-tuned for a few epochs on this subset.

### Loss Function: Refinement Penalty
**File:** `scripts/models/losses.py` (Class `RefinementLoss`)
*   **Formula:** `L_total = L_ce + β * Σ (p_j + p_k - 1)^2`
*   **Mechanism:** It adds a penalty term specifically for the confused pair indices $(j, k)$. It forces the model to separate the probability mass of these two classes, pushing it to make a decisive choice rather than hovering in ambiguity.

---

## Summary of Files
*   **`scripts/data/style_graph.py`**: The "Brain" of the semantic similarity. Defines which styles are close.
*   **`scripts/models/losses.py`**: The "Heart" of the learning process. Implements the custom math (Weighted Contrastive, Soft Targets, Refinement Penalty).
*   **`scripts/train_transfer_detector.py`**: The "Body". The main training loop that puts it all together.
