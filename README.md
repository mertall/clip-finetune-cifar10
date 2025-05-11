# clip-finetune-cifar10
Finetuning CLIP with CIFAR-10 dataset with augmeneted image and text data.
## 📄 Notebook Walkthrough

The notebook is organized into eight main sections. Here’s what you can expect in each:

1. **Introduction & Objectives**  
   - **What you’ll learn**: Fine‑tuning CLIP, contrastive learning foundations, embedding extraction, HNSW indexing, and retrieval evaluation.  
   - **Motivation**: Why contrastive fine‑tuning and fast similarity search matter for practical applications.

2. **Setup & Installation**  
   - **Environment**: Install required libraries (PyTorch, transformers, datasets, hnswlib, etc.).  
   - **Configuration**: Set device (CPU/GPU) and define helper imports.

3. **Data Visualization & Augmentation**  
   - **CIFAR‑10 Overview**: Load and display random image samples from each class.  
   - **Augmentation Pipeline**: Define and preview torchvision transforms (random crop, flip, color jitter).

4. **Dataset & DataLoader Definitions**  
   - **Contrastive Dataset**: Custom `CIFAR10Contrastive` class returns both image and text inputs per sample.  
   - **Synonym Expansion**: Enrich text prompts with template‑based synonyms for broader concept coverage.  
   - **Collate Function**: Batching logic to pad text sequences and stack image tensors.

5. **Training & Validation (Contrastive Learning)**  
   - **Model Setup**: Instantiate CLIP models (pretrained vs. fine‑tuned).  
   - **Optimizer & Scheduler**: AdamW and learning‑rate schedules (linear or cosine).  
   - **Training Loop**: Joint image/text loss, gradient accumulation, and epoch‑by‑epoch logs.  
   - **Validation**: Zero‑shot accuracy using text‑prototype matching.

6. **Embedding Extraction & Normalization**  
   - **Projection**: Extract final image and text embeddings from the model.  
   - **Normalization**: L₂‑normalize vectors to unit length for cosine similarity.

7. **HNSW Index Construction & Querying**  
   - **Index Creation**: Build an HNSWLib index with combined embeddings.  
   - **Text → Image**: Demonstrate ANN search with multimodal RAG.  
   - **Retrieval Evaluation**: Display top‑k retrieved images for sample prompts.

8. **Advanced Evaluation & Visualization**  
   - **Recall/Precision@K**: Compute retrieval metrics against ground‑truth labels.  
   - **Dimensionality Reduction**: UMAP builds its neighborhood graph from pairwise distances, which can become unreliable in very high‑dimensional spaces. 
      By first applying PCA to reduce dimensionality and remove noisy, low‑variance directions, class separations become clearer—enabling UMAP to produce more distinct, well‑separated clusters.
   - **Image Display**: Unfuzzify tensors back to viewable PIL images.

---

## 📊 Results & Insights

- Expect a significant lift in zero‑shot accuracy after fine‑tuning (e.g., >9% vs. base CLIP).  
- Demonstrations of fast and accurate retrieval.  
- UMAP visualizations reveal clean class clusters in embedding space.

---

## 🛠️ Extending this Work

- Swap CIFAR‑10 for a custom dataset by modifying the `CIFAR10Contrastive` class.  
- Experiment with different CLIP variants or backbone architectures.  
- Embeddings text and image in the same index allowing reverse search, first attempt failed- thinking its because non descriptive text was being used.