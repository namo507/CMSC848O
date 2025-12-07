# Long-Context Language Models — Research Papers & Course Materials

This repository contains a comprehensive collection of research papers, lecture notes, and course materials from CMSC848O: Seminar on Long-Context Language Models, Spring 2025, taught by Professor Mohit Iyyer at the University of Maryland. The course explores cutting-edge developments in large language models, with particular emphasis on extending context windows and improving long-range understanding.

## Course Overview

CMSC848O is an advanced graduate seminar focusing on the latest research in long-context language models — covering foundational architectures, attention mechanisms, scaling laws, efficient training methods, and novel approaches to handling extended sequences. The course combines reading influential papers, in-class discussions, hands-on assignments, and student presentations.

**Course Website:** https://www.cs.umd.edu/~miyyer/cmsc848o/

## Repository Contents

### Foundational Papers

#### **Attention Is All You Need** — Vaswani et al. (2017)
[`1706.03762v7-2.pdf`]
The landmark paper introducing the Transformer architecture, proposing a model based solely on attention mechanisms without recurrence or convolutions. This work revolutionized natural language processing by enabling parallel processing and capturing long-range dependencies more effectively. The multi-head self-attention mechanism allows models to focus on different representation subspaces simultaneously, fundamentally changing how we approach sequence modeling tasks.

#### **A Neural Probabilistic Language Model** — Bengio et al. (NIPS 2000)
[`NIPS-2000-a-neural-probabilistic-language-model-Paper-2.pdf`]
A seminal work that introduced neural language modeling by learning distributed representations for words and using neural networks to predict the next word in a sequence. This paper laid the groundwork for modern language models by demonstrating that neural networks could learn meaningful word embeddings and capture linguistic regularities far better than traditional n-gram approaches.

#### **Scaling Laws for Neural Language Models** — Kaplan et al. (2020)
[`2001.08361v1.pdf`]
This influential research established empirical scaling laws showing that language model performance follows predictable power-law relationships with model size, dataset size, and compute budget. The findings revealed that larger models are significantly more sample-efficient, fundamentally shaping how the field approaches model development — suggesting training very large models on modest amounts of data and stopping before full convergence.

### Efficient Training & Fine-Tuning

#### **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al. (2021)
[`2105.13626v3.pdf`]
LoRA introduces an efficient fine-tuning technique that freezes pre-trained model weights and injects trainable low-rank decomposition matrices into each Transformer layer. This approach reduces trainable parameters by up to 10,000 times and GPU memory requirements by 3 times compared to full fine-tuning, while maintaining comparable performance. LoRA enables efficient task-specific adaptation without additional inference latency.

#### **LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models** — Chen et al. (2023)
[`2309.16575v2.pdf`]
Building on LoRA, this paper presents an efficient method to extend context windows of pre-trained LLMs with limited computational cost. LongLoRA uses shifted sparse attention (S2-Attn) during training while maintaining dense attention at inference, enabling fine-tuning of LLaMA2 7B to 100k tokens or LLaMA2 70B to 32k tokens on a single 8×A100 machine.

#### **Training Language Models to Follow Instructions with Human Feedback** — Ouyang et al. (OpenAI, 2022)
[`Training_language_models_to_follow_instructions_with_human_feedback-3.pdf`]
The InstructGPT paper demonstrating how reinforcement learning from human feedback (RLHF) can align language models with human preferences and instructions. Despite having 100x fewer parameters, the 1.3B InstructGPT model outperformed GPT-3 175B on human evaluations, showing improvements in truthfulness and reductions in toxic outputs — establishing RLHF as a critical component of modern LLM development.

### Position Encoding & Context Extension

#### **NoPE: No Position Encoding** 
[`NoPE.pdf`, `position_notes.pdf`]
Research exploring alternatives to traditional positional encodings in Transformers, investigating whether models can learn positional information implicitly through data and architectural design rather than explicit encoding schemes.

### Model Architecture & Efficiency

#### **Memory-Efficient Attention Mechanisms** — Various Authors (2023-2024)
[`2104.09864v5.pdf`, `2109.01652v5-2.pdf`, `2112.04426v3.pdf`]
Collection of papers exploring efficient attention mechanisms that reduce memory consumption and computational costs while maintaining model quality — critical for extending context windows beyond traditional limits.

#### **Qwen2.5 1M Context Technical Report**
[`Qwen2_5_1M_Technical_Report.pdf`]
Technical documentation of the Qwen2.5 model series, showcasing advances in extending language model context windows to 1 million tokens while maintaining performance and efficiency.

### Tokenization & Preprocessing

#### **Tokenization Methods**
[`tokenization.pdf`]
Comprehensive notes on tokenization strategies for language models, covering byte-pair encoding (BPE), WordPiece, SentencePiece, and their impact on model performance and vocabulary efficiency.

### Advanced Training Techniques

#### **Pre-training and Post-training Methods**
[`preposttrain_notes.pdf`]
Lecture notes covering the full training pipeline for large language models — from initial pre-training on massive corpora to fine-tuning, instruction tuning, and alignment with human preferences through RLHF.

#### **Scaling and Reinforcement Learning**
[`scaling_and_rl_notes.pdf`, `scaling_laws.pdf`]
Materials discussing scaling behaviors in neural language models and the application of reinforcement learning techniques to improve model capabilities and alignment.

### Recent Advances (2024-2025)

#### **State-of-the-Art Research Papers**

[`1508.07909v5.pdf`] — Neural machine translation advances  
[`1911.05507v1.pdf`] — Language model architectures  
[`2024.tacl-1.9.pdf`] — TACL 2024 publication  
[`2203.15556v1.pdf`, `2203.15556v1-2.pdf`] — Contextualized representations  
[`2205.14135v2.pdf`] — Efficient training methods  
[`2305.13245v3.pdf`] — Long-context modeling  
[`2305.19466v2.pdf`] — Attention mechanisms  
[`2306.05685v4-2.pdf`] — Model optimization  
[`2306.14048v3.pdf`] — Inference efficiency  
[`2310.01889v4.pdf`] — Novel architectures  
[`2312.00752v2.pdf`] — Training innovations  
[`2402.10171v1.pdf`] — Context extension methods  
[`2402.17762v2.pdf`] — Performance improvements  
[`2402.19427v1.pdf`] — Efficiency research  
[`2404.06654v3-2.pdf`] — Memory optimization  
[`2404.07143v2.pdf`] — Inference speed  
[`2404.11912v3.pdf`] — Model compression  
[`2406.16264v3.pdf`] — Training stability  
[`2407.16833v2.pdf`, `2407.16833v2-2.pdf`] — Latest architectural innovations  
[`2408.10188v6.pdf`] — Recent efficiency gains  
[`2409.12961v4.pdf`] — September 2024 advances  
[`2409.19151v2.pdf`] — Context handling improvements  
[`2410.02603v2.pdf`] — October 2024 research  
[`2410.02660v4.pdf`] — Training optimizations  
[`2410.23771v5.pdf`] — Attention mechanism refinements  
[`2412.01769v1.pdf`] — December 2024 findings  
[`2501.05414v3.pdf`] — January 2025 publications  
[`2501.12948v1.pdf`] — Scaling investigations  
[`2501.19393v3.pdf`] — Recent breakthroughs  
[`2502.05252v1-2.pdf`] — February 2025 research  
[`2503.22828v2.pdf`, `2503.22832v2.pdf`] — March 2025 developments  
[`2504.07128v2.pdf`] — April 2025 innovations

### Course Materials

#### **Lecture Notes**

[`00-intro.pdf`] — Course introduction and overview  
[`ngram_notes.pdf`] — N-gram language models  
[`nlm_notes.pdf`] — Neural language model fundamentals  
[`attention_notes.pdf`] — Attention mechanisms explained  
[`transformer_notes.pdf`] — Transformer architecture details

#### **Supplementary Papers**

[`3.pdf`, `7.pdf`] — Additional course readings

#### **Programming Assignments**

[`CMSC848O_HW1_S25.ipynb`] — Homework 1: Implementing language models  
[`intro_to_latex.ipynb`] — LaTeX tutorial for academic writing

## Learning Objectives

Through this course material, students gain deep understanding of:

- The evolution of language models from n-grams to neural networks to Transformers
- Attention mechanisms and their role in capturing long-range dependencies
- Scaling laws and their implications for model development
- Efficient training techniques including LoRA and low-rank adaptation
- Methods for extending context windows in large language models
- Reinforcement learning from human feedback (RLHF) for alignment
- Position encoding schemes and alternatives
- Memory-efficient architectures for long-context processing
- Tokenization strategies and their impact on model performance
- Current research directions in long-context language modeling

## Academic Context

**Institution:** University of Maryland, College Park  
**Department:** Computer Science  
**Course:** CMSC848O — Seminar on Long-Context Language Models  
**Semester:** Spring 2025  
**Instructor:** Professor Mohit Iyyer

## Usage & Citation

These materials are intended for educational and research purposes. When referencing papers from this collection, please cite the original publications. For course-specific materials, acknowledge:

> CMSC848O: Seminar on Long-Context Language Models, University of Maryland, Spring 2025

## Key Research Areas

- **Transformer Architectures** — Self-attention, multi-head attention, positional encoding
- **Scaling Laws** — Understanding performance as a function of model size, data, and compute
- **Efficient Fine-Tuning** — Parameter-efficient methods like LoRA for task adaptation
- **Context Extension** — Techniques to extend effective context windows beyond training limits
- **Alignment & Safety** — RLHF and instruction tuning for human-aligned behavior
- **Memory Efficiency** — Novel attention mechanisms reducing computational overhead
- **Long-Range Dependencies** — Modeling relationships across extended sequences

## Repository Structure

This repository organizes materials by type — foundational papers, recent research publications, lecture notes, and assignments — providing a comprehensive resource for understanding the current state and future directions of long-context language modeling research.

---

*Note: This repository represents coursework and reading materials. All papers retain their original authorship and copyright. ArXiv IDs and publication venues are preserved in filenames for easy reference to original sources.*
