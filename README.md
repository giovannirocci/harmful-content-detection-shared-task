# Harmful Content Detection - GermEval 2026

This repository contains the code, experiments, and submission artifacts for "The MMGs"s harmful-content detection system built for the GermEval 2026 shared task. The project fine-tunes transformer encoders for multiple moderation-related classification tasks and also includes several data augmentation strategies to improve performance under class imbalance.

## Project Overview

The main training pipeline is in [train.py](train.py), which fine-tunes a Hugging Face sequence classification model on one of four GermEval tasks:

- `c2a`: Call to Action, binary classification (`FALSE` / `TRUE`)
- `dbo`: Democratic Basic Order attack, 4-way classification
- `def`: Defamatory Offences, binary classification (`FALSE` / `TRUE`)
- `vio`: Violence Detection, 6-way classification

The default backbone is `cardiffnlp/twitter-xlm-roberta-base`, but the training script accepts any compatible Hugging Face model checkpoint or local model path.

## What Is Included

- Supervised training with stratified validation splits
- Class-balanced loss to reduce the effect of label imbalance
- Early stopping and best-checkpoint saving
- Validation reporting with macro-F1 and per-class classification reports
- Single-model and ensemble inference scripts
- Data augmentation pipelines based on backtranslation, synonym replacement, word embedding substitution, and semantic mining
- Saved model checkpoints, training summaries, and submission files under `models/`

## Repository Layout

- `train.py`: fine-tuning entry point for all GermEval tasks
- `inference.py`: single-model and ensemble inference on validation or test data
- `backtranslation_aug.py`: backtranslation-based augmentation
- `synonym_aug.py`: synonym replacement augmentation
- `wembedding_aug.py`: word-embedding-based augmentation
- `data_mining_aug.py`: augmentation by mining semantically similar external examples
- `data_analysis.py`, `test_duplicates.py`: data inspection and validation utilities
- `GermEval2026/`: task data and baseline dataset files
- `models/`: trained checkpoints, ensemble outputs, and submission CSVs
- `augmented/`: generated augmented datasets
- `Alternate approach/`: exploratory experiments and alternative baselines

## Training

Train a model for one task with:

```bash
python train.py --task def --model cardiffnlp/twitter-xlm-roberta-base --output_dir models/modernGBERT/baseline --epochs 4 --batch_size 16
```

Useful options:

- `--task`: one of `c2a`, `dbo`, `def`, `vio`
- `--augment_file`: optional semicolon-separated CSV with extra training samples
- `--bfloat`: enable bf16 mixed precision when supported
- `--cb_beta`: controls the strength of class-balanced weighting

The script writes the best checkpoint to `OUTPUT_DIR/TASK/best` and a training summary to `OUTPUT_DIR/TASK/train_results.txt`.

## Inference

Single-model inference on the test set:

```bash
python inference.py --single_model_path models/modernGBERT/baseline/def/best --output_dir models/submission/run-single
```

Ensemble inference with multiple checkpoints:

```bash
python inference.py --model_paths models/modernGBERT/baseline/def/best models/modernGBERT/embedding_aug/def/best --output_dir models/ensemble_output/baseline
```

Use `--test_mode` with ensemble inference to generate GermEval submission files from the test split.

## Augmentation Pipelines

The repository includes multiple augmentation scripts for producing additional training data:

- `backtranslation_aug.py`: generates paraphrases by translating text to English and back to German
- `synonym_aug.py`: replaces selected words with synonyms
- `wembedding_aug.py`: swaps words with nearest neighbors from pre-trained embeddings
- `data_mining_aug.py`: retrieves semantically similar examples from external corpora using sentence embeddings and FAISS

The generated files are stored under `augmented/` and can be passed to `train.py` through `--augment_file`.

## Outputs

Training and inference artifacts are organized under `models/`, including:

- task-specific checkpoints
- validation reports
- ensemble results
- submission CSV files
