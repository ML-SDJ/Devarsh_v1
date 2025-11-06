"""NLP pipeline for MedExplain AI – Multimodal Clinical Insight Assistant.

This module loads summarization datasets, fine-tunes a T5-small model using the
Hugging Face Transformers Trainer API, evaluates with ROUGE, and provides an
inference helper for generating summaries.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)
from transformers.trainer_utils import EvalPrediction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "artifacts" / "nlp"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Placeholder for the fine-tuned checkpoint.
FINETUNED_MODEL_PATH = MODEL_DIR / "t5_small_finetuned"

DEFAULT_MODEL_NAME = "t5-small"
DEFAULT_DATASET = "scientific_papers"
DEFAULT_DATASET_CONFIG = "pubmed"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 150


def load_summarization_dataset(
    dataset_name: str = DEFAULT_DATASET,
    dataset_config: Optional[str] = DEFAULT_DATASET_CONFIG,
    split: str = "train",
    sample_size: Optional[int] = None,
):
    """Load a summarization dataset split from Hugging Face Datasets."""
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if sample_size is not None:
        dataset = dataset.select(range(sample_size))
    return dataset


def preprocess_function(examples: Dict[str, List[str]], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """Tokenize inputs and targets for T5."""
    inputs = examples["article"] if "article" in examples else examples.get("text", [])
    targets = examples["abstract"] if "abstract" in examples else examples.get("summary", [])
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        padding="max_length",
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=MAX_TARGET_LENGTH,
            padding="max_length",
            truncation=True,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_model_and_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    """Load a pretrained T5-small model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(DEVICE)
    return model, tokenizer


def compute_metrics(pred: EvalPrediction, tokenizer: AutoTokenizer):
    """Compute ROUGE scores for evaluation."""
    from datasets import load_metric

    rouge = load_metric("rouge")
    decoded_preds = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    prediction_lens = [len(pred.split()) for pred in decoded_preds]
    result["gen_len"] = sum(prediction_lens) / len(prediction_lens)
    return result


def fine_tune(
    output_dir: Path = FINETUNED_MODEL_PATH,
    dataset_name: str = DEFAULT_DATASET,
    dataset_config: Optional[str] = DEFAULT_DATASET_CONFIG,
    num_train_samples: Optional[int] = 200,
    num_eval_samples: Optional[int] = 50,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    learning_rate: float = 5e-5,
    warmup_steps: int = 0,
) -> Seq2SeqTrainer:
    """Fine-tune T5-small using the Seq2SeqTrainer API."""
    model, tokenizer = build_model_and_tokenizer()
    train_dataset = load_summarization_dataset(dataset_name, dataset_config, "train", num_train_samples)
    eval_dataset = load_summarization_dataset(dataset_name, dataset_config, "validation", num_eval_samples)

    tokenized_train = train_dataset.map(lambda batch: preprocess_function(batch, tokenizer), batched=True)
    tokenized_eval = eval_dataset.map(lambda batch: preprocess_function(batch, tokenizer), batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        logging_dir=str(output_dir / "logs"),
        warmup_steps=warmup_steps,
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")
    return trainer


def load_finetuned_model(path: Path = FINETUNED_MODEL_PATH):
    """Load a fine-tuned T5-small model from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Fine-tuned model directory '{path}' not found. Run fine_tune() first or set FINETUNED_MODEL_PATH accordingly."
        )
    model = T5ForConditionalGeneration.from_pretrained(path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


def generate_summary(text: str, max_length: int = 150, num_beams: int = 4) -> str:
    """Generate a summary using the fine-tuned model."""
    try:
        model, tokenizer = load_finetuned_model()
    except FileNotFoundError:
        # Fallback to base model if fine-tuned weights are unavailable.
        print("Warning: Fine-tuned model not found. Loading base T5-small instead.")
        model, tokenizer = build_model_and_tokenizer()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=MAX_INPUT_LENGTH,
    ).to(DEVICE)

    summary_ids = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def evaluate_model(path: Path = FINETUNED_MODEL_PATH, dataset_name: str = DEFAULT_DATASET, dataset_config: Optional[str] = DEFAULT_DATASET_CONFIG, num_eval_samples: int = 100):
    """Evaluate the fine-tuned model on a validation split and compute ROUGE."""
    model, tokenizer = load_finetuned_model(path)
    eval_dataset = load_summarization_dataset(dataset_name, dataset_config, "validation", num_eval_samples)
    tokenized_eval = eval_dataset.map(lambda batch: preprocess_function(batch, tokenizer), batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(path / "eval"),
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        do_train=False,
        do_predict=True,
        evaluation_strategy="no",
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    metrics = trainer.evaluate()
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
    return metrics


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    if FINETUNED_MODEL_PATH.exists():
        print("Fine-tuned model detected. Running evaluation on a small subset...")
        evaluate_model(num_eval_samples=20)
    else:
        print("Fine-tuned model not found. Run fine_tune() to train the summarizer.")
