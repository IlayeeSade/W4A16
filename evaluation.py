import torch
import torch.nn as nn
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def prepare_lm_datasets(tokenizer, block_size: int):
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_function(examples, tokenizer_arg=None):
        tok = tokenizer_arg or tokenizer
        return tok(examples["text"], truncation=True)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer_arg": tokenizer},
    )

    def group_texts(examples):
        keys = ["input_ids", "attention_mask"]
        concatenated = {k: sum(examples[k], []) for k in keys if k in examples}
        total = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenized_datasets.map(
        group_texts,
        batched=True,
        remove_columns=tokenized_datasets["train"].column_names,
    )


def _get_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def calculate_perplexity(
    model_to_evaluate: nn.Module,
    dataset_split: HFDataset,
    batch_size: int = 1,
) -> float:
    model_to_evaluate.eval()

    class PerplexityTorchDataset(torch.utils.data.Dataset):
        def __init__(self, hf_split):
            self.input_ids = [torch.tensor(x, dtype=torch.long) for x in hf_split["input_ids"]]
            self.attention_mask = [torch.tensor(x, dtype=torch.long) for x in hf_split["attention_mask"]]
            self.labels = [torch.tensor(x, dtype=torch.long) for x in hf_split["labels"]]

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx],
            }

    eval_ds = PerplexityTorchDataset(dataset_split)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size)
    model_device = _get_model_device(model_to_evaluate)

    print(f"Calculating perplexity on {len(dataset_split)} samples (batch_size={batch_size}) ...")

    total_loss = 0.0
    num_batches = 0
    progress = tqdm(eval_loader, desc="Evaluating", unit="batch")

    for batch in progress:
        input_ids = batch["input_ids"].to(model_device)
        attention_mask = batch["attention_mask"].to(model_device)
        labels = batch["labels"].to(model_device)

        with torch.no_grad():
            output = model_to_evaluate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        loss = output.loss
        total_loss += loss.item()
        num_batches += 1
        progress.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            avg_loss=f"{total_loss / num_batches:.4f}",
        )

    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return torch.exp(torch.tensor(average_loss)).item()
