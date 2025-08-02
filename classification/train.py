import os
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from datasets import Dataset as HFDataset
import evaluate
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset():
    """Load dataset directly from Excel file with query and label columns"""
    logger.info("Loading dataset from analysis/dataset.xlsx...")
    
    # Read the Excel file directly - should have 'query' and 'label' columns
    df = pd.read_excel("analysis/dataset.xlsx")
    
    logger.info(f"Loaded {len(df)} examples")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Extract queries and labels
    queries = df['query'].tolist() if 'query' in df.columns else df.iloc[:, 0].tolist()
    labels = df['label'].tolist() if 'label' in df.columns else df.iloc[:, 1].tolist()
    
    logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    return queries, labels
        

def compute_metrics(eval_pred):
    """Compute metrics for evaluation using HuggingFace evaluate library"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Load metrics from evaluate library
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    
    # Compute metrics
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
    precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted')
    recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')
    
    return {
        'accuracy': accuracy['accuracy'],
        'f1': f1['f1'],
        'precision': precision['precision'],
        'recall': recall['recall']
    }

def create_hf_dataset(texts, labels, tokenizer, max_length=512):
    """Create HuggingFace dataset from texts and labels"""
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Create HF dataset
    dataset = HFDataset.from_dict({
        'text': texts,
        'labels': labels
    })
    
    # Tokenize
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset

def main():
    # Configuration - optimized hyperparameters
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5  # Recommended for DistilBERT
    NUM_EPOCHS = 5  # More epochs for better convergence
    DROPOUT_RATE = 0.1
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    queries, labels = load_dataset()
    
    logger.info(f"Dataset stats:")
    logger.info(f"  Total samples: {len(queries)}")
    logger.info(f"  Unique labels: {len(set(labels))}")
    logger.info(f"  Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    # Create train/val/test splits (0.7/0.1/0.2)
    logger.info("Creating train/val/test splits...")
    
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        queries, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Second split: 87.5% train, 12.5% val (of the remaining 80%)
    # This gives us 70% train, 10% val, 20% test overall
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Split sizes:")
    logger.info(f"  Train: {len(X_train)} ({len(X_train)/len(queries):.1%})")
    logger.info(f"  Val: {len(X_val)} ({len(X_val)/len(queries):.1%})")
    logger.info(f"  Test: {len(X_test)} ({len(X_test)/len(queries):.1%})")
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create model with 4 labels and optimal configuration
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        id2label={0: 'MULTIMODAL-MULTI', 1: 'MULTIMODAL-SINGLE', 2: 'TEXT-MULTI', 3: 'TEXT-SINGLE'},
        label2id={'MULTIMODAL-MULTI': 0, 'MULTIMODAL-SINGLE': 1, 'TEXT-MULTI': 2, 'TEXT-SINGLE': 3},
        hidden_dropout_prob=DROPOUT_RATE,
        attention_probs_dropout_prob=DROPOUT_RATE
    )
    
    # Freeze DistilBERT layers (keep only classification head trainable)
    logger.info("Freezing DistilBERT base layers...")
    for name, param in model.distilbert.named_parameters():
        param.requires_grad = False
    
    # Keep classifier trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.1%})")
    
    # Create datasets
    logger.info("Creating tokenized datasets...")
    train_dataset = create_hf_dataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = create_hf_dataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = create_hf_dataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments following best practices
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,  # Recommended warmup steps
        warmup_ratio=0.1,  # 10% of training steps for warmup
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",  # Evaluate every epoch
        save_strategy="epoch",
        save_total_limit=2,  # Keep only best 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="linear",
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        dataloader_num_workers=4,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,  # Disable wandb/tensorboard for now
    )
    
    # Create trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    logger.info("Test Results:")
    for key, value in test_results.items():
        if key.startswith('eval_'):
            metric_name = key.replace('eval_', '')
            logger.info(f"  {metric_name}: {value:.4f}")
    
    # Save the model
    model_save_path = './trained_model'
    logger.info(f"Saving model to {model_save_path}")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Test predictions on a few examples
    logger.info("Testing predictions on sample queries...")
    model.eval()
    
    sample_queries = [
        "What was the revenue growth in the last quarter?",
        "Show me financial charts and performance metrics",
        "Analyze the profit margins from the annual report"
    ]
    
    for query in sample_queries:
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
            
        label_mapping = {0: 'MULTIMODAL-MULTI', 1: 'MULTIMODAL-SINGLE', 2: 'TEXT-MULTI', 3: 'TEXT-SINGLE'}
        predicted_label = label_mapping[predicted_class]
        logger.info(f"Query: '{query}'")
        logger.info(f"  Predicted: {predicted_label} (confidence: {confidence:.3f})")
        logger.info("")

if __name__ == "__main__":
    main()
