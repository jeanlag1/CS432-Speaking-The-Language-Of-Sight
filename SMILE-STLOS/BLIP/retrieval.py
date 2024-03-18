import argparse
import os
import json
import datetime

import numpy as np
import pandas as pd  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import clip  

try:
    import ruamel.yaml as yaml
except ImportError:
    import ruamel_yaml as yaml

from pathlib import Path

from sklearn.metrics import f1_score, precision_score, recall_score



def train(model, train_loader, optimizer, scheduler, epoch, device, max_grad_norm=2.0):
    model.train()
    metrics_accumulator = {'ClipScore': 0, 'R@1': 0, 'R@5': 0, 'average_loss': 0}
    total_batches = len(train_loader)

    for batch_idx, (images, captions) in enumerate(train_loader):
        images, captions = images.to(device), captions.to(device)

        optimizer.zero_grad()
        predicted_embeddings, true_embeddings = model(images, captions)
        
        batch_metrics = calculate_metrics(predicted_embeddings, true_embeddings, captions)

        loss = batch_metrics['average_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if scheduler:
            scheduler.step()

        for key in metrics_accumulator:
            metrics_accumulator[key] += batch_metrics.get(key, 0)

        if batch_idx % 10 == 0:
            log_training_progress(epoch, batch_idx, total_batches, batch_metrics)

    for metric in metrics_accumulator:
        metrics_accumulator[metric] /= total_batches

    print(f"Epoch {epoch} completed. Summary:")
    detailed_metrics_report(metrics_accumulator)

@torch.no_grad()
def evaluation(model, data_loader, device):
    model.eval()
    metrics_accumulator = {'f1_score': 0, 'r@1': 0, 'r@5': 0}
    total_batches = len(data_loader)

    for images, true_labels in data_loader:
        images, true_labels = images.to(device), true_labels.to(device)

        with torch.no_grad():
            outputs = model(images)

        predicted_labels = process_model_output(outputs)
        batch_metrics = calculate_metrics(predicted_labels, true_labels, outputs)

        # Accumulate metrics
        for key in metrics_accumulator:
            metrics_accumulator[key] += batch_metrics[key]

    # Average metrics over all batches
    for metric in metrics_accumulator:
        metrics_accumulator[metric] /= total_batches

    # Assuming a more detailed print/display of metrics is desired
    detailed_metrics_report(metrics_accumulator)

    return metrics_accumulator

def detailed_metrics_report(metrics):
    print(f"{'Metric':<15}{'Value':<10}")
    for metric, value in metrics.items():
        print(f"{metric:<15}{value:<10.4f}")
def process_model_output(outputs):
    sigmoid = torch.nn.Sigmoid()
    probabilities = sigmoid(outputs)
    predicted_labels = (probabilities > 0.5).int()
    return predicted_labels

def calculate_metrics(predicted_labels, true_labels, outputs):
    batch_size = true_labels.size(0)
    
    true_positives = (predicted_labels * true_labels).sum(dim=0).float()
    precision = true_positives / (predicted_labels.sum(dim=0).float() + 1e-8)
    recall = true_positives / (true_labels.sum(dim=0).float() + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    f1_score = f1_score.mean().item()
    
    r1 = (predicted_labels != true_labels).float().sum() / (batch_size * true_labels.size(1))
    
    r5 = (predicted_labels == true_labels).all(dim=1).float().mean()
    
    return {
        'f1_score': f1_score,
        'r@1': r1,
        'r@5': r5
    }


def main(training_args, config_settings):
    setup_environment(training_args.seed, training_args.distributed, training_args.device)

    train_loader = prepare_data_loader(config_settings, 'train', training_args.device, training_args.distributed)
    val_loader = prepare_data_loader(config_settings, 'val', training_args.device, training_args.distributed)
    test_loader = prepare_data_loader(config_settings, 'test', training_args.device, training_args.distributed)

    model = initialize_model(config_settings)
    model.to(training_args.device)

    optimizer = setup_optimizer(model, config_settings)
    lr_scheduler = setup_scheduler(optimizer, config_settings)

    # Training loop
    best_val_score = float('-inf')
    for epoch in range(config_settings['epochs']):
        train_epoch(model, train_loader, optimizer, epoch, training_args.device)
        lr_scheduler.step()

        # Validation step
        if epoch % config_settings['val_interval'] == 0 or epoch == config_settings['epochs'] - 1:
            val_score = validate_model(model, val_loader, training_args.device)

            if val_score > best_val_score:
                best_val_score = val_score
                save_checkpoint(model, optimizer, training_args.output_dir, epoch, best_val_score)


    if training_args.evaluate:
        test_score = evaluate_model(model, test_loader, training_args.device)
        print(f"Test Score: {test_score}")

    print("Training completed successfully.")

def setup_environment():
    parser = argparse.ArgumentParser(description='Training.')
    parser.add_argument('--config_file', default='./configs/retrieval_config.yaml', help='Path to the configuration file.')
    parser.add_argument('--output_directory', default='./training_outputs', help='Directory for saving training outputs.')
    parser.add_argument('--evaluate_only', action='store_true', help='Run in evaluation mode only.')
    parser.add_argument('--device_type', default='cuda', choices=['cuda', 'cpu'], help='Device to use for training.')
    parser.add_argument('--random_seed', default=37, type=int, help='Seed for random number generators.')
    parser.add_argument('--distributed_training', action='store_true', help='Enable distributed training mode.')
    args = parser.parse_args()


    with open(args.config_file, 'r') as config_file:
        training_config = yaml.safe_load(config_file)

    Path(args.output_directory).mkdir(parents=True, exist_ok=True)
    result_directory = Path(args.output_directory, 'results')
    result_directory.mkdir(exist_ok=True)

    with open(os.path.join(args.output_directory, 'config_copy.yaml'), 'w') as config_copy:
        yaml.safe_dump(training_config, config_copy)

    return args, training_config

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    args = parser.parse_args()

    #config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    yaml = yaml.YAML(typ='rt')
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)