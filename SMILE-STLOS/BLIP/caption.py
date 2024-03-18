import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from models.model import CaptionGenerator
import utility_functions as util_funcs
from data_handling import prepare_data, organize_sampler, setup_data_loader
from evaluation_metrics import store_results, evaluate_captions
from models.model import CaptionGenerator
import utility_functions as util_funcs
from data_handling import prepare_data, organize_sampler, setup_data_loader
from evaluation_metrics import store_results, evaluate_captions

def setup_training_environment(config_settings, save_directory):
    util_funcs.set_random_seeds(config_settings['seed'])
    util_funcs.prepare_device(config_settings['distributed'])

def initialize_model_and_optimizer(config_settings):
    model = CaptionGenerator(config_settings['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config_settings['lr'])
    return model, optimizer

def training_epoch(model, data_loader, optimizer, epoch_index, device):
    model.train()
    performance_logger = util_funcs.PerformanceLogger()

    for batch_index, (images, captions, _) in enumerate(data_loader):
        if batch_index > 10000:
            print("Only runnning on 10000 images DUHHHH")
            break
        images, captions = images.to(device), captions.to(device)
        training_loss = model(images, captions)
        
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()
        
        performance_logger.update_metric('training_loss', training_loss.item())
    
    print(f"Epoch {epoch_index} Training Loss: {performance_logger.metric_average('training_loss')}")
    return performance_logger.metric_average('training_loss')

def run_training_loop(model, train_loader, val_loader, optimizer, device, config_settings):
    for epoch in range(config_settings['epochs']):
        train_loss = training_epoch(model, train_loader, optimizer, epoch, device)
        val_results = run_evaluation(model, val_loader, device)
        print(f"Epoch {epoch} Validation Results: {val_results}")

@torch.no_grad()
def run_model_evaluation(caption_generator, evaluation_data_loader, device_setting, evaluation_config):
    caption_generator.eval()  # Switch model to evaluation mode

    logger = util_funcs.MetricLogger()  # Initialize a logger for performance metrics
    results_list = []  # List to store evaluation results

    # Iterate over the data loader
    for images, target_captions, image_ids in evaluation_data_loader:
        images = images.to(device_setting)  # Move images to the specified device (e.g., GPU)

        # Generate captions using the model
        generated_captions = caption_generator(images)

        # Process and store the results
        for img_id, gen_caption in zip(image_ids, generated_captions):
            results_list.append({
                'image_id': img_id.item(),  # Convert to a Python scalar
                'caption': gen_caption
            })

    # Evaluate the generated captions using COCO caption evaluation metrics
    eval_metrics = evaluate_captions(results_list, evaluation_config)

    # Log the evaluation metrics
    for metric_name, score in eval_metrics.items():
        logger.log(f"{metric_name}: {score:.3f}")

    return eval_metrics


def main(args):
    config_settings = util_funcs.load_yaml_config(args.config_file)
    setup_training_environment(config_settings, args.save_directory)
    util_funcs.save_config_copy(config_settings, args.save_directory)

    model, optimizer = initialize_model_and_optimizer(config_settings)
    train_loader, val_loader = prepare_data(config_settings)

    device = torch.device(args.computation_device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    run_training_loop(model, train_loader, val_loader, optimizer, device, config_settings)
    run_model_evaluation(model, val_loader, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='./configurations/caption_config.yaml')
    parser.add_argument('--save_directory', default='./model_outputs')
    parser.add_argument('--computation_device', default='cuda')
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--distributed_mode', default=True, type=bool)
    
    args = parser.parse_args()
    main(args)
