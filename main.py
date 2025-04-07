import os
import argparse
import yaml
import time
import json
import sys
import re
from datetime import datetime
import random
import numpy as np
import torch
import wandb
import logging
from tqdm import tqdm

# Import project modules
from src.models.cnn import PretrainedCNN
from src.models.rl_agent import HyperParameterOptimizer
from src.trainers.cnn_trainer import CNNTrainer
from src.trainers.trainer import ModelTrainer
from src.envs.hpo_env import HPOEnvironment
from src.data_loaders.data_loader import load_dataset  # Import the new data loader module

# Configure Python logging to support Unicode
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Use sys.stdout to support Unicode
        logging.FileHandler("training.log", encoding="utf-8")  # Ensure file handler uses UTF-8
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train CNN with RL-based hyperparameter optimization"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration file"
    )
    # Update default path to the custom images folder
    parser.add_argument(
        "--data-dir", type=str, default="data/merged_images",
        help="Directory for custom image dataset"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output",
        help="Directory for output files"
    )
    parser.add_argument(
        "--wandb", action="store_true", default=True,
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--no-wandb", action="store_false", dest="wandb",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None,
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb-name", type=str, default=None,
        help="Weights & Biases run name"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for training (overrides config)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--rl-brain", type=str, default=None,
        help="Path to pre-trained RL brain to load"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda, cuda:0, cpu, etc.)"
    )
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Set deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.debug(f"Set random seed to {seed}")

def get_device(args_device=None):
    """Get device for PyTorch"""
    if args_device:
        device = torch.device(args_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
        # Enable performance optimizations for GPU
        torch.backends.cudnn.benchmark = True  # Enable for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for better performance
        logger.info(f"CUDA optimizations enabled")
    
    return device

def create_model(config, device):
    """Create CNN model"""
    # Update config with dataset-specific parameters
    dataset_name = config["data"]["dataset_name"].lower()
    
    # Create model
    model = PretrainedCNN(config["model"])
    model.to(device)
    
    logger.info(f"Created model: {type(model).__name__}")
    logger.debug(f"Model config: {config['model']}")
    logger.debug(f"Model device: {next(model.parameters()).device}")
    
    return model

def init_wandb(args, config):
    """Initialize Weights & Biases for experiment tracking"""
    if not args.wandb:
        logger.warning("Weights & Biases logging disabled")
        return False
    
    try:
        # Override config with command line arguments if provided
        wandb_config = config.get("wandb", {})
        project_name = args.wandb_project or wandb_config.get("project", "CNN-with-RL")
        run_name = args.wandb_name or wandb_config.get("run_name") or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check if wandb is already initialized
        if wandb.run is not None:
            logger.info("WandB already initialized, using existing run")
            return True
            
        # Initialize wandb
        wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            dir=args.output_dir
        )
        
        # Double check that initialization worked
        if wandb.run is None:
            logger.warning("WandB initialization failed to create a run")
            return False
            
        logger.info(f"Initialized Weights & Biases: project='{project_name}', run='{run_name}'")
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize Weights & Biases: {e}")
        return False

def save_config(config, output_dir):
    """Save configuration to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved configuration to {config_path}")

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device for training
    device = get_device(args.device)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config["training"]["max_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    
    # Update config paths
    config["output_dir"] = args.output_dir
    config["data_dir"] = args.data_dir
    
    # Update wandb config
    config["logging"]["use_wandb"] = args.wandb
    if args.wandb_project:
        config.setdefault("wandb", {})["project"] = args.wandb_project
    if args.wandb_name:
        config.setdefault("wandb", {})["run_name"] = args.wandb_name
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Set up logging directories
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    config["logging"]["log_dir"] = log_dir
    
    # Save configuration
    save_config(config, run_dir)
    
    # Initialize wandb - ensure this is done BEFORE any calls to wandb.log()
    wandb_initialized = init_wandb(args, config)
    
    try:
        print('\n')
        logger.info("Loading Data and Model")
        print('\n')
        
        # Load dataset
        print('\n')
        logger.info("Loading datasets...")
        print('\n')
        train_loader, val_loader, test_loader = load_dataset(
            config, args.data_dir
        )
        print('\n')
        logger.info(f"✓ Datasets loaded successfully")
        print('\n')
        
        # Create model
        print('\n')
        logger.info("Creating model...")
        print('\n')
        model = create_model(config, device)
        print('\n')
        logger.info(f"✓ Model created: {type(model).__name__}")
        print('\n')
        
        print('\n')
        logger.info("Setting up Training Environment")
        print('\n')
        
        # Create CNN trainer
        cnn_trainer = CNNTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config["training"]
        )
        
        # Initialize HPO environment with CNN trainer
        env = HPOEnvironment(cnn_trainer, config['env'])
        
        # Create RL optimizer
        rl_optimizer = HyperParameterOptimizer(env, config['rl'])
        
        # Determine if there are previous episodes to load
        loading_episodes = os.path.exists('logs/rl_episodes.json')
        
        # Handle RL brain loading (completely separate from episode loading)
        if args.rl_brain:
            brain_path = args.rl_brain
            logger.info(f"Loading RL brain from {brain_path}...")
            if rl_optimizer.load_brain(brain_path):
                logger.info(f"✓ Successfully loaded RL brain")
            else:
                logger.warning(f"Failed to load RL brain from {brain_path}")

        # Handle environment reset and previous episodes (separate from brain loading)
        # We want to preserve metrics if previous episodes exist, regardless of RL brain loading
        if loading_episodes:
            observation, info = env.reset(options={'loading_previous_episodes': True})
            logger.info("Environment reset with preserved best metrics from previous episodes")
        else:
            observation, info = env.reset()
            logger.info("Environment reset with fresh metrics (no previous episodes found)")
        
        # Set the agent in the environment for direct saving
        env.set_agent(rl_optimizer.agent)
        
        # Create model trainer
        trainer = ModelTrainer(
            cnn_trainer=cnn_trainer,
            rl_optimizer=rl_optimizer,
            config=config
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}...")
            trainer.load_checkpoint(args.resume)
            logger.info(f"✓ Checkpoint loaded")
        
        print('\n')
        logger.info("Starting Training")
        print('\n')
        print('\n')
        logger.info("Training model with RL-based hyperparameter optimization...")
        print('\n')
        start_time = time.time()
        
        history = trainer.train(epochs=args.epochs)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        print('\n')
        logger.info("Evaluation")
        print('\n')
        print('\n')
        logger.info("Evaluating on test set...")
        print('\n')
        test_metrics = cnn_trainer.evaluate(test_loader)
        test_loss = test_metrics.get('loss', 0)
        test_acc = test_metrics.get('accuracy', 0)
        
        # Convert the values to float before formatting
        try:
            test_acc = float(test_acc) if isinstance(test_acc, str) else test_acc
            test_loss = float(test_loss) if isinstance(test_loss, str) else test_loss
            print('\n')
            logger.info(f"Test results: Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
            print('\n')
        except (ValueError, TypeError):
            logger.warning(f"Test results: Accuracy: {test_acc}, Loss: {test_loss}")
            logger.warning("Warning: Could not format test metrics as floats. Check data types.")
        
        # Save results
        results_file = os.path.join(run_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump({
                "best_val_accuracy": trainer.best_val_acc,
                "best_val_loss": trainer.best_val_loss,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "training_time": training_time,
                "epochs": trainer.current_epoch + 1,
                "early_stopped": trainer.epochs_without_improvement >= trainer.early_stopping_patience,
                "rl_interventions_count": len(trainer.history["rl_interventions"]),
            }, f, indent=2)
        
        print('\n')
        logger.info(f"✓ Results saved to {results_file}")
        print('\n')
        
        # Save final RL brain
        final_brain_path = os.path.join(run_dir, "final_rl_brain.zip")
        rl_optimizer.save_brain(final_brain_path)
        print('\n')
        logger.info(f"✓ Final RL brain saved to {final_brain_path}")
        print('\n')
        
        # Finalize wandb
        if wandb_initialized and wandb.run is not None:
            # Log final results
            try:
                wandb.log({
                    "best_val_accuracy": trainer.best_val_acc,
                    "best_val_loss": trainer.best_val_loss,
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                    "training_time": training_time,
                    "final_epoch": trainer.current_epoch + 1,
                })
                
                # Save artifacts
                best_model_path = os.path.join(log_dir, "best_model.pt")
                if os.path.isfile(best_model_path):
                    model_artifact = wandb.Artifact(
                        name=f"model_{wandb.run.id}",
                        type="model",
                        description="Trained CNN model with RL-optimized hyperparameters"
                    )
                    model_artifact.add_file(best_model_path)
                    wandb.log_artifact(model_artifact)
                    logger.info(f"✓ Model artifact uploaded to W&B")
                else:
                    logger.error(f"✗ Best model file not found: {best_model_path}")
                
                wandb.finish()
                logger.info("✓ W&B logging completed")
            except Exception as wandb_error:
                logger.error(f"Error during W&B logging: {str(wandb_error)}")
                # Try to finish the run even if there was an error
                try:
                    if wandb.run is not None:
                        wandb.finish(exit_code=1)
                except:
                    pass
        
        print('\n')
        logger.info("Training Complete!")
        print('\n')
        
    except Exception as e:
        logger.exception("An error occurred during training")
        logger.error(f"Error during training: {str(e)}")
        if wandb_initialized and wandb.run is not None:
            # Log error and finish wandb run
            try:
                wandb.log({"error": str(e)})
                wandb.finish(exit_code=1)
            except:
                pass
        raise

if __name__ == "__main__":
    main()