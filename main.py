import argparse
import torch
import wandb
from stable_baselines3 import PPO
from src.data.fundus_dataset import get_fundus_data_loaders
from src.models.cnn_model import FlexibleCNN
from src.training.trainer import ModelTrainer
from src.rl.hpo_env import HPOEnvironment
import gymnasium as gym
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="csv_files/balanced_full_df.csv",
                      help="Path to balanced_full_df.csv")
    parser.add_argument("--images_dir", type=str, default="datasets/merged_images",
                      help="Path to merged_images folder")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--rl_steps", type=int, default=20480,  # 10 updates * 2048 steps
                      help="Number of timesteps for RL training")
    parser.add_argument("--experiment_name", type=str, default="HPO-CNN-Test",
                      help="Name for the wandb experiment")
    return parser.parse_args()

def main(args):
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("Loading datasets...")
    train_loader, val_loader = get_fundus_data_loaders(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        batch_size=args.batch_size
    )
    print(f"Dataset loaded - Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    print(f"Batch size: {args.batch_size}")
    
    # Initialize model
    print("Initializing CNN model...")
    model = FlexibleCNN(num_classes=args.num_classes)
    model = model.to(device)
    print(f"Model initialized with {args.num_classes} classes")
    print("Model architecture:")
    print(model)
    
    # Initialize trainer
    print("Setting up model trainer...")
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_wandb=True  # Enable wandb in trainer
    )
    
    # Create RL environment with explicit float32 bounds
    print("Creating RL environment...")
    env = HPOEnvironment(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=args.num_classes,
        experiment_name=args.experiment_name,
        dtype=np.float32  # Explicitly set dtype for Box spaces
    )
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    
    try:
        # Initialize RL agent with better default parameters
        print("Initializing PPO agent...")
        rl_model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            device='cpu',
            n_steps=2048,      # Default PPO steps per update
            batch_size=64,     # Reasonable batch size for training
            n_epochs=10,       # More epochs for better policy optimization
            learning_rate=3e-4, # Standard learning rate for PPO
            clip_range=0.2,
            ent_coef=0.01,     # Encourage exploration
            vf_coef=0.5,       # Balance value function learning
            max_grad_norm=0.5,  # Prevent exploding gradients
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[64, 64],  # Policy network architecture
                    vf=[64, 64]   # Value function network architecture
                )
            )
        )
        
        # Train RL agent
        print(f"Starting training for {args.rl_steps} steps...")
        history = []
        
        def callback(locals, globals):
            # Store training history
            if locals.get('self') and hasattr(locals['self'], 'env'):
                try:
                    val_accuracy = locals['self'].env.get_attr('trainer')[0].validate()['accuracy']
                    history.append({
                        'step': len(history),
                        'reward': locals.get('rewards', [0])[0],
                        'val_accuracy': val_accuracy
                    })
                    print(f"Step {len(history)}: Reward = {history[-1]['reward']:.2f}, "
                          f"Validation Accuracy = {val_accuracy:.2f}%")
                except Exception as e:
                    print(f"Warning: Could not log step metrics: {str(e)}")
            return True
        
        rl_model.learn(
            total_timesteps=args.rl_steps,
            callback=callback,
            progress_bar=True
        )
        print("Training completed")
        
        # Print final results
        print("\nTraining History:")
        for step in history:
            print(f"Step {step['step']}: Reward = {step['reward']:.2f}, "
                  f"Validation Accuracy = {step['val_accuracy']:.2f}%")
        
        # Save the best model
        print("Saving model...")
        rl_model.save("best_hpo_model")
        print("Model saved successfully")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean up
        print("Cleaning up...")
        env.close()
        print("Done")

if __name__ == "__main__":
    args = parse_args()
    main(args)
