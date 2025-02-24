import argparse
import torch
from stable_baselines3 import PPO
from src.data.fundus_dataset import get_fundus_data_loaders
from src.models.cnn_model import FlexibleCNN
from src.training.trainer import ModelTrainer
from src.rl.hpo_env import HPOEnvironment
import numpy as np
import wandb

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

def create_ppo_agent(env):
    # Custom schedule for exploration rate
    def exploration_schedule(progress):
        return max(0.05, 0.5 * (1 - progress))  # Reduce from 0.5 to 0.05

    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device='cpu',
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=exploration_schedule,  # Dynamic exploration rate
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128],  # Larger policy network
                vf=[128, 128]   # Larger value network
            ),
            log_std_init=-2.0,  # Lower initial exploration
            ortho_init=True     # Better weight initialization
        ),
        # Enable learning from past episodes
        use_sde=True,          # State-Dependent Exploration
        sde_sample_freq=4,     # Update exploration noise every 4 steps
    )

def main(args):
    # Initialize wandb
    wandb.init(project="cnn-with-rl", name=args.experiment_name)
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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
    
    # Initialize model
    print("Initializing CNN model...")
    model = FlexibleCNN(num_classes=args.num_classes)
    model = model.to(device)
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_wandb=True,  # Enable wandb in trainer
        rl_agent=None  # Placeholder for RL agent
    )
    
    # Increase training steps
    args.rl_steps = 40960  # Double the steps
    
    # Create RL environment with explicit float32 bounds
    env = HPOEnvironment(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=args.num_classes,
        experiment_name=args.experiment_name,
        dtype=np.float32  # Explicitly set dtype for Box spaces
    )
    print("RL environment created")
    
    try:
        rl_model = create_ppo_agent(env)
        
        # Train RL agent
        print(f"Starting training for {args.rl_steps} steps...")
        history = []
        
        def callback(locals, globals):
            # Store training history
            if locals.get('self') and hasattr(locals['self'], 'env'):
                try:
                    reward = locals.get('rewards', [0])[0]
                    history.append({
                        'step': len(history),
                        'reward': reward
                    })
                    print(f"Step {len(history)}: Reward = {reward:.2f}")
                    wandb.log({
                        "step": len(history),
                        "reward": reward
                    })
                except Exception as e:
                    pass
            return True
        
        rl_model.learn(
            total_timesteps=args.rl_steps,
            callback=callback
        )
        print("Training completed")
        
        # Save the best model
        rl_model.save("best_hpo_model")
        print("Model saved successfully")
        
    except Exception as e:
        raise
    finally:
        # Clean up
        env.close()
        print("Environment closed")
        wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)
