import argparse
from stable_baselines3 import PPO
from src.data.fundus_dataset import get_fundus_data_loaders
from src.models.cnn_model import FlexibleCNN
from src.training.trainer import ModelTrainer
from src.rl.hpo_env import HPOEnvironment
import gymnasium as gym

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True,
                      help="Path to balanced_full_df.csv")
    parser.add_argument("--images_dir", type=str, required=True,
                      help="Path to merged_images folder")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--rl_steps", type=int, default=10000)
    return parser.parse_args()

def main(args):
    # Load data
    train_loader, val_loader = get_fundus_data_loaders(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        batch_size=args.batch_size
    )
    
    # Initialize model
    model = FlexibleCNN(num_classes=args.num_classes)
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Create RL environment
    env = HPOEnvironment(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=args.num_classes
    )
    
    # Initialize RL agent
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train RL agent
    model.learn(total_timesteps=args.rl_steps)
    
    # Save the best model
    model.save("best_hpo_model")

if __name__ == "__main__":
    args = parse_args()
    main(args)
