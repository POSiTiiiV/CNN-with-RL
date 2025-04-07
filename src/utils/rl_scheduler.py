import logging

logger = logging.getLogger(__name__)

class RLStepScheduler:
    """
    Dynamically schedules both the RL agent's training effort (training_timesteps) and 
    the number of epochs per RL step based on CNN training performance.
    
    The scheduler adjusts both parameters based on:
    - Validation loss improvements
    - Stagnation patterns
    - Training performance trends
    """
    
    def __init__(self,
                initial_training_timesteps=1000,
                min_training_timesteps=500,
                max_training_timesteps=5000,
                initial_epochs_per_step=3,
                min_epochs_per_step=2,
                max_epochs_per_step=7,
                stagnation_patience=2):
        """
        Initialize the RL step scheduler.
        
        Args:
            initial_training_timesteps (int): Starting number of timesteps for RL training
            min_training_timesteps (int): Minimum allowed timesteps for RL training
            max_training_timesteps (int): Maximum allowed timesteps for RL training
            initial_epochs_per_step (int): Starting number of epochs to train CNN per RL step
            min_epochs_per_step (int): Minimum epochs to train CNN per RL step
            max_epochs_per_step (int): Maximum epochs to train CNN per RL step
            stagnation_patience (int): Number of steps with minimal improvement to detect stagnation
        """
        # RL training effort parameters - more moderate values to avoid excessive training
        self.training_timesteps = initial_training_timesteps
        self.min_training_timesteps = min_training_timesteps
        self.max_training_timesteps = max_training_timesteps
        
        # Epochs per step parameters
        self.epochs_per_step = initial_epochs_per_step
        self.min_epochs_per_step = min_epochs_per_step
        self.max_epochs_per_step = max_epochs_per_step
        
        self.stagnation_patience = stagnation_patience
        self.val_loss_history = []
        self.improvement_history = []
        self.adjustment_count = 0
        
    def record_validation_loss(self, val_loss):
        """
        Record a new validation loss value.
        
        Args:
            val_loss (float): The validation loss to record
        """
        self.val_loss_history.append(val_loss)
        
        # Also track improvements if we have at least 2 data points
        if len(self.val_loss_history) >= 2:
            improvement = self.val_loss_history[-2] - self.val_loss_history[-1]
            self.improvement_history.append(improvement)
            
            # Log the improvement
            if improvement > 0:
                logger.info(f"Validation loss improved by {improvement:.6f}")
            else:
                logger.info(f"Validation loss worsened by {abs(improvement):.6f}")

    def should_adjust_steps(self):
        """
        Determine if we have enough data to adjust RL parameters.
        
        Returns:
            bool: True if we have enough data to make an adjustment
        """
        return len(self.val_loss_history) >= self.stagnation_patience + 1

    def adjust_steps(self):
        """
        Adjust both training_timesteps and epochs_per_step based on recent validation loss trends.
        
        Returns:
            tuple: (training_timesteps, epochs_per_step) - The updated values
        """
        if not self.should_adjust_steps():
            logger.info(f"Not enough data to adjust RL parameters yet, keeping at training_timesteps={self.training_timesteps}, epochs_per_step={self.epochs_per_step}")
            return self.training_timesteps, self.epochs_per_step

        # Compare latest losses
        recent_losses = self.val_loss_history[-(self.stagnation_patience+1):]
        
        # Check if validation loss is consistently improving
        improving = all(
            recent_losses[i] > recent_losses[i + 1]
            for i in range(len(recent_losses) - 1)
        )
        
        # Use the is_stagnating method instead of duplicating the stagnation check
        stagnating = self.is_stagnating()
        
        # Calculate average recent improvement
        recent_improvements = [recent_losses[i] - recent_losses[i+1] for i in range(len(recent_losses)-1)]
        avg_improvement = sum(recent_improvements) / len(recent_improvements)
        
        prev_training_timesteps = self.training_timesteps
        prev_epochs_per_step = self.epochs_per_step
        
        # Decision logic for adjusting steps
        if improving and avg_improvement > 0.001:
            # Significant improvement
            # - Increase training timesteps (let RL learn more thoroughly)
            # - Increase epochs per step (more CNN training per step for faster convergence)
            self.training_timesteps = min(self.training_timesteps + 1000, self.max_training_timesteps)
            self.epochs_per_step = min(self.epochs_per_step + 1, self.max_epochs_per_step)
            adjustment_reason = "significant improvement"
        elif improving:
            # Moderate improvement
            # - Slightly increase training timesteps
            # - Keep epochs per step the same
            self.training_timesteps = min(self.training_timesteps + 500, self.max_training_timesteps)
            adjustment_reason = "moderate improvement"
        elif stagnating:
            # Stagnation
            # - Decrease training timesteps (less time on current policy)
            # - Decrease epochs per step (less time wasted on poor hyperparams)
            self.training_timesteps = max(self.training_timesteps - 1000, self.min_training_timesteps)
            self.epochs_per_step = max(self.epochs_per_step - 1, self.min_epochs_per_step)
            adjustment_reason = "stagnation"
        else:
            # Loss fluctuating or slightly worsening
            # - Decrease training timesteps slightly
            # - Keep epochs per step the same
            self.training_timesteps = max(self.training_timesteps - 500, self.min_training_timesteps)
            adjustment_reason = "fluctuating or worsening"
        
        # Track adjustment for logging
        self.adjustment_count += 1
        
        # Log changes
        logger.info(f"RL parameter adjustment #{self.adjustment_count} (reason: {adjustment_reason}, avg improvement: {avg_improvement:.6f})")
        
        # Log training timesteps changes if any
        if self.training_timesteps != prev_training_timesteps:
            logger.info(f"Training timesteps: {prev_training_timesteps} → {self.training_timesteps}")
        else:
            logger.info(f"Training timesteps unchanged at {self.training_timesteps}")
            
        # Log epochs per step changes if any
        if self.epochs_per_step != prev_epochs_per_step:
            logger.info(f"Epochs per step: {prev_epochs_per_step} → {self.epochs_per_step} epochs")
        else:
            logger.info(f"Epochs per step unchanged at {self.epochs_per_step} epochs")
        
        return self.training_timesteps, self.epochs_per_step
    
    def get_current_params(self):
        """
        Get the current scheduling parameters without adjusting.
        
        Returns:
            tuple: (training_timesteps, epochs_per_step)
        """
        return self.training_timesteps, self.epochs_per_step

    def is_stagnating(self):
        """
        Check if the validation loss is stagnating based on recent history.
        This method allows external code to check for stagnation without duplicating the logic.
        
        Returns:
            bool: True if validation loss is stagnating, False otherwise
        """
        if len(self.val_loss_history) < self.stagnation_patience + 1:
            return False
            
        recent_losses = self.val_loss_history[-(self.stagnation_patience+1):]
        
        # Check if validation loss is stagnating (very small changes)
        return all(
            abs(recent_losses[i] - recent_losses[i + 1]) < 1e-4
            for i in range(len(recent_losses) - 1)
        )