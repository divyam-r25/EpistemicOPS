import matplotlib.pyplot as plt
import numpy as np

def plot_reward_curve(baseline_scores: list, trained_scores: list, save_path: str = None):
    """Generates a comparison plot of reward curves before and after training."""
    epochs = np.arange(1, len(baseline_scores) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, baseline_scores, label="Baseline (Zero-shot)", color="red", linestyle="--", marker="o")
    plt.plot(epochs, trained_scores, label="Trained (GRPO)", color="blue", linewidth=2, marker="s")
    
    plt.title("EpistemicOps: Total Reward per Era (Scenario 1)")
    plt.xlabel("Era Sequence")
    plt.ylabel("Total Normalized Reward (0.0 - 1.0)")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        return plt.gcf()
