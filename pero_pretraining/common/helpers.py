import os

def get_checkpoint_path(checkpoints_directory, iteration):
    return os.path.join(checkpoints_directory, f"checkpoint_{iteration:06d}.pth")

def get_visualization_path(visualizations_directory, iteration, part):
    return os.path.join(visualizations_directory, f"{part}_{iteration:06d}.png")
