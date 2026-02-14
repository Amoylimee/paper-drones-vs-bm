import os
from pathlib import Path


def set_working_directory() -> Path:
    """
    Set the working directory to the project root.
    This function should be called at the beginning of each script.

    Returns:
        Path: Project root directory path
    """
    # Get the directory of the current script
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # Set working directory to project root (parent of current script)
    os.chdir(current_dir)

    return current_dir


def setup_paths() -> dict:
    """
    Create a dictionary of important project paths.

    Returns:
        dict: Dictionary containing paths for different project directories
    """
    root_dir = set_working_directory()

    paths = {
        "root": root_dir,
        "data": root_dir / "data",
        "output": root_dir / "output",
        "logs": root_dir / "logs",
        "bash": root_dir / "bash",
    }

    return paths


# Example usage:
if __name__ == "__main__":
    # Set working directory
    project_dir = set_working_directory()
    print(f"Working directory set to: {project_dir}")

    # Get project paths
    paths = setup_paths()
    print("Project paths:")
    for key, path in paths.items():
        print(f"{key}: {path}")
