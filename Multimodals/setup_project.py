import os
import pandas as pd

def setup_project_structure():
    # Define the base path (current directory)
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Define directories to create
    directories = [
        os.path.join(base_path, "dataset", "real_videos"),
        os.path.join(base_path, "dataset", "scam_videos"),
        os.path.join(base_path, "dataset", "processed_frames"),
        os.path.join(base_path, "models", "weights"),
    ]
    
    # Create directories
    print("Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")
        
    # Create dummy CSV
    csv_path = os.path.join(base_path, "training_data.csv")
    print(f"Creating dummy CSV at: {csv_path}")
    
    data = {
        "filename": ["video_real_001.mp4", "video_scam_001.mp4"],
        "followers": [15000, 200],
        "likes": [1200, 50],
        "comment_text": ["Great content!", "Click link for free money"],
        "label": [1, 0]  # 1 for Real, 0 for Scam
    }
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print("Dummy CSV created successfully.")

if __name__ == "__main__":
    setup_project_structure()
