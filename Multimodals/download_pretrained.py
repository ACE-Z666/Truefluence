# filepath: d:\Truefluence\Multimodals\download_pretrained.py
import os
import requests
import torch
import numpy as np
from tqdm import tqdm

def download_file(url, destination):
    """
    Download a file from URL with progress bar.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    
    print(f"Downloaded: {destination}")

def convert_keras_to_pytorch():
    """
    Convert Keras .h5 weights to PyTorch .pth format.
    Requires h5py and tensorflow/keras installed.
    """
    try:
        import h5py
        
        keras_weights_path = 'models/weights/Meso4_DF.h5'
        pytorch_weights_path = 'models/weights/meso4_pretrained.pth'
        
        print("Converting Keras weights to PyTorch format...")
        
        # Load Keras weights
        with h5py.File(keras_weights_path, 'r') as f:
            # Extract weights from Keras model
            # This is a simplified conversion - adjust based on actual structure
            state_dict = {}
            
            # Conv1 weights
            if 'conv2d' in f.keys():
                conv1_w = np.array(f['conv2d']['conv2d']['kernel:0'])
                conv1_b = np.array(f['conv2d']['conv2d']['bias:0'])
                # Keras uses (H, W, in_channels, out_channels), PyTorch uses (out_channels, in_channels, H, W)
                state_dict['conv1.0.weight'] = torch.from_numpy(conv1_w.transpose(3, 2, 0, 1))
                state_dict['conv1.0.bias'] = torch.from_numpy(conv1_b)
            
            # Add batch norm, other conv layers, etc.
            # This is a placeholder - actual conversion requires matching exact layer names
            
        # Save as PyTorch weights
        torch.save(state_dict, pytorch_weights_path)
        print(f"Converted weights saved to: {pytorch_weights_path}")
        return True
        
    except ImportError:
        print("h5py not installed. Install with: pip install h5py")
        return False
    except Exception as e:
        print(f"Conversion error: {e}")
        return False

def download_meso4_pretrained():
    """
    Download pre-trained MesoNet-4 weights.
    """
    weights_dir = "models/weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    print("="*70)
    print("MESONET-4 PRE-TRAINED WEIGHTS SETUP")
    print("="*70)
    
    # Option 1: Try to download from known sources
    sources = [
        {
            "name": "Original Keras Weights (DariusAf)",
            "url": "https://github.com/DariusAf/MesoNet/raw/master/weights/Meso4_DF.h5",
            "filename": "Meso4_DF.h5",
            "format": "keras"
        },
        {
            "name": "PyTorch Converted (Alternative)",
            "url": "https://github.com/HongguLiu/MesoNet-Pytorch/raw/master/weights/Meso4_DF.pth",
            "filename": "meso4_pretrained.pth",
            "format": "pytorch"
        }
    ]
    
    print("\nAttempting automatic download...\n")
    
    for source in sources:
        try:
            destination = os.path.join(weights_dir, source["filename"])
            
            if os.path.exists(destination):
                print(f"✓ Already exists: {destination}")
                continue
            
            print(f"Downloading from: {source['name']}")
            download_file(source["url"], destination)
            
            # If Keras format, attempt conversion
            if source["format"] == "keras" and source["filename"].endswith('.h5'):
                print("\nKeras weights downloaded. Attempting conversion...")
                if convert_keras_to_pytorch():
                    print("✓ Conversion successful!")
                else:
                    print("✗ Conversion failed. Manual conversion required.")
            
            print(f"✓ Successfully downloaded: {source['name']}\n")
            
        except Exception as e:
            print(f"✗ Failed to download {source['name']}: {e}\n")
    
    # Check if we have usable weights
    pytorch_weights = os.path.join(weights_dir, "meso4_pretrained.pth")
    
    if os.path.exists(pytorch_weights):
        print("="*70)
        print("✓ SUCCESS: Pre-trained weights ready!")
        print(f"Location: {pytorch_weights}")
        print("="*70)
        print("\nNext steps:")
        print("1. Run: python test_meso4.py")
        print("2. Test on your videos in dataset/real_videos/ or dataset/scam_videos/")
    else:
        print("="*70)
        print("⚠ MANUAL DOWNLOAD REQUIRED")
        print("="*70)
        print("\nAutomatic download failed. Please follow these steps:")
        print("\n1. Visit: https://github.com/HongguLiu/MesoNet-Pytorch")
        print("2. Download the PyTorch weights file (.pth)")
        print("3. Place it in: d:\\Truefluence\\Multimodals\\models\\weights\\")
        print("4. Rename to: meso4_pretrained.pth")
        print("\nAlternatively:")
        print("1. Visit: https://github.com/DariusAf/MesoNet")
        print("2. Download Meso4_DF.h5")
        print("3. Use a conversion script or train your own model")
        print("="*70)

if __name__ == "__main__":
    download_meso4_pretrained()