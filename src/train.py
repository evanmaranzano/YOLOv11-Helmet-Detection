from ultralytics import YOLO
import sys
import os

# Add local src to path if needed (implicit in current execution)
# Import custom modules so they are available
from modules import CBAM

# Monkey patch to allow custom modules in dataset parsing if needed
# Actually, for Ultralytics 8/11, it's safer to register the module or simple patch
# But often, just having it in the namespace might not be enough for `parse_model`.
# We'll use a standard workaround: intercepting the yaml parsing or assume the user has modifying ultralytics source is too hard.
# Let's try the standard "load weights then cfg" approach, but we need the CFG to recognize `CBAM`.
# The standard Ultralytics way for custom modules is to modify `ultralytics/nn/tasks.py`. 
# Since we can't easily modify the installed package, we will define a custom training script that overrides the internal map.

# However, to keep it simple for the user, I will try to verify if `ultralytics` has a hook. 
# If not, I'll proceed with a standard "build from yaml". If it fails on "KeyError: CBAM", I will fix it.
# Actually, I'll just write the code to use the yaml. If it fails, I'll add the patch. 
# Better: Add the patch proactively.

from ultralytics.nn import tasks, modules

import argparse
from utils import load_config
from loss import CustomDetectionLoss
from ultralytics.models.yolo.detect import DetectionTrainer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='train_config.yaml', help='Path to training config file')
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    
    print(f"Loaded configuration from: {args.config}")
    print(cfg)
    
    # Toggle Innovations based on config (Default to True if not specified)
    use_innovations = cfg.get('use_innovations', True)
    
    if use_innovations:
        print("🚀 Innovation Mode: Enabled (CBAM + Inner-WIoU)")
        # --- Apply Monkey Patches for Innovations ---
        
        # 1. CBAM Injection
        setattr(modules, 'CBAM', CBAM)
        sys.modules['ultralytics.nn.modules'].CBAM = CBAM
        setattr(tasks, 'CBAM', CBAM)
        
        # 2. Loss Injection (Inner-WIoU)
        def custom_get_loss(self):
            return CustomDetectionLoss(self)
        DetectionTrainer.get_loss = custom_get_loss
        
    else:
        print("🛡️ Baseline Mode: Enabled (Original YOLOv11s)")
        # No patches applied - using standard Ultralytics components
    
    # Load a model
    # 1. Build model from config
    model = YOLO(cfg['model_cfg'])
    
    # 2. Load pretrained weights (transfer learning)
    # This will load matching layers and ignore mismatched ones (like our new CBAM)
    model.load(cfg['weights'])

    # Train the model
    results = model.train(
        data=cfg['data'],  # path to dataset YAML
        epochs=cfg['epochs'],  # number of epochs to train for
        imgsz=cfg['imgsz'],  # size of input images as integer
        batch=cfg['batch'],  # number of images per batch
        device=cfg['device'],  # device to run on
        project=cfg['project'],  # project name
        name=cfg['name'],  # experiment name
        exist_ok=False,  # whether to overwrite existing experiment
    )

if __name__ == '__main__':
    main()
