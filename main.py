import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from data_loader import get_dataloaders
from models import BaseCNN
from train_base_model import train_model

# Import new modules
from traditional_methods import run_pearson, run_dom, run_snr, run_mrmr
from xai_methods import run_saliency, run_smoothgrad, run_integrated_gradients, run_lrp

def save_plot(data, title, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(data.flatten())
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Importance/Value")
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()



def main():
    parser = argparse.ArgumentParser(description="SCA POI Selection Framework")
    
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to .h5 file')
    parser.add_argument('--method', type=str, default='train', 
                        choices=['train', 'test', 'pearson', 'dom', 'snr', 'mrmr', 
                                 'saliency', 'smoothgrad', 'ig', 'lrp'], 
                        help='Method to run')
    parser.add_argument('--model_path', type=str, default='base_model.pth', help='Path to save/load model')
    parser.add_argument('--input_length', type=int, default=700)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading Data from {args.dataset_path}...")
    train_loader, test_loader = get_dataloaders(args.dataset_path, args.batch_size, args.input_length)
    
    model = BaseCNN(input_length=args.input_length).to(device)
    
    # --- Training ---
    if args.method == 'train':
        print("Starting Training...")
        train_model(model, train_loader, test_loader, epochs=args.epochs, device=device, save_path=args.model_path)
        return

    elif args.method == 'test':
        print("--- Running Full Evaluation (Accuracy + GE) ---")
        try:
            model.load_state_dict(torch.load(args.model_path))
            print(f"Loaded weights from {args.model_path}")
        except FileNotFoundError:
            print("Error: Model file not found. Train first!")
            return

        model.eval()
        
        # 1. Calculate Accuracy
        # (Assuming evaluate is defined/imported from train_base_model)
        from train_base_model import evaluate
        acc = evaluate(model, test_loader, device)
        print(f"Test Accuracy: {acc:.2f}%")
        
        # 2. Calculate Guessing Entropy (GE)
        all_preds = []
        all_pts = []
        all_keys = []
        
        with torch.no_grad():
            for traces, labels, pts, keys in test_loader:
                traces = traces.to(device)
                output = model(traces)
                
                all_preds.append(output.cpu().numpy())
                all_pts.append(pts.numpy())
                all_keys.append(keys.numpy())
        
        predictions = np.concatenate(all_preds)
        plaintexts = np.concatenate(all_pts)
        keys = np.concatenate(all_keys)
        
        # --- FIX IS HERE ---
        # Select ONLY the target byte (Byte 2 is standard for ASCAD)
        # If using AES_HD or others, check which byte is the target (often 0 or 12/15)
        target_byte = 2 
        
        # Check dimensions before slicing
        if plaintexts.ndim > 1:
            plaintexts_byte = plaintexts[:, target_byte]
        else:
            plaintexts_byte = plaintexts

        if keys.ndim > 1:
            real_keys_byte = keys[:, target_byte]
        else:
            real_keys_byte = keys
            
        print(f"Calculating GE for Key Byte {target_byte}...")
        
        import evaluation_utils
        # Now passing shape (N,) instead of (N, 16)
        rk = evaluation_utils.compute_key_rank(predictions, plaintexts_byte, real_keys_byte)
        
        print(f"Final Guessing Entropy: {rk[-1]}")
        
        # Optional: Plot GE
        import matplotlib.pyplot as plt
        plt.plot(rk)
        plt.savefig('ge_result.png')

    # --- Traditional Methods (Need Numpy Data) ---
    if args.method in ['pearson', 'dom', 'snr', 'mrmr']:
        print("Extracting data for statistical analysis...")
        all_traces, all_pts, all_keys = [], [], []
        limit = 5000 # Use subset for speed
        count = 0
        for tr, _, pt, k in test_loader:
            all_traces.append(tr.numpy().squeeze())
            all_pts.append(pt.numpy())
            all_keys.append(k.numpy())
            count += tr.shape[0]
            if count >= limit: break
        
        traces = np.concatenate(all_traces)[:limit]
        plaintexts = np.concatenate(all_pts)[:limit]
        keys = np.concatenate(all_keys)[:limit]
        
        if args.method == 'pearson':
            res = run_pearson(traces, plaintexts, keys)
        elif args.method == 'dom':
            res = run_dom(traces, plaintexts, keys)
        elif args.method == 'snr':
            res = run_snr(traces, plaintexts, keys)
        elif args.method == 'mrmr':
            res = run_mrmr(traces, plaintexts, keys) # Returns indices, not a trace
            print("MRMR Indices:", res)
            return

        save_plot(res, f"{args.method.upper()} Result", f"{args.method}.png")

    # --- XAI Methods (Need Trained Model) ---
    elif args.method in ['saliency', 'smoothgrad', 'ig', 'lrp']:
        print("Loading Model for XAI...")
        try:
            model.load_state_dict(torch.load(args.model_path))
        except FileNotFoundError:
            print("Error: Model not found. Train first!")
            return
            
        if args.method == 'saliency':
            res = run_saliency(model, test_loader, device)
        elif args.method == 'smoothgrad':
            res = run_smoothgrad(model, test_loader, device)
        elif args.method == 'ig':
            res = run_integrated_gradients(model, test_loader, device)
        elif args.method == 'lrp':
            res = run_lrp(model, test_loader, device)
            
        if res is not None:
            save_plot(res.squeeze(), f"{args.method.upper()} Attribution", f"{args.method}.png")

if __name__ == "__main__":
    main()