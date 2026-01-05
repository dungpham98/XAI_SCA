import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Try importing Captum for robust XAI
try:
    from captum.attr import Saliency, IntegratedGradients, NoiseTunnel, LRP
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("Warning: Captum not installed. LRP will not be available. Using manual implementations for others.")

def run_saliency(model, data_loader, device='cuda'):
    """Vanilla Saliency Map (Gradient w.r.t Input)."""
    print("Running Saliency Map...")
    model.eval()
    model.to(device)
    
    accumulated_grads = None
    count = 0
    
    for traces, labels, _, _ in data_loader:
        traces = traces.to(device)
        traces.requires_grad = True
        labels = labels.to(device)
        
        # Forward
        outputs = model(traces)
        # Get score for the correct class
        score_max, _ = torch.max(outputs, dim=1)
        
        # Backward
        model.zero_grad()
        score_max.sum().backward()
        
        # Get Gradients
        grads = traces.grad.abs().detach().cpu().numpy()
        grads = np.mean(grads, axis=0) # Average over batch
        
        if accumulated_grads is None:
            accumulated_grads = np.zeros_like(grads)
        
        accumulated_grads += grads
        count += 1
        
    return accumulated_grads / count

def run_smoothgrad(model, data_loader, device='cuda', n_samples=20, stdev_spread=0.15):
    """SmoothGrad: Averages Saliency Maps over noisy inputs."""
    print(f"Running SmoothGrad (samples={n_samples})...")
    model.eval()
    model.to(device)
    
    accumulated_grads = None
    count = 0
    
    for traces, labels, _, _ in data_loader:
        traces = traces.to(device)
        labels = labels.to(device)
        
        stdev = stdev_spread * (traces.max() - traces.min()).item()
        
        batch_grads = np.zeros(traces.shape) # Aggregate for this batch
        
        for _ in range(n_samples):
            noise = torch.normal(0, stdev, size=traces.shape).to(device)
            noisy_traces = traces + noise
            noisy_traces.requires_grad = True
            
            outputs = model(noisy_traces)
            score_max, _ = torch.max(outputs, dim=1)
            
            model.zero_grad()
            score_max.sum().backward()
            
            grads = noisy_traces.grad.abs().detach().cpu().numpy()
            batch_grads += grads
            
        batch_grads /= n_samples
        batch_mean = np.mean(batch_grads, axis=0) # Average over batch
        
        if accumulated_grads is None:
            accumulated_grads = np.zeros_like(batch_mean)
            
        accumulated_grads += batch_mean
        count += 1
        
    return accumulated_grads / count

def run_integrated_gradients(model, data_loader, device='cuda', steps=50):
    """Integrated Gradients: Integral of gradients from baseline (0) to input."""
    print(f"Running Integrated Gradients (steps={steps})...")
    model.eval()
    model.to(device)
    
    accumulated_attr = None
    count = 0
    
    for traces, labels, _, _ in data_loader:
        traces = traces.to(device)
        # Baseline is zero
        baseline = torch.zeros_like(traces)
        
        # Scaled inputs
        # We process one batch, but IG needs a loop over steps
        # To avoid OOM, we do it manually per batch
        
        batch_attr = torch.zeros_like(traces)
        
        for step in range(steps):
            alpha = step / steps
            interpolated = baseline + alpha * (traces - baseline)
            interpolated.requires_grad = True
            
            outputs = model(interpolated)
            score_max, _ = torch.max(outputs, dim=1)
            
            model.zero_grad()
            score_max.sum().backward()
            
            batch_attr += interpolated.grad
            
        # Approximation: (Input - Baseline) * AvgGrad
        avg_grad = batch_attr / steps
        attr = (traces - baseline) * avg_grad
        
        attr = attr.abs().detach().cpu().numpy()
        attr_mean = np.mean(attr, axis=0)
        
        if accumulated_attr is None:
            accumulated_attr = np.zeros_like(attr_mean)
        
        accumulated_attr += attr_mean
        count += 1
        
    return accumulated_attr / count

def run_lrp(model, data_loader, device='cuda'):
    """Layerwise Relevance Propagation (LRP). Uses Captum."""
    if not CAPTUM_AVAILABLE:
        print("Error: Captum not found. Please install via 'pip install captum'.")
        return None
        
    print("Running LRP (via Captum)...")
    model.eval()
    model.to(device)
    lrp = LRP(model)
    
    accumulated_attr = None
    count = 0
    
    for traces, labels, _, _ in data_loader:
        traces = traces.to(device)
        labels = labels.to(device)
        
        # Captum expects target class index
        # LRP can be unstable with certain layers (ReLU), Captum handles basic rules
        # epsilon rule is default usually
        attr = lrp.attribute(traces, target=labels)
        
        attr = attr.abs().detach().cpu().numpy()
        attr_mean = np.mean(attr, axis=0)
        
        if accumulated_attr is None:
            accumulated_attr = np.zeros_like(attr_mean)
        
        accumulated_attr += attr_mean
        count += 1
        
    return accumulated_attr / count