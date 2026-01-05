import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif

def get_intermediate_values(plaintexts, keys, byte_index=2):
    """Helper to generate SBox outputs for a specific key byte."""
    sbox = np.array([
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
    ])
    # Assuming keys are fixed or variable, we take the 3rd byte (index 2)
    # Adjust logic if key is variable per trace
    real_key_byte = keys[:, byte_index] if keys.ndim > 1 else keys[byte_index]
    vals = sbox[plaintexts[:, byte_index] ^ real_key_byte]
    return vals

def run_pearson(traces, plaintexts, keys):
    """Pearson Correlation Coefficient (CPA)."""
    print("Running Pearson (CPA)...")
    # Target: Hamming Weight of SBox Output
    vals = get_intermediate_values(plaintexts, keys)
    hw = [bin(x).count("1") for x in range(256)]
    hypotheses = np.array([hw[v] for v in vals])

    mean_h = np.mean(hypotheses)
    mean_t = np.mean(traces, axis=0)
    numerator = np.sum((hypotheses[:, None] - mean_h) * (traces - mean_t), axis=0)
    denominator = np.sqrt(np.sum((hypotheses[:, None] - mean_h)**2, axis=0) * np.sum((traces - mean_t)**2, axis=0))
    return np.abs(numerator / denominator)

def run_dom(traces, plaintexts, keys):
    """Difference of Means (DoM)."""
    print("Running Difference of Means (DoM)...")
    vals = get_intermediate_values(plaintexts, keys)
    # Split into two groups based on MSB (Most Significant Bit) of the SBox output
    # Group 0: MSB is 0, Group 1: MSB is 1
    msb = (vals >> 7) & 1
    
    group0 = traces[msb == 0]
    group1 = traces[msb == 1]
    
    mean0 = np.mean(group0, axis=0)
    mean1 = np.mean(group1, axis=0)
    
    dom = np.abs(mean0 - mean1)
    return dom

def run_snr(traces, plaintexts, keys):
    """Signal to Noise Ratio (SNR)."""
    print("Running SNR...")
    vals = get_intermediate_values(plaintexts, keys)
    
    # Calculate Mean Trace per Class (Signal)
    unique_vals = np.unique(vals)
    mean_traces = []
    var_traces = []
    
    for v in unique_vals:
        group = traces[vals == v]
        if len(group) > 1:
            mean_traces.append(np.mean(group, axis=0))
            var_traces.append(np.var(group, axis=0))
            
    mean_traces = np.array(mean_traces)
    var_traces = np.array(var_traces)
    
    # Signal = Variance of the Means
    var_signal = np.var(mean_traces, axis=0)
    
    # Noise = Mean of the Variances
    mean_noise = np.mean(var_traces, axis=0)
    
    # Avoid div by zero
    snr = var_signal / (mean_noise + 1e-10)
    return snr

def run_mrmr(traces, plaintexts, keys, n_features=10):
    """
    Minimum Redundancy Maximum Relevance (MRMR).
    Uses Mutual Information. Simplified greedy implementation.
    """
    print("Running MRMR (Simplified)...")
    vals = get_intermediate_values(plaintexts, keys)
    hw = np.array([bin(x).count("1") for x in range(256)])
    targets = np.array([hw[v] for v in vals])
    
    # 1. Relevance: Calculate MI between each point and Target
    # For speed, we might use Correlation as a proxy for Relevance in large traces
    # Using sklearn mutual_info_classif is accurate but slow for 90k points
    # Here we perform a filter first using CPA to get top 500 candidates
    
    initial_scores = run_pearson(traces, plaintexts, keys)
    candidate_indices = np.argsort(initial_scores)[-500:] # Pre-select top 500
    candidate_traces = traces[:, candidate_indices]
    
    selected_features = []
    candidate_set = list(range(candidate_traces.shape[1]))
    
    # Iterative selection
    for _ in tqdm(range(n_features), desc="MRMR Selection"):
        best_score = -np.inf
        best_idx = -1
        
        for idx in candidate_set:
            # Relevance (MI with target)
            # Using correlation as fast proxy for MI here
            rel = initial_scores[candidate_indices[idx]]
            
            # Redundancy (Average MI/Corr with already selected features)
            red = 0
            if len(selected_features) > 0:
                # Calculate corr with selected
                corrs = [np.abs(np.corrcoef(candidate_traces[:, idx], candidate_traces[:, s])[0,1]) 
                         for s in selected_features]
                red = np.mean(corrs)
                
            # MRMR Score = Rel - Red
            score = rel - red
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx != -1:
            selected_features.append(best_idx)
            candidate_set.remove(best_idx)
            
    # Map back to original indices
    final_indices = candidate_indices[selected_features]
    print(f"MRMR Top Features: {final_indices}")
    return final_indices