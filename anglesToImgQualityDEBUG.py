#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Condensed FBP Reconstruction Function
Performs FBP reconstruction using specified angles and evaluates quality using ESF.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
from skimage.transform import iradon
from skimage.io import imsave
from scipy import ndimage
from scipy.interpolate import interp1d
#up2
def calculate_esf(image):
    """
    Calculate the Edge Spread Function (ESF) of an image.
    
    Args:
        image: 2D numpy array containing the image
        
    Returns:
        tuple: (positions, esf) - position values and ESF values
    """
    # Get image dimensions
    height, width = image.shape
    
    # Detect edges in the image
    edges = ndimage.sobel(image)
    edges = np.abs(edges)
    
    # Threshold to find significant edges
    threshold = np.percentile(edges, 95)  # Use top 5% of edge values
    edge_mask = edges > threshold
    
    # If no strong edges found, create a synthetic edge for analysis
    if np.sum(edge_mask) < 10:
        # Create a horizontal line across the center for edge analysis
        center_row = height // 2
        profile = image[center_row, :]
    else:
        # Find coordinates of strong edges
        edge_y, edge_x = np.where(edge_mask)
        
        # Choose the most prominent edge (highest edge response)
        if len(edge_y) > 0:
            max_idx = np.argmax(edges[edge_y, edge_x])
            y, x = edge_y[max_idx], edge_x[max_idx]
            
            # Determine edge direction (horizontal or vertical)
            h_strength = np.sum(edge_mask[y, max(0, x-5):min(width, x+6)])
            v_strength = np.sum(edge_mask[max(0, y-5):min(height, y+6), x])
            
            # Extract profile perpendicular to the edge
            if h_strength > v_strength:
                # Horizontal edge - extract vertical profile
                profile = image[max(0, y-25):min(height, y+26), x]
            else:
                # Vertical edge - extract horizontal profile
                profile = image[y, max(0, x-25):min(width, x+26)]
        else:
            # Fallback to center row if edge detection fails
            center_row = height // 2
            profile = image[center_row, :]
    
    # Normalize profile to [0, 1]
    if np.max(profile) > np.min(profile):
        profile = (profile - np.min(profile)) / (np.max(profile) - np.min(profile))
    
    # Generate position values
    positions = np.arange(len(profile)) / len(profile)
    
    # Sort the profile to create a proper ESF (monotonically increasing)
    sorted_indices = np.argsort(profile)
    esf = profile[sorted_indices]
    
    # Normalize to [0, 1]
    esf = (esf - np.min(esf)) / (np.max(esf) - np.min(esf)) if np.max(esf) > np.min(esf) else esf
    
    # Generate equally spaced position values for the sorted ESF
    positions = np.linspace(0, 1, len(esf))
    
    return positions, esf

def compare_esf_to_ground_truth(esf_pos, esf_values, gt_esf_pos, gt_esf_values):
    """
    Compare the calculated ESF to a ground truth ESF using RMSE.
    
    Args:
        esf_pos: Position values for the calculated ESF
        esf_values: ESF values
        gt_esf_pos: Position values for the ground truth ESF
        gt_esf_values: Ground truth ESF values
        
    Returns:
        float: Root Mean Square Error between the ESFs
    """
    # Interpolate to ensure both ESFs have the same position points
    if np.array_equal(esf_pos, gt_esf_pos):
        # If positions are already identical
        common_pos = esf_pos
        common_esf = esf_values
        common_gt_esf = gt_esf_values
    else:
        # Interpolate both ESFs to a common position range
        common_pos = np.linspace(0, 1, 100)  # 100 points from 0 to 1
        
        # Create interpolation functions
        esf_interp = interp1d(esf_pos, esf_values, bounds_error=False, fill_value=(esf_values[0], esf_values[-1]))
        gt_esf_interp = interp1d(gt_esf_pos, gt_esf_values, bounds_error=False, fill_value=(gt_esf_values[0], gt_esf_values[-1]))
        
        # Interpolate to common position range
        common_esf = esf_interp(common_pos)
        common_gt_esf = gt_esf_interp(common_pos)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((common_esf - common_gt_esf)**2))
    
    return rmse

def generate_ground_truth_esf(slice_idx, all_angles_reconstruction=None):
    """
    Calculate the ground truth ESF from a full 360° angle reconstruction.
    
    Args:
        slice_idx: Index of the slice
        all_angles_reconstruction: Reconstruction using all available angles (if None, will be calculated)
        
    Returns:
        tuple: (positions, esf_values) - position values and ESF values (normalized to [0, 1])
    """
    # If a full reconstruction is provided, use it
    if all_angles_reconstruction is not None:
        # Calculate ESF for the provided full reconstruction
        positions, esf_values = calculate_esf(all_angles_reconstruction)
        return positions, esf_values
    
    # If no reconstruction is provided, calculate one using all 360 angles
    print(f"Calculating ground truth using all 360 angles for slice {slice_idx}...")
    
    # Load all projections (using all available angles)
    projection_files = sorted(glob.glob(os.path.join('standard_dxrHD', "projection_*.npy")))
    all_angle_files = []
    
    for file_path in projection_files:
        base_name = os.path.basename(file_path)
        match = re.search(r'projection_(\d+\.\d+)\.npy', base_name)
        if match:
            angle = float(match.group(1))
            all_angle_files.append((angle, file_path))
    
    # Sort by angle
    all_angle_files.sort()
    
    # Get angles
    all_angles = np.array([angle for angle, _ in all_angle_files])
    
    # Create sinogram for this height using all angles
    height = slice_idx
    
    # First determine the width from the first projection
    first_proj = np.load(all_angle_files[0][1])
    first_proj = np.transpose(first_proj)
    _, width = first_proj.shape
    
    # Create sinogram with all angles
    sinogram = np.zeros((width, len(all_angles)))
    
    for i, (_, file_path) in enumerate(all_angle_files):
        proj = np.load(file_path)
        proj = np.transpose(proj)
        sinogram[:, i] = proj[height, :]
    
    # Reconstruct using FBP with all angles
    reconstruction = iradon(sinogram, theta=all_angles, filter_name='ramp',  #parameters of recontuction method
                           circle=True, interpolation='linear',
                           output_size=width)
    
    # Normalize to [0, 1]
    if np.max(reconstruction) > np.min(reconstruction):
        reconstruction = (reconstruction - np.min(reconstruction)) / (np.max(reconstruction) - np.min(reconstruction))
    
    # Calculate ESF for the full reconstruction
    positions, esf_values = calculate_esf(reconstruction)
    
    return positions, esf_values

def perform_fbp_reconstruction(angle_indices=None, gt_dir='ground_truth_esfs'):
    """
    Perform FBP reconstruction using specified angle indices and
    evaluate quality using ESF comparison against a ground truth
    reconstruction using all 360 angles.
    
    Args:
        angle_indices: List of indices to select which projections to use.
                       If None, all projections will be used.
        gt_dir: Directory containing pre-computed ground truth ESFs
                       
    Returns:
        tuple: (mean_error, std_error) - mean and standard deviation of ESF errors
    """
    # All the existing function code, but without the call to visualize_reconstruction_comparison
    # at the end, we'll move that to the anglesToImgQuality function
    
    # Configuration
    PROJECTION_DIR = 'standard_dxrHD'
    OUTPUT_DIR = 'outputs/HD_fbp_output'
    if angle_indices is not None:
        OUTPUT_DIR += f"_{len(angle_indices)}angles"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading and transposing projections...")
    
    # Get list of all .npy files
    projection_files = sorted(glob.glob(os.path.join(PROJECTION_DIR, "projection_*.npy")))
    
    if not projection_files:
        raise FileNotFoundError(f"No projection files found in {PROJECTION_DIR}")
    
    # Extract angles and sort files
    angle_files = []
    for file_path in projection_files:
        base_name = os.path.basename(file_path)
        match = re.search(r'projection_(\d+\.\d+)\.npy', base_name)
        if match:
            angle = float(match.group(1))
            angle_files.append((angle, file_path))
    
    # Sort by angle
    angle_files.sort()
    
    # If angle_indices is provided, select only those angles
    if angle_indices is not None:
        if max(angle_indices) >= len(angle_files):
            raise ValueError(f"Angle index {max(angle_indices)} is out of range (max index: {len(angle_files)-1})")
        selected_files = [angle_files[i] for i in angle_indices]
        angle_files = selected_files
    
    angles = np.array([angle for angle, _ in angle_files])
    
    # Load and transpose projections
    projections = []
    for angle, file_path in angle_files:
        proj = np.load(file_path)
        # Transpose the projection to correct orientation
        proj = np.transpose(proj)
        projections.append(proj)
    
    projections = np.array(projections)
    n_angles, height, width = projections.shape
    
    print(f"Loaded {len(projections)} projections with angles: {angles[0]:.1f}° to {angles[-1]:.1f}°")
    print(f"Projection dimensions after transpose: {height}x{width}")
    
    # Create directory for output files
    sinogram_dir = os.path.join(OUTPUT_DIR, "sinograms")
    slice_dir = os.path.join(OUTPUT_DIR, "slices")
    esf_dir = os.path.join(OUTPUT_DIR, "esf")
    os.makedirs(sinogram_dir, exist_ok=True)
    os.makedirs(slice_dir, exist_ok=True)
    os.makedirs(esf_dir, exist_ok=True)
    
    # Calculate center slice
    center_slice = height // 2
    
    # Generate 10 evenly spaced slices, including the center slice
    slice_positions = list(np.linspace(height // 10, height - height // 10, 9, dtype=int))
    if center_slice not in slice_positions:
        slice_positions.append(center_slice)
    slice_positions = sorted(slice_positions)[:10]  # Ensure we have at most 10 slices
    
    print(f"Generating reconstructions for slices: {slice_positions}")
    
    # First generate ground truth reconstructions using all angles (if not using specified angles)
    ground_truth_reconstructions = {}
    if angle_indices is not None and len(angle_indices) < 360:
        print("Generating ground truth reconstructions using all 360 angles...")
        # We'll generate these as needed in the loop below
    else:
        # If we're already using all angles, no need for separate ground truth
        print("Using current reconstructions as ground truth (already using all angles)")
    
    # Create reconstructions
    reconstructions = {}
    esf_errors = {}
    
    for h in slice_positions:
        # Create sinogram for this height (detector positions in rows, angles in columns)
        sinogram = np.zeros((width, n_angles))
        
        for i, proj in enumerate(projections):
            sinogram[:, i] = proj[h, :]
        
        # Save sinogram
        plt.figure(figsize=(10, 6))
        plt.imshow(sinogram, cmap='viridis', aspect='auto')
        plt.title(f"Sinogram at height {h}")
        plt.xlabel("Angle Index")
        plt.ylabel("Detector Position")
        plt.colorbar()
        plt.savefig(os.path.join(sinogram_dir, f"sinogram_h{h}.png"), dpi=150)
        plt.close()
        
        # Save raw sinogram as numpy array
        np.save(os.path.join(sinogram_dir, f"sinogram_h{h}.npy"), sinogram)
        
        # Reconstruct using FBP
        reconstruction = iradon(sinogram, theta=angles, filter_name='ramp', 
                               circle=True, interpolation='linear',
                               output_size=width)  # Match output size to avoid cropping
        
        # Normalize to [0, 1]
        reconstruction = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min())
        
        # Calculate ESF for this reconstruction
        esf_pos, esf_values = calculate_esf(reconstruction)
        
        # Load ground truth ESF from file - do not generate it on-the-fly
        if angle_indices is not None and len(angle_indices) < 360:
            # Check if the ground truth directory exists
            if not os.path.isdir(gt_dir):
                raise FileNotFoundError(f"Ground truth directory '{gt_dir}' not found. Please run generate_ground_truth_esfs.py first.")
            
            # Construct the filename for precomputed ground truth
            gt_file = os.path.join(gt_dir, f"gt_esf_slice_{h}.npy")
            
            # Check if the file exists
            if not os.path.exists(gt_file):
                raise FileNotFoundError(f"Ground truth ESF file for slice {h} not found at '{gt_file}'. Please run generate_ground_truth_esfs.py first.")
            
            # Load the precomputed ground truth
            try:
                gt_esf_pos, gt_esf_values = np.load(gt_file, allow_pickle=True)
                print(f"Loaded ground truth ESF for slice {h}")
            except Exception as e:
                raise Exception(f"Error loading ground truth ESF for slice {h}: {e}")
        else:
            # If we're already using all angles, use the same ESF as ground truth
            gt_esf_pos, gt_esf_values = esf_pos.copy(), esf_values.copy()
            # This will result in zero error as we're comparing against itself
        
        # Compare ESF to ground truth
        esf_error = compare_esf_to_ground_truth(esf_pos, esf_values, gt_esf_pos, gt_esf_values)
        esf_errors[h] = esf_error
        
        # Save reconstruction
        reconstructions[h] = reconstruction
        
        # Save as TIF file
        reconstruction_8bit = (reconstruction * 255).astype(np.uint8)
        imsave(os.path.join(slice_dir, f"slice_{h}.tif"), reconstruction_8bit)
        
        # Save ESF plot
        plt.figure(figsize=(10, 6))
        plt.plot(esf_pos, esf_values, 'b-', label='Calculated ESF')
        plt.plot(gt_esf_pos, gt_esf_values, 'r--', label='Ground Truth ESF')
        plt.title(f"ESF Comparison for Slice {h} (RMSE: {esf_error:.4f})")
        plt.xlabel("Position")
        plt.ylabel("ESF Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(esf_dir, f"esf_comparison_h{h}.png"), dpi=150)
        plt.close()
        
        # Save ESF data
        np.save(os.path.join(esf_dir, f"esf_h{h}.npy"), (esf_pos, esf_values))
        
        # Report error
        print(f"Slice {h}: ESF RMSE = {esf_error:.4f}")
    
    # Calculate mean and standard deviation of errors
    error_values = list(esf_errors.values())
    mean_error = np.mean(error_values)
    std_error = np.std(error_values)
    
    print(f"Mean ESF Error: {mean_error:.4f}")
    print(f"Std Dev of ESF Error: {std_error:.4f}")
    
    # Create comparison visualization of slices with errors
    plt.figure(figsize=(15, 10))
    
    # Select 5 evenly spaced slices from our reconstructions for display
    display_slices = slice_positions[::max(1, len(slice_positions)//5)][:5]
    
    for i, h in enumerate(display_slices):
        plt.subplot(2, 5, i+1)
        plt.imshow(reconstructions[h], cmap='gray')
        plt.title(f"Slice {h}")
        plt.axis('off')
        
        # Plot ESF for this slice
        plt.subplot(2, 5, i+6)
        plt.plot(esf_pos, esf_values, 'b-', label='ESF')
        plt.plot(gt_esf_pos, gt_esf_values, 'r--', label='Ground Truth')
        plt.title(f"RMSE: {esf_errors[h]:.4f}")
        plt.xlabel("Position")
        plt.ylabel("ESF Value")
        plt.grid(True)
        if i == 0:  # Only show legend for first subplot
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "slice_esf_comparison.png"), dpi=150)
    plt.close()
    
    # Save errors to a CSV file
    try:
        import pandas as pd
        errors_df = pd.DataFrame.from_dict(esf_errors, orient='index', columns=['ESF_RMSE'])
        errors_df.index.name = 'Slice'
        errors_df.to_csv(os.path.join(OUTPUT_DIR, "esf_errors.csv"))
        print(f"ESF errors saved to {os.path.join(OUTPUT_DIR, 'esf_errors.csv')}")
    except ImportError:
        # If pandas is not available, save as simple text file
        with open(os.path.join(OUTPUT_DIR, "esf_errors.txt"), "w") as f:
            f.write("Slice,ESF_RMSE\n")
            for h in sorted(esf_errors.keys()):
                f.write(f"{h},{esf_errors[h]:.4f}\n")
    
    print(f"Reconstruction complete. Results saved to {OUTPUT_DIR}")
    
    return mean_error, std_error, reconstructions, slice_positions, OUTPUT_DIR

def anglesToImgQuality(inputAngles, generate_visualizations=True, gt_dir='ground_truth_esfs'):
    """
    Main function to assess image quality based on input angles.
    Uses pre-computed ground truth ESFs from the specified directory.
    
    Args:
        inputAngles: List of angles to use for reconstruction
        generate_visualizations: Whether to generate comparison visualizations
        gt_dir: Directory containing pre-computed ground truth ESFs
        
    Returns:
        tuple: (mean_error, std_error) - mean and standard deviation of ESF errors
        
    Raises:
        FileNotFoundError: If the ground truth directory or required files don't exist
    """
    # Check if ground truth directory exists
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"Ground truth directory '{gt_dir}' not found. Please run generate_ground_truth_esfs.py first.")
        
    # Convert input angles to indices if necessary
    # This assumes angles are at 1-degree intervals (0, 1, 2, ...)
    angle_indices = inputAngles
    
    # If inputAngles are actual angle values (not indices), convert to indices
    if any(angle > 359 for angle in inputAngles):
        # Assuming angles are stored at 1-degree intervals in the dataset
        angle_indices = [int(angle) % 360 for angle in inputAngles]
    
    # Perform reconstruction and get quality metrics
    mean_error, std_error, reconstructions, slice_positions, output_dir = perform_fbp_reconstruction(
        angle_indices, gt_dir=gt_dir)
    
 
    
    return mean_error, std_error

# For testing the function independently
#if __name__ == "__main__":
#    # Example usage with angles at 10-degree intervals
#    test_angles = list(range(0, 360, 10))
#    mean_error, std_error = anglesToImgQuality(test_angles)
#    print(f"Test Results: Mean Error = {mean_error:.4f}, Std Error = {std_error:.4f}")