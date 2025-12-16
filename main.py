#!/usr/bin/env python3
"""
Main script for ice shelf flow regime classification pipeline.

This script runs the complete workflow:
1. Use existing processed dataset (or guide you to create one)
2. K-value selection with interactive GUI
3. K-means classification with results display

Usage:
    python main.py [options]
    
Note: Requires existing processed dataset in data/real_data_analysis/processed_dataset.npz
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_data_extraction(diffice_data, viscosity_data, output_dir):
    """Run data extraction and preprocessing."""
    print("Step 1: Data extraction and preprocessing...")
    
    # Check if processed dataset already exists
    existing_processed = Path("data/real_data_analysis/processed_dataset.npz")
    output_file = output_dir / 'processed_dataset.npz'
    
    if existing_processed.exists():
        print(f"Using existing processed dataset: {existing_processed}")
        print("✓ No copy needed - using dataset directly")
        return existing_processed
    else:
        print("✗ No processed dataset found.")
        print(f"Please ensure {existing_processed} exists, or run data preprocessing separately.")
        print("You can create a processed dataset using the individual scripts:")
        print("1. Process your raw data first")
        print("2. Or use the existing processed dataset in data/real_data_analysis/")
        raise FileNotFoundError(f"Processed dataset not found at {existing_processed}")

def run_k_selection(processed_data_path, output_dir):
    """Run k-selection analysis and return the plot path."""
    print("Step 2: Running k-value selection analysis...")
    
    k_selection_dir = output_dir / 'k_select'
    k_selection_dir.mkdir(exist_ok=True)
    
    # Run k-selection script
    cmd = [
        sys.executable, 'scripts/run_k_selection.py',
        '--diffice_data', 'dummy',  # Not used when processed_data provided
        '--viscosity_data', 'dummy',  # Not used when processed_data provided  
        '--processed_data', str(processed_data_path),
        '--output_dir', str(k_selection_dir),
        '--k_range', '2', '8'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ K-selection analysis completed")
        
        # Return path to the k-selection plot
        plot_path = k_selection_dir / 'k_selection_analysis.png'
        return plot_path
        
    except subprocess.CalledProcessError as e:
        print(f"✗ K-selection failed: {e}")
        print(f"Error output: {e.stderr}")
        raise

def show_k_selection_gui(plot_path):
    """Show k-selection plot in GUI and get user input for k value."""
    
    class KSelectionWindow:
        def __init__(self, plot_path):
            self.k_value = None
            self.root = tk.Tk()
            self.root.title("K-Value Selection")
            self.root.geometry("1000x700")
            
            # Bring window to front and make it topmost temporarily
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.after_idle(lambda: self.root.attributes('-topmost', False))
            self.root.focus_force()
            
            # Main frame with smaller padding
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Title with larger font
            title_label = ttk.Label(main_frame, text="K-Value Selection Analysis", 
                                  font=('Arial', 18, 'bold'))
            title_label.pack(pady=(0, 10))
            
            # Instructions with larger font
            instructions = ttk.Label(main_frame, 
                text="Examine the plots below:\n" +
                     "• Elbow Method: Look for the 'elbow' point where the curve flattens\n" +
                     "• Silhouette Analysis: Look for the highest peak\n" +
                     "• Recommended k=3 for ice shelf flow regimes (compression, transition, extension)",
                justify=tk.LEFT, font=('Arial', 12))
            instructions.pack(pady=(0, 10))
            
            # Load and display properly sized image
            try:
                img = Image.open(plot_path)
                # Keep original proportions, but limit to reasonable size for GUI
                original_width, original_height = img.size
                max_width = 800  # Smaller to fit better
                max_height = 400  # Limit height to leave room for buttons
                
                # Calculate scale factor to fit within both width and height limits
                scale_w = max_width / original_width
                scale_h = max_height / original_height
                scale_factor = min(scale_w, scale_h, 1.0)  # Don't upscale
                
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(img)
                
                img_label = ttk.Label(main_frame, image=self.photo)
                img_label.pack(pady=(0, 10))
            except Exception as e:
                error_label = ttk.Label(main_frame, text=f"Could not load plot: {e}")
                error_label.pack(pady=(0, 10))
            
            # Input frame with larger text
            input_frame = ttk.Frame(main_frame)
            input_frame.pack(pady=(0, 10))
            
            ttk.Label(input_frame, text="Select k value:", font=('Arial', 14)).pack(side=tk.LEFT, padx=(0, 10))
            
            self.k_var = tk.StringVar(value="3")
            k_spinbox = ttk.Spinbox(input_frame, from_=2, to=8, textvariable=self.k_var, width=8, font=('Arial', 12))
            k_spinbox.pack(side=tk.LEFT, padx=(0, 15))
            
            # Buttons with larger text
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(pady=10)
            
            continue_btn = ttk.Button(button_frame, text="Continue with Selected k", 
                                     command=self.on_continue)
            continue_btn.pack(side=tk.LEFT, padx=(0, 10))
            
            cancel_btn = ttk.Button(button_frame, text="Cancel", 
                                  command=self.on_cancel)
            cancel_btn.pack(side=tk.LEFT)
            
            # Add keyboard shortcuts
            self.root.bind('<Return>', lambda event: self.on_continue())
            self.root.bind('<Escape>', lambda event: self.on_cancel())
            
            # Set focus to the continue button by default
            continue_btn.focus_set()
        
        def on_continue(self):
            try:
                self.k_value = int(self.k_var.get())
                if 2 <= self.k_value <= 8:
                    # Immediately hide the window and force updates
                    self.root.withdraw()  # Hide window immediately
                    self.root.update()    # Force immediate update
                    self.root.quit()      # Exit mainloop
                else:
                    messagebox.showerror("Invalid k", "Please select k between 2 and 8")
            except ValueError:
                messagebox.showerror("Invalid k", "Please enter a valid integer for k")
        
        def on_cancel(self):
            self.k_value = None  # Make sure k_value is None for cancellation
            self.root.withdraw()  # Hide window immediately
            self.root.update()    # Force immediate update
            self.root.quit()      # Exit mainloop
        
        def get_k_value(self):
            self.root.mainloop()
            # Destroy the window properly after mainloop exits
            try:
                self.root.destroy()
            except:
                pass  # Window might already be destroyed
            return self.k_value
    
    window = KSelectionWindow(plot_path)
    return window.get_k_value()

def run_kmeans_classification(processed_data_path, k_value, output_dir):
    """Run k-means classification with the selected k value."""
    print(f"Step 3: Running k-means classification with k={k_value}...")
    
    classification_dir = output_dir / f'k_classify'
    classification_dir.mkdir(exist_ok=True)
    
    # Run optimized k-means script
    cmd = [
        sys.executable, 'scripts/run_optimized_kmeans.py',
        '--processed_data', str(processed_data_path),
        '--output_dir', str(classification_dir),
        '--k', str(k_value)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ K-means classification completed")
        
        # Return path to the spatial plot
        spatial_plot_path = classification_dir / 'spatial_ice_shelf_regimes.png'
        return spatial_plot_path, classification_dir
        
    except subprocess.CalledProcessError as e:
        print(f"✗ K-means classification failed: {e}")
        print(f"Error output: {e.stderr}")
        raise

def show_results_gui(spatial_plot_path, results_dir, k_value):
    """Show classification results in GUI."""
    
    class ResultsWindow:
        def __init__(self, spatial_plot_path, results_dir, k_value):
            self.root = tk.Tk()
            self.root.title(f"Classification Results (k={k_value})")
            self.root.geometry("1000x750")
            
            # Bring window to front
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.after_idle(lambda: self.root.attributes('-topmost', False))
            self.root.focus_force()
            
            # Main frame with smaller padding
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Title with larger font
            title_label = ttk.Label(main_frame, 
                text=f"Ice Shelf Flow Regime Classification Results (k={k_value})", 
                font=('Arial', 18, 'bold'))
            title_label.pack(pady=(0, 10))
            
            # Results info with larger font
            info_text = f"Classification completed successfully!\n" + \
                       f"All results saved to: {results_dir}\n" + \
                       f"Generated plots: spatial map, feature analysis, PCA, centroids, distributions, silhouette analysis"
            
            info_label = ttk.Label(main_frame, text=info_text, justify=tk.LEFT, font=('Arial', 12))
            info_label.pack(pady=(0, 10))
            
            # Load and display properly sized spatial plot
            try:
                img = Image.open(spatial_plot_path)
                # Keep original proportions, but limit to fit in window with room for buttons
                original_width, original_height = img.size
                max_width = 800  # Smaller to fit better
                max_height = 450  # Limit height to leave room for buttons
                
                # Calculate scale factor to fit within both width and height limits
                scale_w = max_width / original_width
                scale_h = max_height / original_height
                scale_factor = min(scale_w, scale_h, 1.0)  # Don't upscale
                
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(img)
                
                img_label = ttk.Label(main_frame, image=self.photo)
                img_label.pack(pady=(0, 10))
            except Exception as e:
                error_label = ttk.Label(main_frame, text=f"Could not load spatial plot: {e}")
                error_label.pack(pady=(0, 10))
            
            # Buttons with larger text
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(pady=10)
            
            open_btn = ttk.Button(button_frame, text="Open Results Directory", 
                                 command=lambda: self.open_directory(results_dir))
            open_btn.pack(side=tk.LEFT, padx=(0, 10))
            
            close_btn = ttk.Button(button_frame, text="Close", command=self.close)
            close_btn.pack(side=tk.LEFT)
        
        def open_directory(self, directory):
            """Open the results directory in file explorer."""
            import platform
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(directory)])
            elif platform.system() == "Windows":
                subprocess.run(["explorer", str(directory)])
            else:  # Linux
                subprocess.run(["xdg-open", str(directory)])
        
        def close(self):
            self.root.quit()
            self.root.destroy()
        
        def show(self):
            self.root.mainloop()
    
    window = ResultsWindow(spatial_plot_path, results_dir, k_value)
    window.show()

def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description='Complete ice shelf flow regime classification pipeline'
    )
    parser.add_argument('--diffice_data', default='data/DIFFICE_jax',
                       help='Path to DIFFUSE data directory (default: data/DIFFICE_jax)')
    parser.add_argument('--viscosity_data', default='data/raw/results.mat',
                       help='Path to viscosity results.mat file (default: data/raw/results.mat)')
    parser.add_argument('--output_dir', default='results',
                       help='Output directory for all results (default: results)')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Ice Shelf Flow Regime Classification Pipeline")
    print("=" * 50)
    print(f"DIFFUSE data: {args.diffice_data}")
    print(f"Viscosity data: {args.viscosity_data}")
    print(f"Output directory: {output_dir}")
    print()
    
    try:
        # Step 1: Data extraction
        processed_data_path = run_data_extraction(
            args.diffice_data, 
            args.viscosity_data, 
            output_dir
        )
        
        # Step 2: K-selection analysis
        k_plot_path = run_k_selection(processed_data_path, output_dir)
        
        # Step 3: Interactive k-selection
        print("Step 2.1: Opening k-selection GUI...")
        k_value = show_k_selection_gui(k_plot_path)
        
        if k_value is None:
            print("Pipeline cancelled by user.")
            return
        
        print(f"User selected k={k_value}")
        print("Starting classification...")
        
        # Step 4: K-means classification
        spatial_plot_path, results_dir = run_kmeans_classification(
            processed_data_path, k_value, output_dir
        )
        
        # Step 5: Show results
        print("Step 4: Opening results GUI...")
        show_results_gui(spatial_plot_path, results_dir, k_value)
        
        print()
        print("Pipeline completed successfully!")
        print(f"All results available in: {output_dir}")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()