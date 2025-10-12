"""
GUI Classes for Breast Ultrasound AI Application
Main window, image viewer, and results panel components
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import threading
from typing import Dict, Optional, Any

# Import our backend classes
from ml_backend import MLBackend
from image_processor import ImageProcessor, ResultProcessor


class ImageViewer:
    """Handle image display and visualization"""
    
    def __init__(self, parent_canvas):
        """
        Initialize image viewer
        Args:
            parent_canvas: tkinter Canvas widget for display
        """
        self.canvas = parent_canvas
        self.current_image = None
        self.current_photo = None
        self.overlay_image = None
        self.zoom_factor = 1.0
        self.overlay_visible = True
        self.original_image = None
        
        # Bind canvas resize event
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        
    def display_image(self, image_path: str) -> bool:
        """
        Display image from file path
        Args:
            image_path: Path to image file
        Returns:
            Success status
        """
        try:
            # Load image using ImageProcessor
            image = ImageProcessor.load_image(image_path)
            if image is None:
                return False
            
            self.original_image = image
            self.current_image = image
            self.overlay_image = None
            
            # Display on canvas
            self._update_canvas_display()
            return True
            
        except Exception as e:
            print(f"Error displaying image: {e}")
            return False
    
    def display_overlay(self, image: np.ndarray, mask: np.ndarray):
        """
        Display image with segmentation overlay
        Args:
            image: Original image
            mask: Segmentation mask
        """
        try:
            if mask is not None:
                # Generate overlay using ResultProcessor
                self.overlay_image = ResultProcessor.generate_overlay(image, mask)
            else:
                self.overlay_image = None
            
            self.original_image = image
            self._update_current_image()
            self._update_canvas_display()
            
        except Exception as e:
            print(f"Error displaying overlay: {e}")
    
    def toggle_overlay(self):
        """Toggle overlay visibility"""
        self.overlay_visible = not self.overlay_visible
        self._update_current_image()
        self._update_canvas_display()
    
    def zoom_in(self):
        """Zoom in on image"""
        self.zoom_factor *= 1.2
        self._update_canvas_display()
    
    def zoom_out(self):
        """Zoom out on image"""
        self.zoom_factor /= 1.2
        self._update_canvas_display()
    
    def reset_view(self):
        """Reset zoom to fit canvas"""
        self.zoom_factor = 1.0
        self._update_canvas_display()
    
    def _update_current_image(self):
        """Update current displayed image based on overlay settings"""
        if self.overlay_visible and self.overlay_image is not None:
            self.current_image = self.overlay_image
        elif self.original_image is not None:
            self.current_image = self.original_image
    
    def _update_canvas_display(self):
        """Update canvas with current image"""
        if self.current_image is None:
            return
        
        try:
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            # Calculate scaling
            img_height, img_width = self.current_image.shape[:2]
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            scale = min(scale_x, scale_y) * self.zoom_factor
            
            # Resize image
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            if new_width > 0 and new_height > 0:
                # Use ImageProcessor for resizing
                resized_image = ImageProcessor.resize_image(
                    self.current_image, (new_width, new_height)
                )
                
                # Convert to PIL and then PhotoImage
                pil_image = Image.fromarray(resized_image)
                self.current_photo = ImageTk.PhotoImage(pil_image)
                
                # Display on canvas
                self.canvas.delete("all")
                x = canvas_width // 2
                y = canvas_height // 2
                self.canvas.create_image(x, y, image=self.current_photo)
                
        except Exception as e:
            print(f"Error updating canvas display: {e}")
    
    def _on_canvas_resize(self, event):
        """Handle canvas resize event"""
        self._update_canvas_display()


class ResultsPanel:
    """Display analysis results and measurements"""
    
    def __init__(self, parent_frame):
        """
        Initialize results panel
        Args:
            parent_frame: Parent tkinter frame
        """
        self.frame = parent_frame
        self.setup_results_widgets()
    
    def setup_results_widgets(self):
        """Create results display widgets"""
        # Main results text area
        self.results_text = tk.Text(
            self.frame, height=12, wrap="word",
            font=("Segoe UI", 10), relief="solid", bd=1,
            state="disabled"
        )
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(
            self.frame, orient="vertical",
            command=self.results_text.yview
        )
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def display_results(self, results: Dict[str, Any]):
        """
        Display complete analysis results
        Args:
            results: Analysis results from MLBackend
        """
        try:
            self.clear_results()
            
            # Display classification results
            if 'classification' in results:
                self._display_classification(results['classification'])
            
            # Display segmentation results
            if 'segmentation' in results:
                self._display_segmentation(results['segmentation'])
            
            # Display summary and recommendations
            if 'summary' in results:
                self._display_summary(results['summary'])
            
            # Display processing info
            self._display_processing_info(results)
            
        except Exception as e:
            print(f"Error displaying results: {e}")
            self._append_text(f"Error displaying results: {e}")
    
    def _display_classification(self, classification: Dict[str, Any]):
        """Display classification results"""
        lines = []
        lines.append("=== CLASSIFICATION ANALYSIS ===")
        lines.append(f"Prediction: {classification.get('prediction', 'Unknown').upper()}")
        lines.append(f"Confidence: {classification.get('confidence', 0)*100:.1f}%")
        lines.append("")
        
        # Show all class probabilities
        probabilities = classification.get('probabilities', {})
        if probabilities:
            lines.append("Class Probabilities:")
            for class_name, prob in probabilities.items():
                lines.append(f"  {class_name.title()}: {prob*100:.1f}%")
        
        lines.append("")
        self._append_text("\\n".join(lines))
    
    def _display_segmentation(self, segmentation: Dict[str, Any]):
        """Display segmentation results"""
        lines = []
        lines.append("=== TUMOR SEGMENTATION ===")
        lines.append(f"Tumor Detected: {'Yes' if segmentation.get('tumor_detected', False) else 'No'}")
        
        if segmentation.get('tumor_detected', False):
            area = segmentation.get('tumor_area_pixels', 0)
            lines.append(f"Tumor Area: {area} pixels")
        
        lines.append("")
        self._append_text("\\n".join(lines))
    
    def _display_summary(self, summary: Dict[str, Any]):
        """Display analysis summary and recommendations"""
        lines = []
        lines.append("=== ANALYSIS SUMMARY ===")
        lines.append(f"Risk Level: {summary.get('risk_level', 'Unknown')}")
        lines.append(f"Overall Confidence: {summary.get('overall_confidence', 0)*100:.1f}%")
        lines.append(f"Follow-up Required: {'Yes' if summary.get('requires_followup', False) else 'No'}")
        lines.append("")
        
        # Display recommendations
        recommendations = summary.get('recommendations', [])
        if recommendations:
            lines.append("RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"{i}. {rec}")
        
        lines.append("")
        self._append_text("\\n".join(lines))
    
    def _display_processing_info(self, results: Dict[str, Any]):
        """Display processing information"""
        lines = []
        lines.append("=== PROCESSING INFO ===")
        lines.append(f"Processing Time: {results.get('processing_time', 0):.2f} seconds")
        lines.append(f"Timestamp: {results.get('timestamp', 'Unknown')}")
        
        # Model status
        models_status = results.get('models_status', {})
        lines.append(f"Classification Model: {'Loaded' if models_status.get('classification_loaded', False) else 'Demo Mode'}")
        lines.append(f"Segmentation Model: {'Loaded' if models_status.get('segmentation_loaded', False) else 'Demo Mode'}")
        
        self._append_text("\\n".join(lines))
    
    def clear_results(self):
        """Clear all displayed results"""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state="disabled")
    
    def _append_text(self, text: str):
        """Append text to results display"""
        self.results_text.config(state="normal")
        self.results_text.insert(tk.END, text)
        self.results_text.see(tk.END)
        self.results_text.config(state="disabled")


class MainWindow:
    """Main application window and controller"""
    
    def __init__(self):
        """Initialize main window and all components"""
        # Create main window to match gui_application
        self.window = tk.Tk()
        self.window.title("Breast Ultrasound Detection")
        self.window.geometry("1400x800")
        self.window.configure(bg="#f4f6f9")
        
        # Initialize ML backend with model paths
        # SPECIFY YOUR MODEL PATHS HERE:
        classification_model_path = "breast_cancer_model_no_finetune.h5"  # Updated path
        segmentation_model_path = "unet_resnet34_best.pth"  # Updated path
        
        self.ml_backend = MLBackend(
            classification_model_path=classification_model_path,
            segmentation_model_path=segmentation_model_path
        )
        
        # Application state
        self.current_image_path = None
        self.current_results = None
        self.is_analyzing = False
        
        # Setup UI
        self.setup_styles()
        self.create_layout()
        self.create_menu()
        
        # Initialize backend
        self._initialize_backend()
    
    def setup_styles(self):
        """Configure ttk styles"""
        self.style = ttk.Style(self.window)
        self.style.theme_use("clam")
        
        # Configure styles
        self.style.configure("TLabel", font=("Segoe UI", 11))
        self.style.configure("TButton", font=("Segoe UI", 10), padding=5)
        self.style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))
        self.style.configure("TLabelframe.Label", font=("Segoe UI", 12, "bold"))
    
    def create_layout(self):
        """Create main application layout"""
        # Configure main grid
        self.window.grid_columnconfigure(1, weight=1)
        self.window.grid_rowconfigure(0, weight=1)
        
        # Create main panels
        self._create_control_panel()
        self._create_main_content()
    
    def _create_control_panel(self):
        """Create left control panel"""
        control_frame = ttk.LabelFrame(
            self.window, text="Controls", padding=12
        )
        control_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        control_frame.configure(width=300)
        
        # Title
        title_label = ttk.Label(
            control_frame, text="Breast Ultrasound\nAI Analysis",
            font=("Segoe UI", 14, "bold"), justify="center"
        )
        title_label.pack(pady=(0, 20))
        
        # Load image button
        self.load_btn = ttk.Button(
            control_frame, text="Load Ultrasound Image",
            command=self.load_image, style="Primary.TButton"
        )
        self.load_btn.pack(fill="x", pady=(0, 10))
        
        # Analyze button
        self.analyze_btn = ttk.Button(
            control_frame, text="Analyze Image",
            command=self.analyze_image, style="Primary.TButton"
        )
        self.analyze_btn.pack(fill="x", pady=(0, 10))
        self.analyze_btn.config(state="disabled")
        
        # Save results button
        self.save_btn = ttk.Button(
            control_frame, text="Save Results",
            command=self.save_results
        )
        self.save_btn.pack(fill="x", pady=(0, 20))
        self.save_btn.config(state="disabled")
        
        # Status label
        self.status_var = tk.StringVar(value="Ready - Load an image to begin")
        status_label = ttk.Label(control_frame, textvariable=self.status_var, 
                                wraplength=250, justify="center")
        status_label.pack(pady=(0, 10))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress_bar.pack(fill="x", pady=(0, 20))
        
        # Image controls
        controls_frame = ttk.LabelFrame(control_frame, text="Image Controls", padding=8)
        controls_frame.pack(fill="x", pady=(0, 10))
        
        # Zoom controls
        zoom_frame = ttk.Frame(controls_frame)
        zoom_frame.pack(fill="x", pady=(0, 5))
        
        ttk.Button(zoom_frame, text="Zoom In", 
                  command=self._zoom_in).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(zoom_frame, text="Zoom Out", 
                  command=self._zoom_out).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(zoom_frame, text="Reset", 
                  command=self._reset_view).pack(side="left", fill="x", expand=True, padx=(2, 0))
        
        # Overlay toggle
        self.overlay_btn = ttk.Button(controls_frame, text="Toggle Overlay", 
                                     command=self._toggle_overlay)
        self.overlay_btn.pack(fill="x")
        self.overlay_btn.config(state="disabled")
    
    def _create_main_content(self):
        """Create main content area"""
        main_frame = ttk.Frame(self.window)
        main_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=2)  # Image area
        main_frame.grid_rowconfigure(1, weight=1)  # Results area
        
        # Image display area
        image_frame = ttk.LabelFrame(main_frame, text="Ultrasound Image", padding=8)
        image_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        image_frame.grid_columnconfigure(0, weight=1)
        image_frame.grid_rowconfigure(0, weight=1)
        
        self.image_canvas = tk.Canvas(image_frame, bg="white", relief="sunken", bd=2)
        self.image_canvas.grid(row=0, column=0, sticky="nsew")
        
        # Create image viewer
        self.image_viewer = ImageViewer(self.image_canvas)
        
        # Results display area
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding=8)
        results_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        
        # Create results panel
        self.results_panel = ResultsPanel(results_frame)
        
        # Create direct reference to results text for easier access
        self.results_text = self.results_panel.results_text
        

    
    def create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self.window)
        self.window.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.window.quit)
        
        # Models menu
        models_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Models", menu=models_menu)
        models_menu.add_command(label="Load Classification Model", 
                               command=self._load_classification_model)
        models_menu.add_command(label="Load Segmentation Model", 
                               command=self._load_segmentation_model)
        models_menu.add_separator()
        models_menu.add_command(label="Model Status", command=self._show_model_status)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Overlay", command=self._toggle_overlay)
        view_menu.add_command(label="Zoom In", command=self._zoom_in)
        view_menu.add_command(label="Zoom Out", command=self._zoom_out)
        view_menu.add_command(label="Reset View", command=self._reset_view)
    
    def _initialize_backend(self):
        """Initialize ML backend in background"""
        def init_backend():
            try:
                self.ml_backend.initialize()
                self.window.after(0, lambda: self.status_var.set("Ready - Backend initialized"))
            except Exception as e:
                self.window.after(0, lambda: self.status_var.set(f"Backend error: {e}"))
        
        # Initialize in background thread
        thread = threading.Thread(target=init_backend)
        thread.daemon = True
        thread.start()
    
    def load_image(self):
        """Load ultrasound image file"""
        file_path = filedialog.askopenfilename(
            title="Select Ultrasound Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            # Load and display image
            try:
                # Load image using OpenCV
                image = cv2.imread(file_path)
                if image is None:
                    messagebox.showerror("Error", "Failed to load image file")
                    return
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Get canvas size and resize image to fit
                canvas_width = self.image_canvas.winfo_width()
                canvas_height = self.image_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:  # Canvas is initialized
                    # Calculate scaling to fit canvas while maintaining aspect ratio
                    h, w = image_rgb.shape[:2]
                    scale = min(canvas_width/w, canvas_height/h)
                    new_w, new_h = int(w*scale), int(h*scale)
                    
                    resized_image = cv2.resize(image_rgb, (new_w, new_h))
                    
                    # Convert to PIL and then to PhotoImage
                    pil_image = Image.fromarray(resized_image)
                    self.current_photo = ImageTk.PhotoImage(pil_image)
                    
                    # Clear canvas and display image
                    self.image_canvas.delete("all")
                    
                    # Center image on canvas
                    x_offset = (canvas_width - new_w) // 2
                    y_offset = (canvas_height - new_h) // 2
                    
                    self.image_canvas.create_image(x_offset, y_offset, anchor="nw", image=self.current_photo)
                
                # Update UI state
                self.current_image_path = file_path
                self.analyze_btn.config(state="normal")
                filename = os.path.basename(file_path)
                self.status_var.set(f"Image loaded: {filename}")
                
                # Clear previous results
                self.results_text.config(state="normal")
                self.results_text.delete(1.0, tk.END)
                self.results_text.config(state="disabled")
                self.current_results = None
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def analyze_image(self):
        """Analyze the loaded image"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if self.is_analyzing:
            return
        
        # Start analysis
        self.is_analyzing = True
        self.analyze_btn.config(state="disabled")
        self.progress_bar.start()
        self.status_var.set("Analyzing image...")
        
        # Run analysis in background thread
        thread = threading.Thread(target=self._run_analysis)
        thread.daemon = True
        thread.start()
    
    def _run_analysis(self):
        """Run analysis in background thread"""
        try:
            # Perform analysis using ML backend
            results = self.ml_backend.analyze_image(self.current_image_path)
            
            # Update UI in main thread
            self.window.after(0, lambda: self._display_analysis_results(results))
            
        except Exception as e:
            error_msg = f"Analysis failed: {e}"
            self.window.after(0, lambda: messagebox.showerror("Error", error_msg))
        finally:
            self.window.after(0, self._analysis_complete)
    
    def _display_analysis_results(self, results: Dict[str, Any]):
        """Display analysis results in UI matching gui_application format"""
        try:
            self.current_results = results
            
            # Display results in text widget
            self.results_text.config(state="normal")
            self.results_text.delete(1.0, tk.END)
            
            # Format results matching gui_application
            result_text = "=== AI ANALYSIS RESULTS ===\n"
            result_text += f"Analysis Date: {results.get('timestamp', 'Unknown')}\n\n"
            
            # Classification results
            classification = results.get('classification', {})
            if classification:
                prediction = classification.get('prediction', 'Unknown')
                confidence = classification.get('confidence', 0) * 100
                result_text += f"CLASSIFICATION: {prediction.upper()} ({confidence:.1f}%)\n"
            
            # Segmentation results
            segmentation = results.get('segmentation', {})
            if segmentation:
                tumor_detected = segmentation.get('tumor_detected', False)
                tumor_area = segmentation.get('tumor_area_pixels', 0)
                result_text += f"Tumor detected: {'Yes' if tumor_detected else 'No'}\n"
                if tumor_detected and tumor_area > 0:
                    result_text += f"Tumor area: {tumor_area} pixels\n"
            
            # Processing time
            proc_time = results.get('processing_time', 0)
            result_text += f"\nAnalysis complete in {proc_time:.1f}s\n"
            
            # Add to results display
            self.results_text.insert(tk.END, result_text)
            self.results_text.config(state="disabled")
            

            
            # Update image with overlay if segmentation available
            if segmentation.get('mask') is not None:
                # Load original image
                image = ImageProcessor.load_image(self.current_image_path)
                if image is not None:
                    mask = segmentation['mask']
                    self.image_viewer.display_overlay(image, mask)
                    self.overlay_btn.config(state="normal")
            
            # Enable save button
            self.save_btn.config(state="normal")
            self.status_var.set("Analysis complete")
            
        except Exception as e:
            print(f"Error displaying results: {e}")
            messagebox.showerror("Error", f"Failed to display results: {e}")
    
    def _analysis_complete(self):
        """Clean up after analysis"""
        self.is_analyzing = False
        self.analyze_btn.config(state="normal")
        self.progress_bar.stop()
        self.status_var.set("Analysis complete")
    
    def save_results(self):
        """Save analysis results to file"""
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Analysis Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Use ResultProcessor to save formatted report
                from image_processor import FileManager
                success = FileManager.save_analysis_report(self.current_results, file_path)
                
                if success:
                    messagebox.showinfo("Success", "Results saved successfully")
                else:
                    messagebox.showerror("Error", "Failed to save results")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {e}")
    
    # Image control methods
    def _zoom_in(self):
        """Zoom in on image"""
        self.image_viewer.zoom_in()
    
    def _zoom_out(self):
        """Zoom out on image"""
        self.image_viewer.zoom_out()
    
    def _reset_view(self):
        """Reset image view"""
        self.image_viewer.reset_view()
    
    def _toggle_overlay(self):
        """Toggle segmentation overlay"""
        self.image_viewer.toggle_overlay()
    
    # Model management methods
    def _load_classification_model(self):
        """Load classification model file"""
        file_path = filedialog.askopenfilename(
            title="Select Classification Model",
            filetypes=[("Keras models", "*.h5"), ("All files", "*.*")]
        )
        
        if file_path:
            if self.ml_backend.load_classification_model(file_path):
                messagebox.showinfo("Success", "Classification model loaded successfully")
            else:
                messagebox.showerror("Error", "Failed to load classification model")
    
    def _load_segmentation_model(self):
        """Load segmentation model file"""
        file_path = filedialog.askopenfilename(
            title="Select Segmentation Model",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
        )
        
        if file_path:
            if self.ml_backend.load_segmentation_model(file_path):
                messagebox.showinfo("Success", "Segmentation model loaded successfully")
            else:
                messagebox.showerror("Error", "Failed to load segmentation model")
    
    def _show_model_status(self):
        """Show current model status"""
        status = self.ml_backend.get_model_status()
        
        status_text = f"""Model Status:

Classification Model:
- Loaded: {status.get('classification_loaded', False)}
- Path: {status.get('classification_info', {}).get('model_path', 'Not set')}

Segmentation Model:
- Loaded: {status.get('segmentation_loaded', False)} 
- Path: {status.get('segmentation_info', {}).get('model_path', 'Not set')}
- Device: {status.get('segmentation_info', {}).get('device', 'N/A')}

Backend Ready: {status.get('backend_ready', False)}
"""
        
        messagebox.showinfo("Model Status", status_text)
    
    def run(self):
        """Start the application"""
        self.window.mainloop()


# Main application entry point
def main():
    """Create and run the application"""
    try:
        app = MainWindow()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Application Error", f"Failed to start application: {e}")


if __name__ == "__main__":
    main()