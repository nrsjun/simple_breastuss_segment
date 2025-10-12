"""
main file for app
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

sys.path.insert(0, os.path.dirname(__file__))

try:
    from gui_classes import MainWindow
    from data_manager import ConfigManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required files are in the same directory")
    sys.exit(1)


def main():
    """Main application entry point"""
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Create and run application
        app = MainWindow()
        
        # Apply window settings from config
        ui_settings = config.get('ui', {})
        width = ui_settings.get('window_width', 1400)
        height = ui_settings.get('window_height', 900)
        app.window.geometry(f"{width}x{height}")
        
        print("Starting Breast Ultrasound AI Application...")
        print("=" * 50)
        print("Application Features:")
        print("- Load ultrasound images (any size)")
        print("- AI classification (Benign/Malignant)")  
        print("- AI tumor segmentation")
        print("- Patient data management")
        print("- Analysis results export")
        print("=" * 50)
        
        # Run the application
        app.run()
        
    except Exception as e:
        error_msg = f"Failed to start application: {e}"
        print(error_msg)
        
        # Show error in GUI if possible
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Application Error", error_msg)
        except:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main()