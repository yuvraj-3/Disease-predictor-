"""
Quick Start Guide and Test Script
Run this to verify the installation and see a demo
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if all required packages are installed"""
    print("=" * 50)
    print("CHECKING DEPENDENCIES")
    print("=" * 50)
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib'
    }
    
    all_installed = True
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name:20s} installed")
        except ImportError:
            print(f"✗ {package_name:20s} NOT installed")
            all_installed = False
    
    return all_installed

def install_dependencies():
    """Install required dependencies"""
    print("\n" + "=" * 50)
    print("INSTALLING DEPENDENCIES")
    print("=" * 50)
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", 
                             "requirements.txt", "-q"])
        print("✓ Dependencies installed successfully!")
        return True
    except Exception as e:
        print(f"✗ Error installing dependencies: {e}")
        return False

def check_data_file():
    """Check if training data exists"""
    print("\n" + "=" * 50)
    print("CHECKING DATA FILES")
    print("=" * 50)
    
    data_path = "data/symptoms_diseases.csv"
    
    if os.path.exists(data_path):
        print(f"✓ Training data found: {data_path}")
        return True
    else:
        print(f"✗ Training data not found: {data_path}")
        return False

def check_model_files():
    """Check if trained model exists"""
    print("\n" + "=" * 50)
    print("CHECKING MODEL FILES")
    print("=" * 50)
    
    model_path = "models/disease_model.pkl"
    symptoms_path = "models/symptom_names.pkl"
    
    model_exists = os.path.exists(model_path)
    symptoms_exists = os.path.exists(symptoms_path)
    
    if model_exists:
        print(f"✓ Model found: {model_path}")
    else:
        print(f"✗ Model not found: {model_path}")
    
    if symptoms_exists:
        print(f"✓ Symptoms database found: {symptoms_path}")
    else:
        print(f"✗ Symptoms database not found: {symptoms_path}")
    
    return model_exists and symptoms_exists

def main():
    """Main quick start guide"""
    print("\n" + "╔" + "=" * 48 + "╗")
    print("║" + " " * 10 + "DISEASE PREDICTION SYSTEM" + " " * 12 + "║")
    print("║" + " " * 15 + "Quick Start Guide" + " " * 17 + "║")
    print("╚" + "=" * 48 + "╝\n")
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\nSome dependencies are missing.")
        response = input("Install them now? (y/n): ").strip().lower()
        
        if response == 'y':
            if not install_dependencies():
                print("Failed to install dependencies. Please install manually:")
                print("  pip install -r requirements.txt")
                return
        else:
            print("Please install dependencies manually:")
            print("  pip install -r requirements.txt")
            return
    
    # Check data files
    if not check_data_file():
        print("\nTraining data is missing! Cannot proceed.")
        return
    
    # Check model files
    model_ok = check_model_files()
    
    if not model_ok:
        print("\n" + "=" * 50)
        print("MODEL NOT TRAINED")
        print("=" * 50)
        print("\nThe machine learning model needs to be trained first.")
        print("This will:")
        print("  1. Load symptom-disease data")
        print("  2. Train Random Forest classifier")
        print("  3. Evaluate model performance")
        print("  4. Save model for predictions")
        
        response = input("\nTrain model now? (y/n): ").strip().lower()
        
        if response == 'y':
            print("\nTraining model... (this may take a moment)")
            print("-" * 50)
            
            try:
                subprocess.call([sys.executable, "train_model.py"])
                print("-" * 50)
                print("✓ Model training completed!")
                model_ok = True
            except Exception as e:
                print(f"✗ Error during training: {e}")
                print("\nPlease run manually:")
                print("  python train_model.py")
                return
        else:
            print("\nTo train the model, run:")
            print("  python train_model.py")
            return
    
    # System ready
    if model_ok:
        print("\n" + "=" * 50)
        print("✓ SYSTEM READY!")
        print("=" * 50)
        print("\nThe disease prediction system is ready to use.")
        print("\nTo start making predictions, run:")
        print("  python predict.py")
        print("\nFeatures:")
        print("  • Quick symptom selection (by number)")
        print("  • Manual symptom entry (yes/no)")
        print("  • Confidence scores and probabilities")
        print("  • Support for 15+ diseases")
        print("\nRemember:")
        print("  ⚠️  This is for educational purposes only")
        print("  ⚠️  Always consult a medical professional")
        
        response = input("\nStart prediction system now? (y/n): ").strip().lower()
        
        if response == 'y':
            print("\nLaunching prediction system...")
            print("-" * 50)
            subprocess.call([sys.executable, "predict.py"])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
