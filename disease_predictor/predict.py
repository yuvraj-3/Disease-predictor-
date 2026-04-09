"""
Disease Predictor - User Interface
Takes user symptoms and predicts possible diseases
"""

import joblib
import numpy as np
import os

class DiseasePredictor:
    """Disease prediction system using trained ML model"""
    
    def __init__(self, model_path='models/disease_model.pkl', 
                 symptoms_path='models/symptom_names.pkl'):
        """Initialize the predictor with trained model"""
        
        if not os.path.exists(model_path) or not os.path.exists(symptoms_path):
            print("Error: Model files not found. Please train the model first.")
            print("Run: python train_model.py")
            exit(1)
        
        self.model = joblib.load(model_path)
        self.symptom_names = joblib.load(symptoms_path)
        
        print("✓ Model loaded successfully!")
        print(f"Available symptoms: {len(self.symptom_names)}")
    
    def display_symptoms(self):
        """Display all available symptoms"""
        print("\n" + "=" * 50)
        print("AVAILABLE SYMPTOMS:")
        print("=" * 50)
        
        for idx, symptom in enumerate(self.symptom_names, 1):
            # Convert underscore to space and title case
            symptom_display = symptom.replace('_', ' ').title()
            print(f"{idx:2d}. {symptom_display}")
        
        print("=" * 50)
    
    def get_user_symptoms(self):
        """Get symptoms from user input"""
        symptoms_vector = np.zeros(len(self.symptom_names))
        
        self.display_symptoms()
        
        print("\nEnter the numbers of symptoms you have (comma-separated)")
        print("Example: 1,2,3 (for symptoms 1, 2, and 3)")
        print("Or enter 0 to skip and get a default prediction")
        
        user_input = input("\nYour symptoms (numbers): ").strip()
        
        if user_input == '0':
            print("No symptoms selected.")
            return None
        
        try:
            selected_indices = [int(x.strip()) - 1 for x in user_input.split(',')]
            
            for idx in selected_indices:
                if 0 <= idx < len(self.symptom_names):
                    symptoms_vector[idx] = 1
                else:
                    print(f"Warning: Invalid symptom number {idx + 1}, skipped")
            
            if sum(symptoms_vector) == 0:
                print("No valid symptoms selected.")
                return None
            
            return symptoms_vector
        
        except ValueError:
            print("Error: Please enter valid symptom numbers")
            return None
    
    def predict_disease(self, symptoms_vector):
        """Predict disease based on symptoms"""
        if symptoms_vector is None:
            return None
        
        # Reshape for single prediction
        symptoms_vector = symptoms_vector.reshape(1, -1)
        
        # Get prediction
        prediction = self.model.predict(symptoms_vector)[0]
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(symptoms_vector)[0]
        classes = self.model.classes_
        
        # Get confidence score
        confidence = max(probabilities) * 100
        
        return prediction, confidence, dict(zip(classes, probabilities))
    
    def display_results(self, prediction, confidence, all_probabilities):
        """Display prediction results"""
        print("\n" + "=" * 50)
        print("PREDICTION RESULTS:")
        print("=" * 50)
        
        print(f"\n🎯 Primary Prediction: {prediction}")
        print(f"📊 Confidence: {confidence:.1f}%")
        
        print(f"\nAll Predictions (sorted by probability):")
        sorted_probs = sorted(all_probabilities.items(), 
                             key=lambda x: x[1], reverse=True)
        
        for disease, prob in sorted_probs:
            percentage = prob * 100
            bar_length = int(percentage / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  {disease:20s} {bar} {percentage:5.1f}%")
        
        print("\n" + "=" * 50)
        print("⚠️  DISCLAIMER: This is for educational purposes only.")
        print("Please consult a medical professional for accurate diagnosis.")
        print("=" * 50 + "\n")
    
    def manual_symptom_entry(self):
        """Allow user to manually enter symptoms"""
        symptoms_vector = np.zeros(len(self.symptom_names))
        
        print("\n" + "=" * 50)
        print("MANUAL SYMPTOM ENTRY:")
        print("=" * 50)
        
        for idx, symptom in enumerate(self.symptom_names):
            symptom_display = symptom.replace('_', ' ').title()
            response = input(f"Do you have {symptom_display}? (y/n): ").strip().lower()
            
            if response == 'y':
                symptoms_vector[idx] = 1
        
        if sum(symptoms_vector) == 0:
            print("No symptoms selected.")
            return None
        
        return symptoms_vector
    
    def run(self):
        """Run the disease predictor interactive interface"""
        print("\n" + "=" * 50)
        print("🏥 DISEASE PREDICTION SYSTEM 🏥")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. Quick Symptom Selection (by number)")
            print("2. Manual Symptom Entry (yes/no)")
            print("3. Exit")
            
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                symptoms_vector = self.get_user_symptoms()
                
                if symptoms_vector is not None:
                    result = self.predict_disease(symptoms_vector)
                    if result:
                        prediction, confidence, all_probs = result
                        self.display_results(prediction, confidence, all_probs)
            
            elif choice == '2':
                symptoms_vector = self.manual_symptom_entry()
                
                if symptoms_vector is not None:
                    result = self.predict_disease(symptoms_vector)
                    if result:
                        prediction, confidence, all_probs = result
                        self.display_results(prediction, confidence, all_probs)
            
            elif choice == '3':
                print("\nThank you for using Disease Prediction System!")
                break
            
            else:
                print("Invalid option. Please select 1, 2, or 3.")

def main():
    """Main entry point"""
    try:
        predictor = DiseasePredictor()
        predictor.run()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
