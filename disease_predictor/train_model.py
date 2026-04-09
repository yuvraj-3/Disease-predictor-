"""
Disease Prediction Model Training Script
Trains a machine learning model to predict diseases based on symptoms
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def load_data(filepath):
    """Load the symptoms-diseases dataset"""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"Diseases in dataset: {df['disease'].unique()}")
    return df

def prepare_data(df):
    """Prepare data for model training"""
    # Separate features and target
    X = df.drop('disease', axis=1)
    y = df['disease']
    
    # Get symptom names
    symptom_names = X.columns.tolist()
    
    # Split into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, symptom_names

def train_model(X_train, y_train):
    """Train Random Forest classifier"""
    print("\nTraining Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy

def save_model(model, symptom_names, filepath='models/disease_model.pkl', 
               symptoms_filepath='models/symptom_names.pkl'):
    """Save trained model and symptom names"""
    joblib.dump(model, filepath)
    joblib.dump(symptom_names, symptoms_filepath)
    print(f"\nModel saved to {filepath}")
    print(f"Symptom names saved to {symptoms_filepath}")

def main():
    """Main training pipeline"""
    print("=" * 50)
    print("Disease Prediction Model Training")
    print("=" * 50)
    
    # Load data
    df = load_data('data/symptoms_diseases.csv')
    
    # Prepare data
    X_train, X_test, y_train, y_test, symptom_names = prepare_data(df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, symptom_names)
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
