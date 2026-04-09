# Disease Prediction System 🏥

A machine learning-based system that predicts diseases based on user-provided symptoms.

## Features

- **Machine Learning Model**: Uses Random Forest Classifier for disease prediction
- **Interactive UI**: Easy-to-use command-line interface
- **Multiple Input Methods**: Quick selection or manual yes/no entry
- **Confidence Scores**: Shows prediction confidence and probability distribution
- **20+ Diseases**: Trained on common diseases and their symptoms

## Project Structure

```
disease_predictor/
├── data/
│   └── symptoms_diseases.csv      # Training dataset
├── models/                         # Trained models (generated after training)
│   ├── disease_model.pkl
│   └── symptom_names.pkl
├── train_model.py                 # Model training script
├── predict.py                     # Prediction interface (main user interface)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

1. **Install Python** (3.7 or higher)

2. **Navigate to project directory**:
   ```bash
   cd disease_predictor
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Train the Model

First, you need to train the machine learning model on the symptom-disease dataset:

```bash
python train_model.py
```

This will:
- Load the training data
- Train a Random Forest classifier
- Evaluate model performance
- Save the model and symptom names for prediction

### Step 2: Make Predictions

Run the prediction interface:

```bash
python predict.py
```

The system will offer two ways to input symptoms:

**Option 1: Quick Symptom Selection**
- Lists all available symptoms with numbers
- Enter comma-separated numbers for your symptoms
- Immediate prediction

**Option 2: Manual Symptom Entry**
- Step-by-step yes/no questions
- More detailed but interactive

### Example Usage

```
🏥 DISEASE PREDICTION SYSTEM 🏥

Options:
1. Quick Symptom Selection (by number)
2. Manual Symptom Entry (yes/no)
3. Exit

Select option (1-3): 1

AVAILABLE SYMPTOMS:
 1. Fever
 2. Cough
 3. Sore Throat
 4. Fatigue
 5. Body Ache
 ...

Enter the numbers of symptoms you have (comma-separated)
Example: 1,2,3

Your symptoms (numbers): 1,2,3,4,5,12

PREDICTION RESULTS:

🎯 Primary Prediction: Common Cold
📊 Confidence: 87.5%

All Predictions (sorted by probability):
  Common Cold          ████████████████░░ 87.5%
  Flu                  ██████████░░░░░░░░ 52.3%
  COVID-19             ████████░░░░░░░░░░ 40.1%
```

## Available Symptoms

The system recognizes the following symptoms:
- Fever
- Cough
- Sore Throat
- Fatigue
- Body Ache
- Headache
- Nausea
- Vomiting
- Diarrhea
- Rash
- Shortness of Breath
- Congestion
- Chills
- Loss of Taste
- Loss of Smell

## Available Diseases

The model can predict the following diseases:
- Common Cold
- COVID-19
- Flu
- Malaria
- Gastroenteritis
- Chickenpox
- Strep Throat
- Migraines
- Food Poisoning
- Pneumonia
- Stomach Flu
- Dengue Fever
- Asthma
- Allergies
- Typhoid
- Measles

## Model Performance

After training, the model achieves:
- Accuracy: ~85-90% (depending on test set)
- Precision and Recall metrics available in training output

## Important Disclaimer ⚠️

**This system is for educational purposes only.**

- **NOT a substitute for professional medical diagnosis**
- Always consult a qualified healthcare professional for medical advice
- Do not rely solely on this tool for health decisions
- Use this for learning about ML applications, not medical decisions

## Technical Details

### Algorithm
- **Model**: Random Forest Classifier
- **n_estimators**: 100 trees
- **max_depth**: 10
- **Train-Test Split**: 80-20
- **Random State**: 42 (for reproducibility)

### Libraries Used
- **pandas**: Data manipulation and CSV reading
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning algorithms
- **joblib**: Model serialization

## Data Format

Training data (CSV format):
```
fever,cough,sore_throat,fatigue,body_ache,headache,...,disease
1,1,1,1,1,0,...,COVID-19
1,0,0,1,0,1,...,Malaria
```

Where:
- 1 = Symptom present
- 0 = Symptom absent

## Troubleshooting

### Models not found error
**Problem**: "Model files not found"
**Solution**: Run `python train_model.py` first

### Import errors
**Problem**: "ModuleNotFoundError"
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

### Invalid symptom numbers
**Problem**: "Invalid symptom number"
**Solution**: Check the displayed symptom list and use correct numbers

## Future Improvements

- [ ] Add more training data for better accuracy
- [ ] Implement web interface (Flask/Django)
- [ ] Add medication/treatment recommendations
- [ ] Support multiple languages
- [ ] Export results to PDF
- [ ] Add symptom severity levels
- [ ] Implement deep learning models
- [ ] Add cross-validation for better evaluation

## License

Educational purpose only. Use responsibly.

## Contact & Support

For questions or issues with the system, please review the code comments or create documentation.

---

**Remember**: This tool is NOT a medical device and should never replace professional medical consultation.
