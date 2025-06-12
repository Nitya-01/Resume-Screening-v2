import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import ResumeDataLoader
import warnings
warnings.filterwarnings('ignore')

class RealResumeClassifier:
    """
    Resume classifier trained on real Kaggle data - v2"""
    
    def __init__(self, data_path='data/', model_path='models/'):
        self.data_path = data_path
        self.model_path = model_path
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.category_mapping = None
        
        # Create directories if they don't exist
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        
        print(f"Data path: {os.path.abspath(data_path)}")
        print(f"Model path: {os.path.abspath(model_path)}")
    
    def load_processed_data(self):
        """
        Load preprocessed data with better error handling(error reporting for now. handlign and correction will be added later)"""
        try:
            print("Checking for existing processed data...")
            
            # Check for processed data file
            processed_file = os.path.join(self.data_path, 'processed_resumes.csv')
            label_encoder_file = os.path.join(self.data_path, 'label_encoder.pkl')
            
            if os.path.exists(processed_file) and os.path.exists(label_encoder_file):
                print("Loading existing processed data...")
                
                try:
                    df = pd.read_csv(processed_file)
                    self.label_encoder = joblib.load(label_encoder_file)
                    
                    print(f"   Loaded processed data: {df.shape}")
                    print(f"   Columns: {list(df.columns)}")
                    
                    required_columns = ['Cleaned_Resume', 'Category_Encoded']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        print(f"Missing columns in processed data: {missing_columns}")
                        print("Will reprocess data...")
                        raise ValueError(f"Missing columns: {missing_columns}")
                    
                except Exception as e:
                    print(f"Error loading processed data: {e}")
                    print("Will reprocess data from scratch...")
                    df = None
            else:
                print("No processed data found. Processing from scratch...")
                df = None
            
            if df is None:
                print("Processing raw data...")
                loader = ResumeDataLoader(self.data_path)
                
                # Load raw data
                raw_df = loader.load_data()
                if raw_df is None:
                    print("Failed to load raw data!")
                    return None, None, None, None
                
                # Preprocess data
                if not loader.preprocess_data():
                    print("Failed to preprocess data!")
                    return None, None, None, None
                
                # Save processed data
                loader.save_processed_data()
                
                # Use the processed data
                df = loader.df
                self.label_encoder = loader.label_encoder
                
                print(f"Data processing completed: {df.shape}")
            
            # Split data
            print("Splitting data...")
            X = df['Cleaned_Resume']
            y = df['Category_Encoded']
            
            # Check for empty or invalid data
            if len(X) == 0 or len(y) == 0:
                print("No valid data found after processing!")
                return None, None, None, None
            
            if len(X) < 10:
                print(f"Very small dataset ({len(X)} samples). Using different split strategy...")
                test_size = max(0.1, 2/len(X))  # Atleast 2 samples for test set
            else:
                test_size = 0.2
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            except ValueError as e:
                print(f"Stratified split failed: {e}")
                print("Using random split instead...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            
            if hasattr(self.label_encoder, 'classes_'):
                self.category_mapping = {i: category for i, category in enumerate(self.label_encoder.classes_)}
            else:
                print("Label encoder not properly initialized!")
                return None, None, None, None
            
            print(f"   Data loaded and split successfully!")
            print(f"   Training samples: {len(X_train)}")
            print(f"   Test samples: {len(X_test)}")
            print(f"   Categories: {len(self.category_mapping)}")
            print(f"   Category distribution:")
            
            for i, category in self.category_mapping.items():
                count = (y == i).sum()
                print(f"     {i}: {category} ({count} samples)")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Error in load_processed_data: {e}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            return None, None, None, None
    
    def train_model(self, X_train, y_train, model_type='svm'):
        """
        Train the classification model with better error handling"""
        print(f"Training {model_type.upper()} model...")
        
        try:
            # Creating TF-IDF vectorizer
            print("🔧 Creating TF-IDF vectorizer...")
            self.vectorizer = TfidfVectorizer(
                max_features=min(5000, len(X_train) * 10),  # Adjusting based on data size
                stop_words='english',
                ngram_range=(1, 2),
                min_df=max(1, len(X_train) // 100),  # Adjusting based on data size
                max_df=0.95
            )
            
            print("Vectorizing training data...")
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            print(f"Feature vector shape: {X_train_tfidf.shape}")
            
            # choosing & configuring classifier
            print(f"Setting up {model_type} classifier...")
            if model_type == 'svm':
                self.classifier = SVC(
                    kernel='linear',
                    C=1.0,
                    probability=True,
                    random_state=42
                )
            elif model_type == 'rf':
                self.classifier = RandomForestClassifier(
                    n_estimators=min(100, len(X_train)),  
                    max_depth=None,
                    min_samples_split=max(2, len(X_train) // 50),
                    min_samples_leaf=max(1, len(X_train) // 100),
                    random_state=42
                )
            
            print("Training model...")
            self.classifier.fit(X_train_tfidf, y_train)
            
            # Cross-validation (if we have enough data)
            if len(X_train) >= 10:
                print("Performing cross-validation...")
                cv_folds = min(5, len(X_train) // 2)  # At least 2 samples per fold
                cv_scores = cross_val_score(self.classifier, X_train_tfidf, y_train, cv=cv_folds)
                print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            else:
                print("Dataset too small for cross-validation. Skipping...")
            
            print("Model training completed!")
            return True
            
        except Exception as e:
            print(f"Error in model training: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model with better error handling (reporting and handling)"""
        if self.classifier is None or self.vectorizer is None:
            print("Model not trained yet!")
            return None
        
        try:
            print("Evaluating model...")
            
            # Transform test data
            X_test_tfidf = self.vectorizer.transform(X_test)
            
            # Make predictions
            y_pred = self.classifier.predict(X_test_tfidf)
            y_pred_proba = self.classifier.predict_proba(X_test_tfidf)
            
            # Calculate and display accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Test Accuracy: {accuracy:.4f}")
            
            # Detailed classification report
            print("\nDetailed Classification Report:")
            target_names = [self.category_mapping[i] for i in sorted(self.category_mapping.keys())]
            print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
            
            # Save performance metrics
            self.save_performance_metrics(accuracy, y_test, y_pred, target_names)
            
            # Show sample predictions
            self.show_sample_predictions(X_test, y_test, y_pred, y_pred_proba, min(5, len(X_test)))
            
            return accuracy
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_performance_metrics(self, accuracy, y_true, y_pred, target_names):
        """
        Save performance metrics to file"""
        
        try:
            # Generate classification report
            classification_rep = classification_report(
                y_true, y_pred, 
                target_names=target_names, 
                zero_division=0
            )
            performance_text = (
                "Model Performance Report\n"
                "=" * 50 + "\n"
                f"Test Accuracy: {accuracy:.4f}\n\n"
                "Classification Report:\n"
                f"{classification_rep}\n"
            )
            
            performance_file = os.path.join(self.model_path, 'model_performance.txt')
            with open(performance_file, 'w') as f:
                f.write(performance_text)
            
            print(f"Performance metrics saved to: {performance_file}")
            
        except Exception as e:
            print(f"Could not save performance metrics: {e}")
    
    def show_sample_predictions(self, X_test, y_true, y_pred, y_pred_proba, n_samples=3):
        """
        Show sample predictions with confidence scores"""
        try:
            print(f"\n🔍 Sample Predictions (showing {n_samples} examples):")
            print("=" * 80)
            
            # Convert to lists for easier handling
            X_test_list = X_test.tolist()
            y_true_list = y_true.tolist()
            y_pred_list = y_pred.tolist()
            
            # Show random samples
            indices = np.random.choice(len(X_test_list), min(n_samples, len(X_test_list)), replace=False)
            
            for i, idx in enumerate(indices):
                true_label = self.category_mapping[y_true_list[idx]]
                pred_label = self.category_mapping[y_pred_list[idx]]
                confidence = max(y_pred_proba[idx])
                
                print(f"\n Sample {i+1}:")
                print(f"   True Category: {true_label}")
                print(f"   Predicted: {pred_label} (Confidence: {confidence:.3f})")
                print(f"   Correct: {'✅' if true_label == pred_label else '❌'}")
                print(f"   Resume excerpt: {X_test_list[idx][:100]}...")
                print("-" * 60)
                
        except Exception as e:
            print(f" Error showing sample predictions: {e}")
    
    def save_model(self):
        """
        Save the trained model and vectorizer"""
        if self.classifier is None or self.vectorizer is None:
            print("No model to save!")
            return False
        
        try:
            # Save model components
            model_files = {
                'resume_classifier.pkl': self.classifier,
                'tfidf_vectorizer.pkl': self.vectorizer,
                'category_mapping.pkl': self.category_mapping
            }
            
            for filename, obj in model_files.items():
                filepath = os.path.join(self.model_path, filename)
                joblib.dump(obj, filepath)
                print(f" Saved: {filepath}")
            
            print("Model saved successfully!")
            return True
            
        except Exception as e:
            print(f" Error saving model: {e}")
            return False
    
    def predict_single_resume(self, resume_text):
        """
        Predict category for a single resume
        """
        if self.classifier is None or self.vectorizer is None:
            print("Model not loaded!")
            return None, None
        
        try:
            # Clean the resume text
            from data_loader import ResumeDataLoader
            loader = ResumeDataLoader()
            cleaned_text = loader.clean_resume_text(resume_text)
            
            if len(cleaned_text.strip()) < 5:
                print("Resume text too short after cleaning")
                return None, None
            
            # Transform text
            text_tfidf = self.vectorizer.transform([cleaned_text])
            
            # Make prediction
            prediction = self.classifier.predict(text_tfidf)[0]
            confidence = max(self.classifier.predict_proba(text_tfidf)[0])
            
            # Get category name
            category = self.category_mapping[prediction]
            
            return category, confidence
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, None

def main():
    print("Training Resume Classifier with Real Data")
    print("=" * 50)
    
    try:
        # Initialize classifier
        classifier = RealResumeClassifier()
        
        # Load data
        print("Loading data...")
        X_train, X_test, y_train, y_test = classifier.load_processed_data()
        
        if X_train is None:
            print("Failed to load data. Please check your data setup.")
            print("\n Troubleshooting steps:")
            print("1. Ensure you have Resume.csv in the data/ directory")
            print("2. Run: python debug_data_loading.py")
            print("3. Check that your CSV has resume text and category columns")
            return False
        
        # Train model
        print("\nTraining model...")
        if not classifier.train_model(X_train, y_train, model_type='svm'):
            print("Model training failed!")
            return False
        
        # Evaluate model
        print("\nEvaluating model...")
        accuracy = classifier.evaluate_model(X_test, y_test)
        
        if accuracy is None:
            print("Model evaluation failed!")
            return False
        
        # Save the trained model
        print("\nSaving model...")
        if not classifier.save_model():
            print("Model saving failed!")
            return False
        
        # Test single prediction
        print("\n🧪 Testing single prediction:")
        sample_resume = """
        Software Engineer with 5 years of experience in Python, Django, and React.
        Built scalable web applications and worked with databases like PostgreSQL.
        Experience with machine learning and data science projects.
        """
        
        category, confidence = classifier.predict_single_resume(sample_resume)
        if category:
            print(f"   Sample resume predicted as: {category} (Confidence: {confidence:.3f})")
        else:
            print("    Sample prediction failed")
        
        print(f"\n TRAINING COMPLETED SUCCESSFULLY!")
        print(f"   Final accuracy: {accuracy:.4f}")
        print(f"   Model saved to: models/")
        print(f"\n--> Next steps:")
        print(f"   1. Run: streamlit run app.py")
        print(f"   2. Upload resumes to test the classifier")
        
        return True
        
    except Exception as e:
        print(f"Unexpected error in main: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("Training failed. Please check the logs for details.")
    else:
        print("Training completed successfully!")