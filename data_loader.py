import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class ResumeDataLoader:
    """
    Load and preprocess resume data from local CSV files"""
    
    def __init__(self, data_path='data/'):
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, csv_file=None):
        """
        1. Load the resume dataset from a local CSV file"""

        try:
            if csv_file is None:
                csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
                if not csv_files:
                    print("❌ No CSV files found in data/ directory")
                    return None
                
                # Prioritize Resume.csv if it exists
                if 'Resume.csv' in csv_files:
                    csv_file = 'Resume.csv'
                else:
                    csv_file = csv_files[0]
                    
                print(f"Found CSV file: {csv_file}")
            
            file_path = os.path.join(self.data_path, csv_file)
            
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return None
            
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
                    print(f"Successfully loaded. Uses {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Error with {encoding}: {e}")
                    continue
            
            if self.df is None:
                print("❌ Error loadinf CSV with any encoding")
                return None
            
            print(f"Loaded dataset with {len(self.df)} records")
            print(f"Dataset shape: {self.df.shape}")
            print(f"Dataset columns: {list(self.df.columns)}")
            
            print("\n📄 First few rows:")
            print(self.df.head(2))
            
            self.adapt_dataset_structure()
            
            return self.df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def adapt_dataset_structure(self):
        """
        Adapt the dataset structure automatically"""
        print("\n Adapting dataset structure...")
        
        # common text column names
        text_column_names = [
            'Resume_str', 'Resume_text', 'Text', 'Content', 'resume_text',
            'Resume', 'resume', 'description', 'Description', 'summary'
        ]
        
        # common category column names
        category_column_names = [
            'Category', 'Job_Category', 'Field', 'Domain', 'Position', 
            'Job_Title', 'category', 'job_category', 'label', 'Label'
        ]
        
        # finding text column
        text_column = None
        for col in self.df.columns:
            if col in text_column_names:
                text_column = col
                break
            # checking if column name contains key words
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['resume', 'text', 'content', 'description']):
                text_column = col
                break
        
        # finding category column
        category_column = None
        for col in self.df.columns:
            if col in category_column_names:
                category_column = col
                break
            # if column name contains key words
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['category', 'label', 'job', 'field']):
                category_column = col
                break
        
        print(f"Text column detected: {text_column}")
        print(f"Category column detected: {category_column}")
        
        # if no dedicated text column found. 
        # trying to combine multiple columns
        if text_column is None:
            print("🔄 No single text column found. Looking for combinable columns...")

            string_columns = self.df.select_dtypes(include=['object']).columns.tolist()
            if category_column and category_column in string_columns:
                string_columns.remove(category_column)
            
            # Remove columns that look like IDs or names
            combinable_columns = []
            for col in string_columns:
                col_lower = col.lower()
                if not any(skip_word in col_lower for skip_word in ['id', 'name', 'email', 'phone']):
                    combinable_columns.append(col)
            
            if combinable_columns:
                print(f"📝 Combining columns: {combinable_columns}")
                self.df['Combined_Text'] = ''
                for col in combinable_columns:
                    self.df['Combined_Text'] += ' ' + self.df[col].astype(str).fillna('')
                text_column = 'Combined_Text'
        
        # If still no category column, try to detect it
        if category_column is None:
            print("🔍 Trying to detect category column...")
            for col in self.df.columns:
                # Check if column has reasonable number of unique values (between 5 and 50)
                # and is not a text column.
                # taking 50 as randomly generated upper limit to avoid too many categories. 
                # modifyable based on dataset
                unique_count = self.df[col].nunique()
                if 5 <= unique_count <= 50:
                    print(f"   {col}: {unique_count} unique values")
                    category_column = col
                    break
        
        # Create standardized columns if it found both
        if text_column and category_column:
            self.df['Resume'] = self.df[text_column].astype(str)
            self.df['Category'] = self.df[category_column].astype(str)
            
            print(f" Successfully mapped as follows:")
            print(f"   Text: '{text_column}' -> 'Resume'")
            print(f"   Category: '{category_column}' -> 'Category'")
            
            # Show category distribution
            print(f"\n Category distribution:")
            category_counts = self.df['Category'].value_counts()
            print(category_counts)
            
            return True
        else:
            print("Could not identify required columns")
            print("Available columns:", list(self.df.columns))
            
            # Show column types and unique values. this helps to understand the dataset structure
            print("\n📊 Column analysis:")
            for col in self.df.columns:
                dtype = self.df[col].dtype
                unique_count = self.df[col].nunique()
                null_count = self.df[col].isnull().sum()
                print(f"   {col}: {dtype}, {unique_count} unique, {null_count} nulls")
            
            return False
    
    def clean_resume_text(self, resume_text):
        """
        Clean and preprocess resume text"""
        if pd.isna(resume_text) or resume_text is None:
            return ""
            
        # converts to string
        resume_text = str(resume_text)
        
        # removes URLs
        # will be modified to keep important links (linkedin, github, portfolio etc.)
        resume_text = re.sub(r'http\S+', '', resume_text)
        resume_text = re.sub(r'www\S+', '', resume_text)
        
        # remove email addresses
        resume_text = re.sub(r'\S+@\S+', '', resume_text)
        
        # remove phone numbers
        resume_text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', resume_text)
        resume_text = re.sub(r'\+\d{1,3}\s?\d{3,4}\s?\d{3,4}\s?\d{3,4}', '', resume_text)
        
        # remove special characters but keep important punctuation
        resume_text = re.sub(r'[^\w\s\.,!?;:]', ' ', resume_text)
        
        # remove extra whitespace
        resume_text = re.sub(r'\s+', ' ', resume_text).strip()
        
        # vonverts to lowercase
        resume_text = resume_text.lower()
        
        return resume_text
    
    def preprocess_data(self):
        """
        Preprocess the loaded data"""
        if self.df is None:
            print("No data loaded. Please load data first.")
            return False
       
        if 'Resume' not in self.df.columns or 'Category' not in self.df.columns:
            print("❌ Missing required columns 'Resume' and 'Category'")
            print("Available columns:", list(self.df.columns))
            return False
        
        print("Preprocessing data...")
        
        # removes rows with missing data
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['Resume', 'Category'])
        print(f"Removed {initial_count - len(self.df)} rows with missing data")
        
        # cleaning the resume text
        print("🧹 Cleaning resume text...")
        self.df['Cleaned_Resume'] = self.df['Resume'].apply(self.clean_resume_text)
        
        # if the resume is very short, it is likely not useful
        min_length = 20 
        self.df = self.df[self.df['Cleaned_Resume'].str.len() > min_length]
        print(f"After cleaning: {len(self.df)} resumes remain (min length: {min_length} chars)")

        self.df['Category'] = self.df['Category'].str.strip()
        
        category_counts = self.df['Category'].value_counts()
        min_samples = max(5, len(self.df) // 100)  # At least 5 samples or 1% of data
        categories_to_keep = category_counts[category_counts >= min_samples].index
        
        print(f"Category filtering (min {min_samples} samples per category):")
        for cat in category_counts.index:
            count = category_counts[cat]
            status = "Keep" if count >= min_samples else "Remove"
            print(f"   {cat}: {count} samples - {status}")
        
        self.df = self.df[self.df['Category'].isin(categories_to_keep)]
        print(f"Keeping {len(categories_to_keep)} categories with sufficient data")
        
        # encode categories
        self.df['Category_Encoded'] = self.label_encoder.fit_transform(self.df['Category'])
        
        # save label encoder
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        
        print("Data preprocessing completed!")
        print(f"Final dataset: {len(self.df)} resumes, {len(categories_to_keep)} categories")
        
        return True
        
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets"""
        if self.df is None or 'Cleaned_Resume' not in self.df.columns:
            print("No preprocessed data available.")
            return None, None, None, None
        
        X = self.df['Cleaned_Resume']
        y = self.df['Category_Encoded']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"   Data splitting completed:")
        print(f"   Training set: {len(self.X_train)} samples")
        print(f"   Test set: {len(self.X_test)} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_category_mapping(self):
        """
        Mapping between encoded labels & category names"""
        if hasattr(self.label_encoder, 'classes_'):
            mapping = {i: category for i, category in enumerate(self.label_encoder.classes_)}
            return mapping
        return None
    
    def save_processed_data(self):
        """
        Save processed data for later use"""
        if self.df is not None:
            processed_file = os.path.join(self.data_path, 'processed_resumes.csv')
            # only saves necessary columns. reduces file size
            columns_to_save = ['Resume', 'Category', 'Cleaned_Resume', 'Category_Encoded']
            existing_columns = [col for col in columns_to_save if col in self.df.columns]
            
            self.df[existing_columns].to_csv(processed_file, index=False)
            print(f"Processed data saved to: {processed_file}")
    
    def get_sample_data(self, n_samples=5):
        """
        Get sample data for testing"""
        if self.df is not None and 'Category' in self.df.columns and 'Cleaned_Resume' in self.df.columns:
            return self.df.sample(min(n_samples, len(self.df)))[['Category', 'Cleaned_Resume']]
        return None

if __name__ == "__main__":
    print("🎯 Resume Screening App-V2 🎯")
    print("=" * 50)
    
    # initializing
    loader = ResumeDataLoader()
    
    df = loader.load_data()
    
    if df is not None:
        print(f"\nDataset loaded successfully!")
        print(f"Shape: {df.shape}")
        
        # preprocessing data
        if loader.preprocess_data():
            # Splitting data
            X_train, X_test, y_train, y_test = loader.split_data()
            
            if X_train is not None:
                category_mapping = loader.get_category_mapping()
                print(f"\n🏷️ Found {len(category_mapping)} categories:")
                for code, category in sorted(category_mapping.items()):
                    count = (loader.df['Category_Encoded'] == code).sum()
                    print(f"   {code:2d}: {category:<25} ({count:4d} samples)")

                loader.save_processed_data()
                
                print(f"\nSample processed resumes:")
                samples = loader.get_sample_data(3)
                if samples is not None:
                    for idx, row in samples.iterrows():
                        print(f"\nCategory: {row['Category']}")
                        print(f"Resume excerpt: {row['Cleaned_Resume'][:150]}...")
                        print("-" * 40)
                
                print(f"\nData preprocessing completed successfully!")
                print(f"Next steps:")
                print(f"1. Run 'python train_real_model.py' to train the model")
                print(f"2. Run 'streamlit run app.py' to start the web application (at localhost:8501)")
            
        else:
            print("\nData preprocessing failed")
            print("(possible error with CSV file structure and column names)")
        
    else:
        print("\nFailed to load dataset")
        print("Please ensure:")
        print("1. You have a CSV file in the data/ directory")
        print("2. The CSV has columns for resume text and job categories")
        print("3. The file is properly formatted and not corrupted")