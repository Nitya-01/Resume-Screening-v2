# app.py - Updated Streamlit app using real trained model
import streamlit as st
import joblib
import os
import PyPDF2
import docx
import io
from data_loader import ResumeDataLoader
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Resume Screening (v2)",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ResumeClassifierApp:
    """
    Streamlit app for resume classification using real trained models"""
    
    def __init__(self):
        self.model_path = 'models/'
        self.classifier = None
        self.vectorizer = None
        self.category_mapping = None
        self.data_loader = ResumeDataLoader()
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self):
        """
        Load the trained model components"""
        try:
            model_files = {
                'classifier': os.path.join(self.model_path, 'resume_classifier.pkl'),
                'vectorizer': os.path.join(self.model_path, 'tfidf_vectorizer.pkl'),
                'categories': os.path.join(self.model_path, 'category_mapping.pkl')
            }
            
            # Check if all model files exist
            missing_files = [name for name, path in model_files.items() if not os.path.exists(path)]
            
            if missing_files:
                st.error(f"❌ Missing model files: {missing_files}")
                st.error("Please run 'python train_real_model.py' first to train the model.")
                return False
            
            # Load model components
            self.classifier = joblib.load(model_files['classifier'])
            self.vectorizer = joblib.load(model_files['vectorizer'])
            self.category_mapping = joblib.load(model_files['categories'])
            
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def extract_text_from_pdf(self, uploaded_file):
        """
        Extract text from PDF file
        """
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return None
    
    def extract_text_from_docx(self, uploaded_file):
        """
        Extract text from DOCX file
        """
        try:
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return None
    
    def extract_text_from_txt(self, uploaded_file):
        """
        Extract text from TXT file
        """
        try:
            return uploaded_file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error reading TXT: {e}")
            return None
    
    def process_uploaded_file(self, uploaded_file):
        """
        Process uploaded file and extract text
        """
        if uploaded_file is None:
            return None
        
        file_type = uploaded_file.type
        
        if file_type == "application/pdf":
            return self.extract_text_from_pdf(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return self.extract_text_from_docx(uploaded_file)
        elif file_type == "text/plain":
            return self.extract_text_from_txt(uploaded_file)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
    
    def predict_resume_category(self, resume_text):
        """
        Predict resume category with confidence score
        """
        if not all([self.classifier, self.vectorizer, self.category_mapping]):
            return None, None, None
        
        try:
            # Clean the resume text
            cleaned_text = self.data_loader.clean_resume_text(resume_text)
            
            if len(cleaned_text.strip()) < 10:
                return None, None, "Resume text too short after cleaning"
            
            # Transform text using the trained vectorizer
            text_tfidf = self.vectorizer.transform([cleaned_text])
            
            # Make prediction
            prediction = self.classifier.predict(text_tfidf)[0]
            probabilities = self.classifier.predict_proba(text_tfidf)[0]
            confidence = max(probabilities)
            
            # Get category name
            category = self.category_mapping[prediction]
            
            # Get all probabilities for visualization
            all_probs = {self.category_mapping[i]: prob for i, prob in enumerate(probabilities)}
            
            return category, confidence, all_probs
            
        except Exception as e:
            return None, None, f"Error in prediction: {e}"
    
    def create_probability_chart(self, probabilities):
        """
        Create a bar chart of category probabilities"""
        if not probabilities:
            return None
        
        # Sort by probability
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        categories, probs = zip(*sorted_probs)
        
        # Create bar chart
        fig = px.bar(
            x=list(probs),
            y=list(categories),
            orientation='h',
            title="Category Prediction Probabilities",
            labels={'x': 'Probability', 'y': 'Category'},
            color=list(probs),
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis={'categoryorder':'total ascending'}
        )
        
        return fig
    
    def show_model_info(self):
        """
        Display model information in sidebar"""
        with st.sidebar:
            st.header("🤖 Model Information")
            
            if all([self.classifier, self.vectorizer, self.category_mapping]):
                st.success("Model loaded successfully")
                
                # Model details
                st.subheader("Model Details")
                st.write(f"**Algorithm:** {type(self.classifier).__name__}")
                st.write(f"**Features:** {self.vectorizer.max_features}")
                st.write(f"**Categories:** {len(self.category_mapping)}")
                
            #     # Show categories
            #     st.subheader("Available Categories")
            #     for code, category in sorted(self.category_mapping.items()):
            #         st.write(f"• {category}")
                
            #     # Model performance (if available)
            #     performance_file = os.path.join(self.model_path, 'model_performance.txt')
            #     if os.path.exists(performance_file):
            #         with open(performance_file, 'r') as f:
            #             performance = f.read()
            #         st.subheader("Model Performance")
            #         st.text(performance)
            # else:
            #     st.error("❌ Model not loaded")
            #     st.write("Please train the model first:")
            #     st.code("python train_real_model.py")
    
    def run_app(self):
        """
        Main Streamlit app"""
        # Title and description
        st.title("Resume screening - v2")
        st.markdown("### Powered by Kaggle Dataset & Machine Learning. \n Dataset: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset")
        
        self.show_model_info()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📄 Upload Resume: ")
     
            uploaded_file = st.file_uploader(
                "Choose a resume file",
                type=['pdf', 'docx', 'txt'],
                help="Supported formats: PDF, DOCX, TXT (Max size: 200MB)"
            )
            
            # Text input option
            st.subheader("📝 Or Paste Resume Text below for analysis:")
            manual_text = st.text_area(
                "Paste resume content here:",
                height=200,
                placeholder="Copy and paste resume text here..."
            )
            
            # Process button
            if st.button("Start analyzing", type="primary"):
                resume_text = None
                
                # Get text from file or manual input
                if uploaded_file is not None:
                    resume_text = self.process_uploaded_file(uploaded_file)
                    if resume_text:
                        st.success(f"Successfully extracted text from {uploaded_file.name}")
                elif manual_text.strip():
                    resume_text = manual_text
                else:
                    st.warning("⚠️ Please upload a file or paste resume text")
                    return
                
                # Make prediction
                if resume_text:
                    with st.spinner("Analyzing resume..."):
                        category, confidence, all_probs = self.predict_resume_category(resume_text)
                    
                    # Display results
                    if category:
                        # Success message
                        st.success("Analysis completd")
                        
                        # Main prediction
                        col_pred1, col_pred2 = st.columns(2)
                        with col_pred1:
                            st.metric("Predicted Category", category)
                        with col_pred2:
                            st.metric("Confidence Score", f"{confidence:.1%}")
                        
                        # Confidence indicator (will be included after accurancy enhancement!)
                        # if confidence >= 0.8:
                        #     st.success(f"🟢 High confidence prediction")
                        # elif confidence >= 0.6:
                        #     st.warning(f"🟡 Moderate confidence prediction")
                        # else:
                        #     st.error(f"🔴 Low confidence prediction. Please consider reviewing resume content.")
                        
                        # Show probability chart
                        if all_probs:
                            st.subheader("Detailed Probability Breakdown")
                            chart = self.create_probability_chart(all_probs)
                            if chart:
                                st.plotly_chart(chart, use_container_width=True)
                            
                            # Show top 2 predictions
                            st.subheader("Top 2 Predictions")
                            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:2]
                            
                            for i, (cat, prob) in enumerate(sorted_probs):
                                st.write(f"{cat}: {prob:.1%}")
                        
                        # Show resume preview
                        with st.expander("Resume Text Preview"):
                            cleaned_preview = self.data_loader.clean_resume_text(resume_text)
                            st.text_area("Cleaned resume text:", cleaned_preview[:100000] + "..." if len(cleaned_preview) > 1000 else cleaned_preview, height=200)
                    
                    else:
                        st.error(f"Prediction failed: {all_probs}")
        
        with col2:
            st.header("Quick Stats")
            
            # Show some quick statistics
            if all([self.classifier, self.vectorizer, self.category_mapping]):
                st.info(f"🎯 **{len(self.category_mapping)}** job categories. (adding more...)")
                st.info(f"🔤 **{self.vectorizer.max_features:,}** features")
                st.info(f"🧠 **{type(self.classifier).__name__}** algorithm")
                
                
                # st.subheader("📋 Example Categories")
                # example_cats = list(self.category_mapping.values())[:8]
                # for cat in example_cats:
                #     st.write(f"• {cat}")
                
                # if len(self.category_mapping) > 8:
                #     st.write(f"• ... and {len(self.category_mapping) - 8} more")
            
            # Tips section
            st.subheader("Tips for Improvement")
            st.markdown("""
            **For accurate predictions:**
            - Include relevant skills and keywords
            - Mention your experience level
            - Use clear, professional language
            - Avoid excessive formatting
            - Include education and certifications
            """)
            
            # About section
            st.subheader("About This Model")
            st.markdown("""
            This classifier is trained on **real resume data** from Kaggle, 
            not synthetic data (like v1). It uses NLP techniques to analyze 
            resume content and predict job categories.
            
            **Key Features:**
            - Real-world training data
            - TF-IDF vectorization
            - Confidence scoring
            - Multiple file formats
            - Detailed probability breakdown
            """)

# Batch processing feature
def show_batch_processing():
    """
    Feature for processing multiple resumes
    """
    st.header("Batch Processing")
    st.markdown("Upload multiple resumes for batch analysis")
    
    uploaded_files = st.file_uploader(
        "Choose resume files",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload multiple resume files for batch processing"
    )
    
    if uploaded_files and st.button("Process all docuemnts"):
        app = ResumeClassifierApp()
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}...")
            
            # Extract text
            text = app.process_uploaded_file(file)
            if text:
                # Predict
                category, confidence, _ = app.predict_resume_category(text)
                results.append({
                    'Filename': file.name,
                    'Predicted Category': category or 'Error',
                    'Confidence': f"{confidence:.1%}" if confidence else 'N/A'
                })
            else:
                results.append({
                    'Filename': file.name,
                    'Predicted Category': 'Error - Could not extract text',
                    'Confidence': 'N/A'
                })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Show results
        status_text.text("✅ Processing complete!")
        
        if results:
            st.subheader("Batch Results")
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="resume_analysis_results.csv",
                mime="text/csv"
            )

def show_contact_page():
    """
    Contact information page
    """
    st.title("📞 Contact Information")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Profile image placeholder
        st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <div style='width: 150px; height: 150px; border-radius: 50%; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        margin: 0 auto; display: flex; align-items: center; 
                        justify-content: center; color: white; font-size: 48px;'>
                👩‍💻
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("## Nitya Gupta")
        st.markdown("### Student, Developer & ML Enthusiast")
        st.markdown("""
        Thank you for using the Resume Screening App! This project showcases 
        machine learning techniques applied to real-world resume classification. 
        Feel free to reach out for questions, feedback, or collaboration opportunities.
        """)
    
    st.markdown("---")
    
    # Contact links
    st.markdown("## 🔗 Get in Touch")
    
    # Create three columns for contact methods
    contact_col1, contact_col2, contact_col3 = st.columns(3)
    
    with contact_col1:
        st.markdown("""
        ### 📧 Email
        [guptanitya.147@gmail.com](mailto:guptanitya.147@gmail.com)
        
        Drop me an email for:
        - Project inquiries
        - Technical discussions
        - Collaboration opportunities
        """)
    
    with contact_col2:
        st.markdown("""
        ### 💼 LinkedIn
        [Connect on LinkedIn](https://linkedin.com/in/nitya-gupta-66361128a/)
        
        Let's connect for:
        - Professional networking
        - Career opportunities
        - Industry discussions
        """)
    
    with contact_col3:
        st.markdown("""
        ### 💻 GitHub
        [View my GitHub Profile](https://github.com/Nitya-01)
        
        Check out my code for:
        - Open source projects
        - Code samples
        - Technical contributions
        """)
    
    st.markdown("---")
    
    # Future Developments Section
    st.markdown("## Future Developments & Roadmap")
    st.markdown("*Exciting features coming soon!*")
    
    # Create tabs for different categories of improvements
    dev_tab1, dev_tab2, dev_tab3, dev_tab4 = st.tabs([
        "🤖 Base Enhancements", 
        "📊 Analytics & Insights", 
        "🔧 Technical Upgrades", 
        "💻 User Experience"
    ])
    
    with dev_tab1:
        st.markdown("### Machine Learning and Model-based Improvements")
        st.markdown("""
        **Advanced NLP Models:**
        - Integration of transformer models (BERT, RoBERTa)
        - Multilingual resume support with foreign language processing and format detection
        - Better context understanding and semantic analysis
        
        **Accuracy Enhancements:**
        - Ensemble models combining multiple algorithms
        - Active learning for continuous model improvement
        - Fine-tuning on domain-specific datasets
        
        **Smart Features:**
        - Skill extraction and matching
        - Experience level detection
        - Salary range prediction
        - Resume quality scoring and suggestions (eg:, according to ATS standards)
        """)
    
    with dev_tab2:
        st.markdown("### Analytics & Business Intelligence")
        st.markdown("""
        **Advanced Analytics Dashboard:**
        - Market trend analysis by job categories
        - Skill demand forecasting and gap analysis
        - Geographic job distribution insights
        
        **Detailed Reports:**
        - Batch processing analytics
        - Performance metrics visualization
        - Category distribution over time
        - Success rate tracking
        
        **Insights Generation:**
        - Resume optimization recommendations
        - Industry-specific keyword suggestions
        - Career path predictions
        """)
    
    with dev_tab3:
        st.markdown("### Technical Infrastructure")
        st.markdown("""
        **Cloud Integration:**
        - AWS/GCP deployment for scalability
        - API endpoints for third-party integration
        - Microservices architecture
        
        **Security & Privacy:**
        - End-to-end encryption for resume data
        - GDPR compliance features for user data protection
        - Secure file handling and storage
        
        **Performance Optimization:**
        - Real-time processing capabilities
        - Caching mechanisms for faster responses
        - Load balancing for high traffic, higher than development stage
        
        **Integration Capabilities:**
        - LinkedIn API integration
        - Other job board API connections
        - ATS (Applicant Tracking System) compatibility
        """)
    
    with dev_tab4:
        st.markdown("### User Experience Enhancements")
        st.markdown("""
        **UI/UX Integration:**
        - Modern, responsive design
        - Dark/light theme toggle
        - Mobile-optimized interface
        - Drag-and-drop file uploads
        
        **New Features:**
        - AI-based Resume builder integration
        - Interview question suggestions
        - Career advice recommendations
        - Job matching system
        
        **Collaboration Features:**
        - Multi-user workspace
        - Team analytics dashboard
        - Role-based access control
        - Comments and feedback system
        
        **Additional Platforms:**
        - Mobile app development
        - Browser extension
        - Desktop application
        """)
    
    # Project info
    st.markdown("## About this version")
    
    project_col1, project_col2 = st.columns(2)
    
    with project_col1:
        st.markdown("""
        ### Technical Stack
        - **Backend:** Python, Scikit-learn
        - **Frontend:** Streamlit
        - **ML Pipeline:** TF-IDF + SVM/Random Forest
        - **Data Processing:** Pandas, NumPy
        - **Visualization:** Plotly
        """)
    
    with project_col2:
        st.markdown("""
        ### Key Features
        - Real dataset training (Kaggle)
        - Multiple file format support
        - Batch processing capability
        - Interactive visualizations
        - Confidence scoring
        """)
    
    # Feedback section
    st.markdown("---")
    st.markdown("## 💭 Feedback & Suggestions")
    
    st.markdown("""
    Your feedback is valuable! If you have suggestions for improvements, 
    found any bugs or wanna contribute to this project, please feel free to:
    
    - 📝 Open an issue on GitHub
    - 📧 Send me an email with your thoughts
    - 💼 Connect with me on LinkedIn to discuss
    
    **Lte's connect and grow!** 🤝
    """)
    
    # Add some styling
    st.markdown("""
    <style>
    .contact-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Main app execution
def main():
    """
    Main function to run the Streamlit app"""
    # Navigation
    # st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📃 Single Resume Analysis", "📦 Batch Processing", "📞 Contact"]
    )
    
    if page == "📃 Single Resume Analysis":
        app = ResumeClassifierApp()
        app.run_app()
    
    elif page == "📦 Batch Processing":
        show_batch_processing()
        
    elif page == "📞 Contact":
        show_contact_page()

if __name__ == "__main__":
    main()