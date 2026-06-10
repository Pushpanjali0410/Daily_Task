import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class ClickbaitDetector:
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # Use single words, pairs, and triplets
            max_features=3000,
            stop_words='english'
        )
        self.model = LogisticRegression(random_state=42)
        self.is_trained = False
        
    def create_dataset(self):
        dataset = {
            "headline": [
                # Real News Headlines (0)
                "Scientists discover new treatment for Alzheimer's disease",
                "Federal Reserve raises interest rates by 0.5 percent",
                "UN passes resolution on climate change action",
                "NASA launches new mission to explore Jupiter's moon",
                "Stock market reaches all-time high after jobs report",
                "New study shows benefits of exercise for mental health",
                "President signs infrastructure bill into law",
                "Tech company announces quarterly earnings beat expectations",
                "WHO approves new vaccine for global distribution",
                "Olympic games to begin next month in Paris",
                "Researchers find link between sleep and memory",
                "Company donates million dollars to local schools",
                "Weather service issues warning for coastal areas",
                "New electric vehicle breaks efficiency records",
                "Hospital opens new wing for cancer treatment",
                
                # Clickbait Headlines (1)
                "You won't believe what this celebrity looks like now",
                "This one weird trick removes belly fat instantly",
                "Doctors hate this simple weight loss secret",
                "What happens next will leave you speechless",
                "Shocking video reveals truth about your food",
                "This mom's recipe changed everything overnight",
                "The secret that experts don't want you to know",
                "His transformation will shock you to the core",
                "Number 7 will make you cry tears of joy",
                "This simple hack will save you thousands of dollars",
                "You'll never guess what happened at the mall",
                "This is the most amazing thing you'll see today",
                "Warning: Don't watch this alone at night",
                "The government doesn't want you to see this",
                "This picture will restore your faith in humanity"
            ],
            "label": [0] * 15 + [1] * 15  # 0 = Real, 1 = Clickbait
        }
        
        df = pd.DataFrame(dataset)
        print(f"✅ Dataset created: {len(df)} headlines")
        print(f"   Real news: {sum(df['label'] == 0)} headlines")
        print(f"   Clickbait: {sum(df['label'] == 1)} headlines")
        return df
    
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Remove extra spaces
        text = ' '.join(text.split())
        return text
    
    def extract_clickbait_features(self, headline):
        features = {}
        headline_lower = headline.lower()
        
        # Clickbait trigger words
        trigger_words = ['you', 'will', 'believe', 'shocking', 'amazing', 
                        'secret', 'trick', 'hack', 'what happens', 
                        'doctors hate', 'won\'t believe', 'speechless',
                        'never guess', 'can\'t believe', 'mind-blowing']
        
        features['trigger_word_count'] = sum(1 for word in trigger_words if word in headline_lower)
        
        # Emotional intensity indicators
        features['exclamation_count'] = headline.count('!')
        features['question_count'] = headline.count('?')
        features['upper_case_ratio'] = sum(1 for c in headline if c.isupper()) / max(len(headline), 1)
        
        # Length features
        features['word_count'] = len(headline.split())
        features['char_count'] = len(headline)
        
        return features
    
    def prepare_features(self, df):
        # Preprocess all headlines
        processed_headlines = df['headline'].apply(self.preprocess_text)
        
        # Extract TF-IDF features
        tfidf_matrix = self.vectorizer.fit_transform(processed_headlines)
        
        # Extract custom features
        custom_features = df['headline'].apply(self.extract_clickbait_features)
        custom_df = pd.DataFrame(custom_features.tolist())
        
        # Combine features (convert to array for scipy sparse matrix)
        from scipy.sparse import hstack
        combined_features = hstack([tfidf_matrix, custom_df.values])
        
        return combined_features
    
    def train(self, test_size=0.3):
        """
        Train the clickbait detection model
        """
        print("\n" + "="*60)
        print("🚀 TRAINING CLICKBAIT DETECTION MODEL")
        print("="*60)
        
        # Create and prepare dataset
        df = self.create_dataset()
        
        # Prepare features
        print("\n📊 Extracting features from headlines...")
        X = self.prepare_features(df)
        y = df['label']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        # Train the model
        print("\n🤖 Training model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Model trained successfully!")
        print(f"📈 Test Accuracy: {accuracy:.2%}")
        
        # Detailed classification report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Real News', 'Clickbait']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n🔢 Confusion Matrix:")
        print(f"               Predicted")
        print(f"               Real  Clickbait")
        print(f"Actual Real    {cm[0,0]:3d}    {cm[0,1]:3d}")
        print(f"       Clickbait {cm[1,0]:3d}    {cm[1,1]:3d}")
        
        return accuracy
    
    def predict(self, headline):
        """
        Predict if a single headline is clickbait or real news
        """
        if not self.is_trained:
            raise Exception("Model not trained yet! Call train() first.")
        
        # Preprocess and extract features
        processed = self.preprocess_text(headline)
        tfidf_features = self.vectorizer.transform([processed])
        custom_features = pd.DataFrame([self.extract_clickbait_features(headline)])
        
        from scipy.sparse import hstack
        combined_features = hstack([tfidf_features, custom_features.values])
        
        # Make prediction
        prediction = self.model.predict(combined_features)[0]
        probability = self.model.predict_proba(combined_features)[0]
        
        result = {
            'headline': headline,
            'is_clickbait': bool(prediction),
            'type': 'Clickbait' if prediction == 1 else 'Real News',
            'confidence': float(max(probability)),
            'clickbait_probability': float(probability[1]),
            'real_probability': float(probability[0])
        }
        
        return result
    
    def analyze_batch(self, headlines):
        """
        Analyze multiple headlines at once
        """
        results = []
        for headline in headlines:
            results.append(self.predict(headline))
        return results

def interactive_test():
    print("\n" + "="*60)
    print("🎯 CLICKBAIT DETECTOR - INTERACTIVE MODE")
    print("="*60)
    
    # Initialize and train detector
    detector = ClickbaitDetector()
    detector.train()
    
    print("\n" + "="*60)
    print("💬 READY TO TEST HEADLINES")
    print("="*60)
    print("Enter headlines to check if they're clickbait (or 'quit' to exit)")
    print("-"*60)
    
    while True:
        headline = input("\n📝 Enter headline: ").strip()
        
        if headline.lower() == 'quit':
            print("\n👋 Goodbye!")
            break
        
        if not headline:
            print("⚠️ Please enter a headline")
            continue
        
        # Make prediction
        result = detector.predict(headline)
        
        # Display result with visual indicator
        print("\n" + "-"*40)
        print(f"📰 Headline: {result['headline']}")
        
        if result['is_clickbait']:
            print(f"⚠️  VERDICT: CLICKBAIT")
            print(f"🎣 This appears to be sensationalized content")
        else:
            print(f"✅ VERDICT: REAL NEWS")
            print(f"📰 This appears to be legitimate news")
        
        print(f"📊 Confidence: {result['confidence']:.1%}")
        print(f"   - Clickbait probability: {result['clickbait_probability']:.1%}")
        print(f"   - Real news probability: {result['real_probability']:.1%}")
        
        # Provide explanation
        if result['is_clickbait']:
            print("\n💡 Why this might be clickbait:")
            if 'you' in headline.lower() or 'your' in headline.lower():
                print("   • Uses direct address ('you'/'your') to create urgency")
            if '?' in headline:
                print("   • Uses questions to create curiosity gap")
            if '!' in headline:
                print("   • Uses exclamation for emotional emphasis")
            if any(word in headline.lower() for word in ['believe', 'shocking', 'amazing', 'secret', 'trick']):
                print("   • Contains emotional trigger words")
        else:
            print("\n💡 Why this appears legitimate:")
            print("   • Uses factual, objective language")
            print("   • No sensationalist trigger words")
            print("   • Professional tone without manipulation tactics")
        
        print("-"*40)

def demo_mode():
    print("\n" + "="*60)
    print("🎯 CLICKBAIT DETECTOR - DEMONSTRATION")
    print("="*60)
    
    # Initialize and train detector
    detector = ClickbaitDetector()
    detector.train()
    
    # Test cases
    test_headlines = [
        "Scientists discover new treatment for cancer",
        "You won't believe what this celebrity did next",
        "Federal Reserve announces interest rate decision",
        "This one weird trick will change your life forever",
        "New study shows benefits of regular exercise",
        "Shocking video reveals truth about your food",
        "Company reports quarterly earnings growth",
        "Doctors hate this simple weight loss secret"
    ]
    
    print("\n📊 Testing multiple headlines...\n")
    
    results = detector.analyze_batch(test_headlines)
    
    # Display results in a table format
    print(f"{'Type':<12} {'Confidence':<12} Headline")
    print("-" * 60)
    
    for result in results:
        type_icon = "⚠️ CLICKBAIT" if result['is_clickbait'] else "✅ REAL NEWS"
        confidence = f"{result['confidence']:.1%}"
        headline = result['headline'][:50] + "..." if len(result['headline']) > 50 else result['headline']
        
        print(f"{type_icon:<12} {confidence:<12} {headline}")
    
    print("\n" + "="*60)
    print("🎯 Accuracy Analysis:")
    print("="*60)
    
    # Calculate statistics
    correct_predictions = 0
    for result in results:
        # Simple heuristic for "correct" prediction based on known patterns
        is_clickbait_indicators = any(word in result['headline'].lower() for word in 
                                     ['believe', 'trick', 'secret', 'shocking', 'hate'])
        
        if (result['is_clickbait'] and is_clickbait_indicators) or \
           (not result['is_clickbait'] and not is_clickbait_indicators):
            correct_predictions += 1
    
    print(f"📈 Model correctly identified patterns in {correct_predictions}/{len(results)} test cases")
    print(f"📊 Success rate: {correct_predictions/len(results):.0%}")

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     🎯 CLICKBAIT DETECTION SYSTEM 🎯                    ║
    ║                                                          ║
    ║     Detect sensationalized headlines using AI            ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    while True:
        print("\n" + "="*40)
        print("MAIN MENU")
        print("="*40)
        print("1. 📊 Run Demonstration")
        print("2. 💬 Interactive Mode (Test your own headlines)")
        print("3. ❌ Exit")
        print("-"*40)
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            demo_mode()
        elif choice == '2':
            interactive_test()
        elif choice == '3':
            print("\n👋 Thank you for using Clickbait Detector!")
            break
        else:
            print("⚠️ Invalid choice. Please select 1, 2, or 3")

if __name__ == "__main__":
    main()
