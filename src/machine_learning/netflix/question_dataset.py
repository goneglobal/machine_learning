import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pickle
import re
from typing import List, Dict, Tuple
import tensorflowjs as tfjs

class NetflixQANeuralNetwork:
    def __init__(self, csv_path: str):
        """
        Initialize Q&A system with custom neural network.
        
        Args:
            csv_path: Path to Netflix CSV file
        """
        print("=" * 70)
        print("Netflix Q&A Neural Network System")
        print("=" * 70)
        
        # Load data
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        print(f"✓ Loaded {len(self.df)} items from Netflix dataset")
        
        # Initialize components
        self.question_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.content_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.label_encoder = LabelEncoder()
        
        # Model will be created during training
        self.model = None
        self.question_types = ['count', 'list', 'find', 'detail', 'genre', 'year', 'country', 'rating']
        
        print(f"✓ System initialized")
    
    def _create_search_text(self, row) -> str:
        """Create searchable text from a row."""
        parts = []
        
        if pd.notna(row['title']):
            parts.append(str(row['title']))
        if pd.notna(row['type']):
            parts.append(str(row['type']))
        if pd.notna(row['director']):
            parts.append(str(row['director']))
        if pd.notna(row['cast']):
            cast_str = str(row['cast'])[:100]  # Limit cast length
            parts.append(cast_str)
        if pd.notna(row['country']):
            parts.append(str(row['country']))
        if pd.notna(row['release_year']):
            parts.append(str(row['release_year']))
        if pd.notna(row['rating']):
            parts.append(str(row['rating']))
        if pd.notna(row['listed_in']):
            parts.append(str(row['listed_in']))
        if pd.notna(row['description']):
            desc = str(row['description'])[:200]  # Limit description
            parts.append(desc)
        
        return " ".join(parts)
    
    def _generate_training_data(self) -> Tuple[List[str], List[str], List[int]]:
        """Generate synthetic training data for question-answer pairs."""
        questions = []
        contents = []
        labels = []
        
        print("\n✓ Generating training data...")
        
        # Create content representations
        self.df['content_text'] = self.df.apply(self._create_search_text, axis=1)
        
        # Generate questions for each item
        for idx, row in self.df.iterrows():
            if idx % 1000 == 0:
                print(f"  Processing item {idx}/{len(self.df)}...")
            
            title = str(row['title']) if pd.notna(row['title']) else "Unknown"
            content = row['content_text']
            
            # Type 0: Count questions
            if pd.notna(row['type']):
                questions.append(f"how many {row['type'].lower()}s are there")
                contents.append(content)
                labels.append(0)
            
            # Type 1: List questions
            if pd.notna(row['type']):
                questions.append(f"list {row['type'].lower()}s")
                contents.append(content)
                labels.append(1)
            
            # Type 2: Find specific title
            questions.append(f"find {title}")
            contents.append(content)
            labels.append(2)
            
            # Type 3: Detail questions
            questions.append(f"tell me about {title}")
            contents.append(content)
            labels.append(3)
            
            # Type 4: Genre questions
            if pd.notna(row['listed_in']):
                genres = str(row['listed_in']).split(',')[0].strip().lower()
                questions.append(f"show me {genres}")
                contents.append(content)
                labels.append(4)
            
            # Type 5: Year questions
            if pd.notna(row['release_year']):
                questions.append(f"movies from {row['release_year']}")
                contents.append(content)
                labels.append(5)
            
            # Type 6: Country questions
            if pd.notna(row['country']):
                country = str(row['country']).split(',')[0].strip()
                questions.append(f"shows from {country}")
                contents.append(content)
                labels.append(6)
            
            # Type 7: Rating questions
            if pd.notna(row['rating']):
                questions.append(f"content rated {row['rating']}")
                contents.append(content)
                labels.append(7)
        
        print(f"✓ Generated {len(questions)} training examples")
        return questions, contents, labels
    
    def build_model(self, question_dim: int, content_dim: int, num_classes: int):
        """
        Build neural network with Dense, ReLU, and Softmax layers.
        
        Architecture:
        - Input: Question features + Content features
        - Dense layer with ReLU activation
        - Dropout for regularization
        - Dense layer with ReLU activation
        - Output layer with Softmax activation
        """
        print("\n✓ Building neural network architecture...")
        
        # Input layers
        question_input = layers.Input(shape=(question_dim,), name='question_input')
        content_input = layers.Input(shape=(content_dim,), name='content_input')
        
        # Concatenate inputs
        concatenated = layers.Concatenate()([question_input, content_input])
        
        # Hidden layer 1 - Dense with ReLU
        dense1 = layers.Dense(256, activation='relu', name='dense_relu_1')(concatenated)
        dropout1 = layers.Dropout(0.3)(dense1)
        
        # Hidden layer 2 - Dense with ReLU
        dense2 = layers.Dense(128, activation='relu', name='dense_relu_2')(dropout1)
        dropout2 = layers.Dropout(0.3)(dense2)
        
        # Hidden layer 3 - Dense with ReLU
        dense3 = layers.Dense(64, activation='relu', name='dense_relu_3')(dropout2)
        
        # Output layer - Dense with Softmax
        output = layers.Dense(num_classes, activation='softmax', name='softmax_output')(dense3)
        
        # Create model
        model = keras.Model(
            inputs=[question_input, content_input],
            outputs=output,
            name='netflix_qa_model'
        )
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Architecture:")
        model.summary()
        
        return model
    
    def train(self, epochs: int = 20, batch_size: int = 32):
        """Train the neural network."""
        print("\n" + "=" * 70)
        print("Training Neural Network")
        print("=" * 70)
        
        # Generate training data
        questions, contents, labels = self._generate_training_data()
        
        # Vectorize questions and contents
        print("\n✓ Vectorizing text data...")
        X_questions = self.question_vectorizer.fit_transform(questions).toarray()
        X_contents = self.content_vectorizer.fit_transform(contents).toarray()
        y = np.array(labels)
        
        print(f"  Question features: {X_questions.shape}")
        print(f"  Content features: {X_contents.shape}")
        print(f"  Labels: {y.shape}")
        
        # Split data
        indices = np.arange(len(questions))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        X_q_train = X_questions[train_idx]
        X_c_train = X_contents[train_idx]
        y_train = y[train_idx]
        
        X_q_test = X_questions[test_idx]
        X_c_test = X_contents[test_idx]
        y_test = y[test_idx]
        
        # Build model
        self.model = self.build_model(
            question_dim=X_questions.shape[1],
            content_dim=X_contents.shape[1],
            num_classes=len(self.question_types)
        )
        
        # Train model
        print("\n✓ Training model...")
        history = self.model.fit(
            [X_q_train, X_c_train],
            y_train,
            validation_data=([X_q_test, X_c_test], y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Evaluate
        print("\n✓ Evaluating model...")
        test_loss, test_accuracy = self.model.evaluate([X_q_test, X_c_test], y_test, verbose=0)
        print(f"  Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}")
        
        return history
    
    def predict_question_type(self, question: str) -> Tuple[str, float]:
        """Predict the question type using the neural network."""
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        
        # Vectorize question
        q_vec = self.question_vectorizer.transform([question]).toarray()
        
        # Use a dummy content vector (we'll search after classification)
        c_vec = np.zeros((1, self.content_vectorizer.max_features))
        
        # Predict
        predictions = self.model.predict([q_vec, c_vec], verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        
        return self.question_types[predicted_idx], confidence
    
    def _search_content(self, question: str, top_k: int = 10) -> pd.DataFrame:
        """Search for relevant content using TF-IDF similarity."""
        # Vectorize question and all content
        q_vec = self.content_vectorizer.transform([question]).toarray()
        content_vecs = self.content_vectorizer.transform(self.df['content_text']).toarray()
        
        # Calculate cosine similarity
        similarities = np.dot(content_vecs, q_vec.T).flatten()
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        result_df = self.df.iloc[top_indices].copy()
        result_df['similarity'] = similarities[top_indices]
        
        return result_df
    
    def _extract_filters(self, question: str) -> Dict:
        """Extract filters from question."""
        filters = {}
        q_lower = question.lower()
        
        # Year
        year_match = re.search(r'\b(19|20)\d{2}\b', question)
        if year_match:
            filters['year'] = int(year_match.group())
        
        # Type
        if 'movie' in q_lower and 'tv' not in q_lower:
            filters['type'] = 'Movie'
        elif 'tv show' in q_lower or 'series' in q_lower:
            filters['type'] = 'TV Show'
        
        # Rating
        ratings = ['PG', 'PG-13', 'R', 'TV-MA', 'TV-14', 'TV-PG', 'G']
        for rating in ratings:
            if rating.lower() in q_lower:
                filters['rating'] = rating
                break
        
        return filters
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to dataframe."""
        filtered = df.copy()
        
        if 'type' in filters:
            filtered = filtered[filtered['type'] == filters['type']]
        if 'year' in filters:
            filtered = filtered[filtered['release_year'] == filters['year']]
        if 'rating' in filters:
            filtered = filtered[filtered['rating'] == filters['rating']]
        
        return filtered
    
    def ask(self, question: str, verbose: bool = False) -> str:
        """Ask a question and get an answer."""
        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print(f"{'='*70}")
        
        if self.model is None:
            return "❌ Model not trained! Please call train() first."
        
        # Predict question type
        q_type, confidence = self.predict_question_type(question)
        if verbose:
            print(f"[Question Type: {q_type} (confidence: {confidence:.2%})]")
        
        # Extract filters
        filters = self._extract_filters(question)
        if verbose and filters:
            print(f"[Filters: {filters}]")
        
        # Search for relevant content
        relevant_df = self._search_content(question, top_k=50)
        relevant_df = self._apply_filters(relevant_df, filters)
        
        if verbose:
            print(f"[Found {len(relevant_df)} relevant items]")
        
        # Generate answer based on question type
        return self._generate_answer(question, relevant_df, q_type)
    
    def _generate_answer(self, question: str, df: pd.DataFrame, q_type: str) -> str:
        """Generate answer based on question type."""
        if len(df) == 0:
            return "I couldn't find any items matching your query."
        
        if q_type == 'count':
            return f"There are **{len(df)}** items matching your query."
        
        elif q_type == 'list':
            result = f"I found **{len(df)}** items. Here are the top 10:\n\n"
            for idx, (_, row) in enumerate(df.head(10).iterrows(), 1):
                result += f"{idx}. **{row['title']}** ({row['release_year']}) - {row['type']}\n"
            if len(df) > 10:
                result += f"\n...and {len(df) - 10} more."
            return result
        
        elif q_type in ['find', 'detail']:
            row = df.iloc[0]
            result = f"# {row['title']}\n\n"
            result += f"**Year:** {row['release_year']} | **Type:** {row['type']}\n"
            if pd.notna(row['director']):
                result += f"**Director:** {row['director']}\n"
            if pd.notna(row['cast']):
                result += f"**Cast:** {str(row['cast'])[:150]}...\n"
            if pd.notna(row['rating']):
                result += f"**Rating:** {row['rating']}\n"
            if pd.notna(row['listed_in']):
                result += f"**Genres:** {row['listed_in']}\n"
            if pd.notna(row['description']):
                result += f"\n**Description:** {row['description']}\n"
            return result
        
        else:
            result = f"Top {min(5, len(df))} results:\n\n"
            for idx, (_, row) in enumerate(df.head(5).iterrows(), 1):
                result += f"{idx}. **{row['title']}** ({row['release_year']})\n"
                if pd.notna(row['description']):
                    result += f"   {str(row['description'])[:100]}...\n\n"
            return result
    
    def save_model(self, model_dir: str = 'netflix_qa_model'):
        """Save the trained model and vectorizers."""
        print(f"\n✓ Saving model to {model_dir}/")
        
        # Save Keras model in new format
        self.model.save(f'{model_dir}/keras_model.keras')
        
        # Save vectorizers and metadata
        with open(f'{model_dir}/question_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.question_vectorizer, f)
        
        with open(f'{model_dir}/content_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.content_vectorizer, f)
        
        metadata = {
            'question_types': self.question_types,
            'vocab_size_q': int(len(self.question_vectorizer.vocabulary_)),
            'vocab_size_c': int(len(self.content_vectorizer.vocabulary_))
        }
        
        with open(f'{model_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model saved successfully!")
    
    def export_to_tensorflowjs(self, output_dir: str = 'tfjs_model'):
        """Export model to TensorFlow.js format for JavaScript."""
        print(f"\n✓ Exporting model to TensorFlow.js format...")
        
        if self.model is None:
            print("❌ No model to export. Train the model first!")
            return
        
        # Convert to TensorFlow.js format
        tfjs.converters.save_keras_model(self.model, output_dir)
        
        # Save vocabularies as JSON for JavaScript (convert int64 to int)
        question_vocab = {word: int(idx) for word, idx in self.question_vectorizer.vocabulary_.items()}
        content_vocab = {word: int(idx) for word, idx in self.content_vectorizer.vocabulary_.items()}
        
        with open(f'{output_dir}/question_vocab.json', 'w') as f:
            json.dump(question_vocab, f)
        
        with open(f'{output_dir}/content_vocab.json', 'w') as f:
            json.dump(content_vocab, f)
        
        # Save metadata
        metadata = {
            'question_types': self.question_types,
            'max_features': self.question_vectorizer.max_features,
            'ngram_range': list(self.question_vectorizer.ngram_range)
        }
        
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model exported to {output_dir}/")
        print(f"✓ Files: model.json, group1-shard*of*, question_vocab.json, content_vocab.json")
        print("\nTo use in JavaScript:")
        print("  1. npm install @tensorflow/tfjs")
        print("  2. Load model: tf.loadLayersModel('file://tfjs_model/model.json')")


def main():
    """Main function."""
    print("=" * 70)
    print("Netflix Q&A Neural Network - TensorFlow & scikit-learn")
    print("=" * 70)
    
    # Get CSV path
    csv_path = input("\nEnter path to Netflix CSV file: ").strip()
    
    try:
        # Initialize system
        qa = NetflixQANeuralNetwork(csv_path)
        
        # Train model
        print("\nTraining neural network...")
        train = input("Train model? (y/n, default: y): ").strip().lower()
        
        if train != 'n':
            epochs = int(input("Number of epochs (default: 20): ").strip() or "20")
            qa.train(epochs=epochs)
            
            # Save model
            save = input("\nSave model? (y/n, default: y): ").strip().lower()
            if save != 'n':
                qa.save_model()
            
            # Export to TensorFlow.js
            export = input("Export to TensorFlow.js for JavaScript? (y/n, default: y): ").strip().lower()
            if export != 'n':
                qa.export_to_tensorflowjs()
        
        # Interactive Q&A
        print("\n" + "=" * 70)
        print("Sample questions:")
        print("  • How many movies from 2021?")
        print("  • List horror movies")
        print("  • Tell me about Midnight Mass")
        print("  • Show me documentaries")
        print("=" * 70)
        
        while True:
            question = input("\nYour question (or 'quit'): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if question:
                answer = qa.ask(question, verbose=True)
                print("\nAnswer:")
                print("=" * 70)
                print(answer)
    
    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{csv_path}'")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()