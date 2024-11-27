from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import csv
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
from flask_cors import CORS
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout
import json

app = Flask(__name__)

CORS(app)

CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Set the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_WORDS = 10000
MAX_LEN = 200

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

base_save_path = 'C:/Users/mrajput1/Downloads/Grad Project/Server/'
save_path_model = base_save_path + 'model/lstm_base_model_30k.h5'
save_path_tokenizer = base_save_path + 'model/tokenizer.json'
pretrained_path = base_save_path + 'uploads/IMDB Dataset.csv'

os.makedirs(os.path.dirname(save_path_model), exist_ok=True)

# Load the model globally when the app starts
loaded_model = None
tokenizer = None
X_test = None
Y_test = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def knn_anomaly(matrix, data):
    k = 7  # Number of neighbors
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)

    distances, indices = knn.kneighbors(X)
    average_distances = np.mean(distances, axis=1)

    threshold = np.percentile(average_distances, 75)  # For example, top 5% as anomalies

    anomalies = average_distances > threshold
    data['anomaly'] = anomalies.astype(int)  # Convert boolean to int

    if 'true_anomaly' in data.columns:
        accuracy = accuracy_score(data['true_anomaly'], data['anomaly'])
        print(f'Accuracy of anomaly detection: {accuracy:.2f}')
    else:
        print('True anomaly labels not found in the DataFrame.')
    # Assuming df has columns 'review' and 'sentiment'
    reviews = sample_knn['review'].values

    # Convert text data to TF-IDF features
    vectorizer = TfidfVectorizer()
    Y = vectorizer.fit_transform(reviews)

    # Initialize KNN
    k = 7  # Number of neighbors
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(Y)

    # Find the distances and indices of the k nearest neighbors
    distances, indices = knn.kneighbors(Y)

    # Calculate anomaly score based on average distance to neighbors
    # A high average distance indicates an anomaly
    average_distances = np.mean(distances, axis=1)

    # Set a threshold for anomaly detection
    threshold = np.percentile(average_distances, 75)  # For example, top 5% as anomalies

    #Add the anomaly scores to the DataFrame
    sample_knn['anomaly_score_algo2'] = average_distances

    # Identify anomalies
    anomalies = (average_distances > threshold).astype(int)

    # Add anomaly detection results to the dataframe
    sample_knn['anomaly'] = anomalies

    sampled_data['anomaly_score_algo2'] = average_distances
    sampled_data['predicted_anomaly_algo2'] = anomalies

def isolatedForest_anomaly(matrix, data):
    model = IsolationForest(contamination=0.05)  # Adjust contamination as needed
    model.fit(matrix.toarray())

    data['anomaly'] = model.predict(matrix.toarray())
    data['anomaly'] = data['anomaly'].map({1: 0, -1: 1}).fillna(0).astype(int)   # Map to 0 for normal, 1 for anomalous

    if 'true_anomaly' in data.columns:
        accuracy = accuracy_score(data['true_anomaly'], data['anomaly'])
        print(f'Accuracy of anomaly detection: {accuracy:.2f}')
    else:
        print('True anomaly labels not found in the DataFrame.')
    return data.copy()

def remove_anomalies(df):
    return df[df['anomaly'] != 1]

def anomalyComparisons(df):
    #Random Forest
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['review'])

    # Train Isolation Forest
    model = IsolationForest(contamination=0.05)  # Adjust contamination as needed
    model.fit(X.toarray())

    # Calculate anomaly scores using decision_function
    # Anomaly scores: Lower values indicate anomalies
    anomaly_scores = model.decision_function(X.toarray())
    # Add the anomaly scores to the DataFrame
    df['anomaly_score_algo1'] = anomaly_scores

    # Predict anomalies
    df['anomaly1'] = model.predict(X.toarray())
    df['anomaly1'] = df['anomaly1'].map({1: 0, -1: 1}).fillna(0).astype(int)  # Map to 0 for normal, 1 for anomalous
    df['predicted_anomaly_algo1'] = df['anomaly1'].map({1: 0, -1: 1}).fillna(0).astype(int)
    anomalies_rf = df[df['anomaly1'] == 1]
    

    # Convert text data to TF-IDF features
    vectorizer = TfidfVectorizer()
    Y = vectorizer.fit_transform(df['review'].values)

    # Initialize KNN
    k = 7  # Number of neighbors
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(Y)

    # Find the distances and indices of the k nearest neighbors
    distances, indices = knn.kneighbors(Y)

    # Calculate anomaly score based on average distance to neighbors
    # A high average distance indicates an anomaly
    average_distances = np.mean(distances, axis=1)

    # Set a threshold for anomaly detection
    threshold = np.percentile(average_distances, 75)  # For example, top 5% as anomalies

    #Add the anomaly scores to the DataFrame
    df['anomaly_score_algo2'] = average_distances

    # Identify anomalies
    anomalies = (average_distances > threshold).astype(int)

    # Add anomaly detection results to the dataframe
    df['anomaly2'] = anomalies

    df['predicted_anomaly_algo2'] = anomalies

    response = {}
    # Accuracy comparison (bar plot)
    accuracy_algo1 = 0.91  # Replace with actual value from your model
    accuracy_algo2 = 0.72  # Replace with actual value from your model
    
    response['accuracy_algo1'] = accuracy_algo1
    response['accuracy_algo2'] = accuracy_algo2

    algorithms = ['Algorithm 1', 'Algorithm 2']
    accuracies = [accuracy_algo1, accuracy_algo2]

    # Confusion Matrix
    true_labels = df['true_anomaly']
    predictions_algo1 = df['predicted_anomaly_algo1']
    predicted_probs_algo1 = df['anomaly_score_algo1']
    predictions_algo2 = df['predicted_anomaly_algo2']
    predicted_probs_algo2 = df['anomaly_score_algo2']
    cm_algo1 = confusion_matrix(true_labels, predictions_algo1)
 
    cm_algo2 = confusion_matrix(true_labels, predictions_algo2)
    
    # Convert confusion matrices to lists
    response['confusion_matrix_algo1'] = cm_algo1.tolist()
    response['confusion_matrix_algo2'] = cm_algo2.tolist()

    # # ROC Curve
    fpr1, tpr1, _ = roc_curve(true_labels, predicted_probs_algo1)
    roc_auc1 = auc(fpr1, tpr1)
    roc_curve_algo1 = {
        'fpr': fpr1.tolist(),
        'tpr': tpr1.tolist(),
        'auc': roc_auc1
    }

    fpr2, tpr2, _ = roc_curve(true_labels, predicted_probs_algo2)
    roc_auc2 = auc(fpr2, tpr2)
    roc_curve_algo2 = {
        'fpr': fpr2.tolist(),
        'tpr': tpr2.tolist(),
        'auc': roc_auc2
    }


    response['roc_curve_algo1'] = roc_curve_algo1
    response['roc_curve_algo2'] = roc_curve_algo2

    # # Precision-Recall Curve
    precision1, recall1, _ = precision_recall_curve(true_labels, predicted_probs_algo1)
    precision_recall_algo1 = {
        'precision': precision1.tolist(),
        'recall': recall1.tolist()
    }

    precision2, recall2, _ = precision_recall_curve(true_labels, predicted_probs_algo2)
    precision_recall_algo2 = {
        'precision': precision2.tolist(),
        'recall': recall2.tolist()
    }

    response['precision_recall_algo1'] = precision_recall_algo1
    response['precision_recall_algo2'] = precision_recall_algo2

    # Anomaly scores (convert Series to list before returning)
    response['anomaly_scores_algo1'] = df['anomaly_score_algo1'].tolist()
    response['anomaly_scores_algo2'] = df['anomaly_score_algo2'].tolist()
    return response

def preprocess_data(df):
    """
    Function to preprocess the data. Modify this function to include any necessary preprocessing steps.
    For example:
    - Handling missing values
    - Encoding categorical variables
    - Scaling/normalizing data
    - Renaming columns, etc.
    """
    sample_data = df.copy()
    # Example Preprocessing Steps:

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sample_data['review'])

    return X, sample_data

def adversarial_model(df):
    adv_reviews = df['review']
    adv_labels = df['sentiment'].map({'positive': 1, 'negative': 0}).astype(int).tolist()
    
    adv_tokenizer = Tokenizer(num_words=MAX_WORDS)
    adv_tokenizer.fit_on_texts(adv_reviews)
    sequences_adv = adv_tokenizer.texts_to_sequences(adv_reviews)
    data_adv = pad_sequences(sequences_adv, maxlen=MAX_LEN)

    # Splitting the data
    x_train_adv = np.array(data_adv)
    y_train_adv = np.array(adv_labels)

    input_layer_adv = Input(shape=(MAX_LEN,))

    # Embedding layer
    embedding_adv = Embedding(input_dim=MAX_WORDS, output_dim=256, input_length=MAX_LEN)(input_layer_adv)

    # LSTM layers
    lstm1_adv = LSTM(128, return_sequences=True)(embedding_adv)
    dropout1_adv = Dropout(0.5)(lstm1_adv)
    lstm2_adv = LSTM(128, return_sequences=True)(dropout1_adv)
    lstm3_adv = LSTM(128)(lstm2_adv)

    # Fully connected output layer
    output_adv = Dense(1, activation='sigmoid')(lstm3_adv)

    model_adv = Model(inputs=input_layer_adv, outputs=output_adv)
    model_adv.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_adv.fit(x_train_adv, y_train_adv, epochs=15, batch_size=64)

    predictions_model1 = loaded_model.predict(X_test)
    predictions_model2 = model_adv.predict(X_test)

    # Averaging the predictions
    final_predictions1 = (predictions_model1 + predictions_model2) / 2

    # If you need binary outputs
    final_predictions_binary1 = (final_predictions1 > 0.5).astype(int)

    accuracy = accuracy_score(Y_test, final_predictions_binary1)
    threshold = 0.82  # Set your desired threshold
    if accuracy > threshold:
        print("Threshold met! Merging embeddings...")

        # Extract embeddings (assume the last LSTM layer outputs embeddings)
        embeddings_loaded = loaded_model.layers[-2].output
        embeddings_new = modal_adv.layers[-2].output

        # Merge embeddings: weighted average or concatenation
        merged_embeddings = Concatenate()([embeddings_loaded, embeddings_new])

        hybrid_output = loaded_model.layers[-1](merged_embeddings)  # Reuse the final output layer
        hybrid_model = Model(inputs=loaded_model.input, outputs=hybrid_output)

        # Compile the hybrid model
        hybrid_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Hybrid model created!")
        hybrid_model.summary()
    else:
        print("Accuracy below threshold. No merging performed.")
    


# This function will load the model once at the start
def load_the_model():
    global loaded_model, tokenizer, X_test, Y_test
    loaded_model = load_model(save_path_model)
    print("Model loaded successfully.")
    # Load the tokenizer from the saved JSON file
    with open(save_path_tokenizer, 'r') as f:
        tokenizer_json = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_json)
    print("Tokenizer loaded successfully.")
    reviews = []
    labels = []
    with open(pretrained_path, mode='r', encoding='utf-8') as file:
        idx = 0
        reader = csv.DictReader(file)
        for row in reader:
          idx = idx + 1
          reviews.append(row['review'])
          labels.append(1 if row['sentiment'] == 'positive' else 0)
    
    X_test = reviews[35000:39000]
    Y_test = labels[35000:39000]

    sequences = tokenizer.texts_to_sequences(X_test)
    data = pad_sequences(sequences, maxlen=MAX_LEN)
    X_test = np.array(data)
    Y_test = np.array(Y_test)
    




@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Save the file
        # filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        # file.save(filepath)
        try:
            df = pd.read_csv(file)
            matrix, sample_data = preprocess_data(df)
            result = anomalyComparisons(sample_data)
            return jsonify({"message": "Upload successful", "data": result}), 200
        except Exception as e:
            return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file format. Only CSV files are allowed."}), 400

@app.route('/run_adversarial_model', methods=['POST'])
def run_adversarial_model():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Save the file
        # filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        # file.save(filepath)
        try:
            df = pd.read_csv(file)
            matrix, sample_forest = preprocess_data(df)
            sample_forest = isolatedForest_anomaly(matrix, sample_forest)
            sample_forest = remove_anomalies(sample_forest)
            adversarial_model(sample_forest)
            return jsonify({"message": "Upload successful", "data_preview": data_preview}), 200
        except Exception as e:
            return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file format. Only CSV files are allowed."}), 400

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data.get('review', None)
    
    if not review:
        return jsonify({"error": "No review provided"}), 400
    
    sequence = tokenizer.texts_to_sequences([review])
    data = pad_sequences(sequence, maxlen=MAX_LEN)
    
    # Make prediction using the loaded model
    prediction = loaded_model.predict(data)
    
    # The model's output is likely a probability (e.g., between 0 and 1), so we'll classify it
    sentiment = 'positive' if prediction[0] > 0.5 else 'negative'
    
    return jsonify({"review": review, "sentiment": sentiment, "confidence": float(prediction[0])})

# Load the model when the app starts
load_the_model()

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)