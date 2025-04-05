from flask import Flask, render_template, request, jsonify
import json
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
import difflib  # For fuzzy matching
import requests  # For API calls
import socket  # For dynamic IP detection

app = Flask(__name__)

# Hugging Face API Key 
HUGGINGFACE_API_KEY = ""

# Load dataset from JSON
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Load data
data = load_data('data.json')

# Normalize dataset for better matching
normalized_data = {item['question'].strip().lower(): item['answer'] for item in data}
all_questions = list(normalized_data.keys())  # Store all questions for fuzzy matching

# Initialize BM25 Document Store
document_store = InMemoryDocumentStore(use_bm25=True)

# Write data to document store
def write_data_to_store(data):
    documents = [{"content": item['answer'], "meta": {"question": item['question']}} for item in data]
    document_store.write_documents(documents)

# Initialize retriever and reader
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

# Create the RAG pipeline
pipeline = ExtractiveQAPipeline(reader, retriever)

# Write data to document store
write_data_to_store(data)

# Function to generate text using Hugging Face API
def generate_text_with_hf_api(prompt):
    url = "https://api-inference.huggingface.co/models/bigscience/bloom-560m"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_length": 100}}

    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result[0]["generated_text"]
    else:
        return "Sorry, I couldn't generate an answer."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def get_response():
    try:
        user_message = None

        # Support form data (web requests)
        if 'msg' in request.form:
            user_message = request.form['msg'].strip().lower()

        # Support JSON data (mobile app requests)
        elif request.is_json:
            data = request.get_json()
            if 'msg' in data:
                user_message = data['msg'].strip().lower()

        # Handle missing input
        if not user_message:
            return jsonify({"error": "Invalid request. 'msg' parameter is required."}), 400

        # Step 1: Check for exact match in dataset
        if user_message in normalized_data:
            return jsonify({"response": normalized_data[user_message]})

        # Step 2: Check for close matches using fuzzy matching
        close_matches = difflib.get_close_matches(user_message, all_questions, n=1, cutoff=0.8)
        if close_matches:
            return jsonify({"response": normalized_data[close_matches[0]]})

        # Step 3: Use BM25 + Reader for retrieval
        prediction = pipeline.run(
            query=user_message, 
            params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}}
        )

        # Extract answer
        if prediction['answers']:
            answer = prediction['answers'][0].answer
        else:
            # Step 4: Use Hugging Face API-generated response as fallback
            answer = generate_text_with_hf_api(user_message)

        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_greeting')
def get_greeting():
    return "Hello! I'm SheGuardian. How can I assist you today?"

# Dynamic IP detection function
def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except Exception as e:
        return "127.0.0.1"  # Fallback in case of error

if __name__ == '__main__':
    local_ip = get_ip_address()
    print(f"Server is running on: http://{local_ip}:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)