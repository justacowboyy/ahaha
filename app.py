from flask import Flask,render_template, request, jsonify
from langchain_groq import ChatGroq  # Ensure you have the correct library installed
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase
import os
from flask_cors import CORS

# Enable CORS
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
def home():
    return render_template('index.html')
# Environment variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load dataset
DATA_FILE = "dataset/imdb_top_1000.csv"
daata = pd.read_csv(DATA_FILE)


# Initialize Neo4j Driver
def init_neo4j():
    """Initialize the Neo4j database connection."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# Initialize ChatGroq Model
def init_chat_groq():
    """Initialize the ChatGroq model."""
    return ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0,
        groq_api_key=GROQ_API_KEY,
    )



# Query Cypher Database via LangChain and Neo4j
def generate_cypher_query(question, chat_model, schema, prompt_template):
    """
    Generate a Cypher query using ChatGroq for a given question.

    :param question: User's input question.
    :param chat_model: Instance of the ChatGroq model.
    :param schema: Schema for Cypher query generation.
    :param prompt_template: Prompt template to guide the model.
    :return: Generated Cypher query or error message.
    """
    # Format the prompt
    formatted_prompt = prompt_template.format(schema=schema, question=question)
    
    try:
        # Generate response from ChatGroq
        query_response = chat_model.invoke(formatted_prompt)
        return query_response.content if hasattr(query_response, "content") else str(query_response)
    except Exception as e:
        return str(e)



# Flask Route for Cypher Q&A
@app.route('/api/cypher', methods=['POST'])
def cypher_qa():
    """
    Handle API requests for generating Cypher queries.
    """
    data = request.get_json()
    question = data.get("query")
    
    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Schema and prompt for Cypher queries
    schema = """
    Node properties:
    Movie {title: STRING, poster_link: STRING, released_year: INTEGER, certificate: STRING, runtime: INTEGER, genre: STRING, imdb_rating: FLOAT, overview: STRING, meta_score: INTEGER, no_of_votes: INTEGER, gross: STRING}
    Genre {name: STRING}
    Director {name: STRING}
    Actor {name: STRING}

    Relationship properties:
    (:Movie)-[:IN_GENRE]->(:Genre)
    (:Director)-[:DIRECTED]->(:Movie)
    (:Actor)-[:ACTED_IN]->(:Movie)

    Important Notes:
    1. There is no direct relationship between a Movie and its Director. You can get Movies from Directors but not the other way around.
    2. A Movie can have multiple genres, separated by commas, so it is essential to account for that in the query.
    3. If the response is huge or a list, provide all results.
    """
    prompt_template = """
    Generate a Cypher query to retrieve the required information. Use the query results (context) to directly answer the question. If the context contains data, extract and summarize the relevant details. Only say 'I don't know the answer' if no relevant data exists in the generated context.
    Given the schema below, give an answer to user's question:

    {schema}
    If question has grammar mistakes, modify it, and if there are incomplete names, complete them. Example, if it has 'fincher', modify to 'David Fincher'.
    Question: {question}
    """

    chat_model = init_chat_groq()
    response = generate_cypher_query(question, chat_model, schema, prompt_template)
    return jsonify({"result": response})


# Run Flask App
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000,debug=False)
