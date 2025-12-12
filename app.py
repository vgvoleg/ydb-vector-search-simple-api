"""
Simple Flask API for vector similarity search in YDB without LangChain.
Based on the implementation from langchain_ydb/vectorstores.py
"""

import json
import logging
import os
import struct
from typing import Optional

import ydb_dbapi
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# # Enable YDB debug logging
# ydb_logger = logging.getLogger('ydb')
# ydb_logger.setLevel(logging.DEBUG)

# ydb_dbapi_logger = logging.getLogger('ydb_dbapi')
# ydb_dbapi_logger.setLevel(logging.DEBUG)

# Configuration from environment variables
YDB_HOST = os.getenv("YDB_HOST", "localhost")
YDB_PORT = int(os.getenv("YDB_PORT", "2136"))
YDB_DATABASE = os.getenv("YDB_DATABASE", "/local")
YDB_TABLE = os.getenv("YDB_TABLE", "ydb_langchain_store")
YDB_SECURE = os.getenv("YDB_SECURE", "false").lower() == "true"

# Column mapping
COLUMN_ID = os.getenv("COLUMN_ID", "id")
COLUMN_TITLE = os.getenv("COLUMN_TITLE", "title")
COLUMN_VENDOR = os.getenv("COLUMN_VENDOR", "vendor")
COLUMN_DESCRIPTION = os.getenv("COLUMN_DESCRIPTION", "description")
COLUMN_EMBEDDING = os.getenv("COLUMN_EMBEDDING", "embedding")

# Search strategy
SEARCH_STRATEGY = os.getenv("SEARCH_STRATEGY", "CosineSimilarity")
SORT_ORDER = "DESC" if SEARCH_STRATEGY.endswith("Similarity") else "ASC"

# Vector index settings (если используется)
INDEX_ENABLED = os.getenv("INDEX_ENABLED", "false").lower() == "true"
INDEX_NAME = os.getenv("INDEX_NAME", "ydb_vector_index")
INDEX_TREE_SEARCH_TOP_SIZE = int(os.getenv("INDEX_TREE_SEARCH_TOP_SIZE", "1"))

# Vector pass as bytes (как в оригинальной реализации)
VECTOR_PASS_AS_BYTES = os.getenv("VECTOR_PASS_AS_BYTES", "true").lower() == "true"


def get_connection():
    """Create YDB connection"""
    logger.debug(f"Creating connection to {YDB_HOST}:{YDB_PORT}/{YDB_DATABASE}")
    return ydb_dbapi.connect(
        host=YDB_HOST,
        port=YDB_PORT,
        database=YDB_DATABASE,
        protocol="grpcs" if YDB_SECURE else "grpc",
    )


def check_connection():
    """Check YDB connection with a simple query"""
    try:
        logger.info("Testing YDB connection...")
        connection = get_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 as test")
            result = cursor.fetchone()
            connection.close()
            logger.info("Connection test successful")
            return result[0] == 1
    except Exception as e:
        logger.error(f"Connection check failed: {str(e)}")
        return False


def get_query_embedding(query: str) -> list[float]:
    """
    Get embedding for a text query.

    TODO: Implement this function to generate embeddings from text queries.
    This could call an embedding service, model API, etc.

    Args:
        query: Text query to convert to embedding

    Returns:
        List of floats representing the embedding vector

    Raises:
        NotImplementedError: This function is not yet implemented
    """
    logger.warning(f"Attempted to use unimplemented query embedding for: {query[:50]}...")
    raise NotImplementedError(
        "Query embedding generation is not yet implemented. "
        "Please provide 'embedding' parameter directly instead of 'query'."
    )


def convert_vector_to_bytes(vector: list[float]) -> bytes:
    """Convert vector to bytes format for YDB"""
    if VECTOR_PASS_AS_BYTES:
        b = struct.pack("f" * len(vector), *vector)
        return b + b'\x01'
    return vector


def prepare_search_query(k: int, filter_params: Optional[dict] = None) -> str:
    """Prepare SQL query for vector search"""

    # WHERE clause для фильтрации
    where_statement = ""
    if filter_params:
        if INDEX_ENABLED:
            raise ValueError("Unable to use filter with enabled vector index.")

        stmts = []
        # Фильтрация по строковым полям title, vendor, description
        if "title" in filter_params:
            stmts.append(f'{COLUMN_TITLE} = "{filter_params["title"]}"')
        if "vendor" in filter_params:
            stmts.append(f'{COLUMN_VENDOR} = "{filter_params["vendor"]}"')
        if "description" in filter_params:
            stmts.append(f'{COLUMN_DESCRIPTION} = "{filter_params["description"]}"')

        if stmts:
            where_statement = f"WHERE {' AND '.join(stmts)}"

    # Pragma для индекса (если включен)
    pragma_statement = ""
    if INDEX_ENABLED:
        pragma_statement = f"""
        PRAGMA ydb.KMeansTreeSearchTopSize="{INDEX_TREE_SEARCH_TOP_SIZE}";
        """

    # VIEW индекса (если включен)
    view_index = ""
    if INDEX_ENABLED:
        view_index = f"VIEW {INDEX_NAME}"

    # DECLARE embedding в зависимости от формата
    if VECTOR_PASS_AS_BYTES:
        declare_embedding = """
        DECLARE $embedding as String;

        $TargetEmbedding = $embedding;
        """
    else:
        declare_embedding = """
        DECLARE $embedding as List<Float>;

        $TargetEmbedding = Knn::ToBinaryStringFloat($embedding);
        """

    return f"""
    {pragma_statement}

    {declare_embedding}

    SELECT
        {COLUMN_ID} as id,
        {COLUMN_TITLE} as title,
        {COLUMN_VENDOR} as vendor,
        {COLUMN_DESCRIPTION} as description,
        Knn::{SEARCH_STRATEGY}({COLUMN_EMBEDDING}, $TargetEmbedding) as score
    FROM {YDB_TABLE} {view_index}
    {where_statement}
    ORDER BY score {SORT_ORDER}
    LIMIT {k};
    """


def execute_search(embedding: list[float], k: int, filter_params: Optional[dict] = None):
    """Execute vector similarity search"""
    logger.info(f"Executing search with k={k}, embedding_dim={len(embedding)}, filter={filter_params}")
    connection = get_connection()

    try:
        query = prepare_search_query(k, filter_params)
        logger.debug(f"Prepared query: {query[:200]}...")

        # Prepare embedding parameter
        embedding_value = convert_vector_to_bytes(embedding)
        logger.debug(f"Converted embedding to bytes: {len(embedding_value) if isinstance(embedding_value, bytes) else len(embedding_value)} bytes/elements")

        with connection.cursor() as cursor:
            logger.info("Executing YDB query...")
            cursor.execute(query, {"$embedding": embedding_value})

            if cursor.description is None:
                logger.warning("Query returned no description")
                return []

            columns = [col[0] for col in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

            logger.info(f"Query returned {len(results)} results")
            return results

    except Exception as e:
        logger.error(f"Search execution failed: {str(e)}")
        raise
    finally:
        connection.close()
        logger.debug("Connection closed")


@app.route('/')
def index():
    """Render web interface"""
    logger.debug("Serving web interface")
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    logger.debug("Health check requested")
    return jsonify({"status": "ok"})


@app.route('/search', methods=['POST'])
def search():
    """
    Vector similarity search endpoint

    Request body:
    {
        "embedding": [0.1, 0.2, 0.3, ...],  # вектор для поиска (используйте embedding ИЛИ query)
        "query": "search text",              # текстовый запрос (используйте embedding ИЛИ query)
        "k": 4,  # количество результатов (опционально, по умолчанию 4)
        "filter": {  # опциональный фильтр по полям
            "title": "Some title",
            "vendor": "Some vendor",
            "description": "Some description"
        }
    }

    Response:
    {
        "results": [
            {
                "id": "item_id",
                "title": "Product Title",
                "vendor": "Vendor Name",
                "description": "Product description",
                "score": 0.95
            },
            ...
        ],
        "count": 4
    }
    """
    try:
        data = request.get_json()
        logger.info(f"Search request received from {request.remote_addr}")

        if not data:
            logger.warning("Empty request body")
            return jsonify({"error": "Request body is required"}), 400

        # Validate that exactly one of 'embedding' or 'query' is provided
        has_embedding = "embedding" in data
        has_query = "query" in data

        logger.debug(f"Request contains: embedding={has_embedding}, query={has_query}, k={data.get('k', 4)}")

        if not has_embedding and not has_query:
            logger.warning("Neither embedding nor query provided")
            return jsonify({
                "error": "Either 'embedding' or 'query' field is required"
            }), 400

        if has_embedding and has_query:
            logger.warning("Both embedding and query provided")
            return jsonify({
                "error": "Provide either 'embedding' or 'query', not both"
            }), 400

        # Get or generate embedding
        if has_query:
            query_text = data["query"]
            logger.info(f"Processing query: {query_text[:100]}...")
            if not isinstance(query_text, str) or not query_text.strip():
                return jsonify({"error": "query must be a non-empty string"}), 400

            try:
                embedding = get_query_embedding(query_text)
            except NotImplementedError as e:
                logger.warning("Query embedding not implemented")
                return jsonify({"error": str(e)}), 501
        else:
            embedding = data["embedding"]
            logger.info(f"Using provided embedding (dimension: {len(embedding) if isinstance(embedding, list) else 'invalid'})")
            # Validate embedding format
            if not isinstance(embedding, list):
                logger.error("Invalid embedding format")
                return jsonify({"error": "embedding must be a list of floats"}), 400

        k = data.get("k", 4)
        filter_params = data.get("filter", None)

        # Validate k
        if not isinstance(k, int) or k <= 0:
            logger.error(f"Invalid k value: {k}")
            return jsonify({"error": "k must be a positive integer"}), 400

        # Execute search
        results = execute_search(embedding, k, filter_params)

        logger.info(f"Search completed successfully, returning {len(results)} results")
        return jsonify({
            "results": results,
            "count": len(results)
        })

    except Exception as e:
        logger.error(f"Error during search: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    logger.debug("Configuration requested")
    return jsonify({
        "host": YDB_HOST,
        "port": YDB_PORT,
        "database": YDB_DATABASE,
        "table": YDB_TABLE,
        "search_strategy": SEARCH_STRATEGY,
        "index_enabled": INDEX_ENABLED,
        "vector_pass_as_bytes": VECTOR_PASS_AS_BYTES
    })


if __name__ == '__main__':
    # Check database connection before starting the server
    logger.info("=" * 60)
    logger.info("YDB Vector Search API - Starting up")
    logger.info("=" * 60)

    logger.info("Configuration:")
    logger.info(f"  YDB Host: {YDB_HOST}:{YDB_PORT}")
    logger.info(f"  Database: {YDB_DATABASE}")
    logger.info(f"  Table: {YDB_TABLE}")
    logger.info(f"  Strategy: {SEARCH_STRATEGY}")
    logger.info(f"  Index enabled: {INDEX_ENABLED}")

    print("\nChecking YDB connection...")
    if check_connection():
        print(f"✓ Connected to YDB: {YDB_HOST}:{YDB_PORT}/{YDB_DATABASE}")
        print(f"✓ Using table: {YDB_TABLE}")
        logger.info("Database connection successful")
    else:
        print(f"✗ Failed to connect to YDB: {YDB_HOST}:{YDB_PORT}/{YDB_DATABASE}")
        print("Please check your .env configuration")
        logger.error("Database connection failed - exiting")
        exit(1)

    flask_host = os.getenv("FLASK_HOST", "::")
    flask_port = int(os.getenv("FLASK_PORT", "5000"))
    flask_debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"

    print(f"\nStarting server on {flask_host}:{flask_port}")
    print(f"Web interface: http://{flask_host if flask_host != '0.0.0.0' else 'localhost'}:{flask_port}")
    logger.info(f"Starting Flask server on {flask_host}:{flask_port} (debug={flask_debug})")
    logger.info("=" * 60)

    app.run(host=flask_host, port=flask_port, debug=flask_debug)
