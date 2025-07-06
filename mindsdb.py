import mindsdb_sdk
import logging
from datasets import load_dataset
import pandas as pd
import os
import re
from yaspin import yaspin
from fastapi import HTTPException
from dotenv import load_dotenv
import openai

load_dotenv()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
client = openai.OpenAI(api_key=openai_api_key)

# Configure logging
logger = logging.getLogger(__name__)

# Global variables to track MindsDB connection and initialization status
server = None
kb_initialized = False

def connect_to_mindsdb():
    """Connect to MindsDB server"""
    global server
    logger.info("Connecting to MindsDB server...")
    server = mindsdb_sdk.connect('http://127.0.0.1:47334')
    logger.info("Connected to MindsDB server")
    return server

def download_and_prepare_dataset():
    """Download and prepare IMDB Movies dataset"""
    logger.info("Downloading IMDB Movies dataset from Hugging Face...")
    dataset = load_dataset("jquigl/imdb-genres")
    df = pd.DataFrame(dataset["train"])
    logger.info(f"Dataset shape: {df.shape}")
    
    # Clean and prepare the dataset
    df = df.rename(columns={
        'movie title - year': 'movie_id',
        'expanded-genres': 'expanded_genres',
        'description': 'content'
    })
    
    def clean_movie_id(movie_id):
        if pd.isna(movie_id) or movie_id == '':
            return "unknown_movie"
        cleaned = str(movie_id)
        cleaned = re.sub(r"['\"\!\?\(\)\[\]\/\\*]", "", cleaned)
        cleaned = cleaned.replace("&", "and").replace(":", "_")
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned if cleaned else "unknown_movie"
    
    df['movie_id'] = df['movie_id'].apply(clean_movie_id)
    logger.info(f"Original dataset size: {len(df)}")
    df = df.drop_duplicates(subset=['movie_id'], keep='first')
    logger.info(f"After removing duplicates: {len(df)}")
    
    df = df.fillna({
        'movie_id': 'unknown_movie',
        'genre': 'unknown',
        'expanded_genres': '',
        'rating': 0.0,
        'content': ''
    })
    
    # Save the prepared dataset
    csv_path = os.path.abspath("imdb_movies_prepared.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Dataset prepared and saved to 'imdb_movies_prepared.csv'")
    
    return df

def upload_dataset_to_mindsdb(df):
    """Upload dataset to MindsDB files database"""
    files_db = server.get_database("files")
    table_name = "movies"
    
    try:
        files_db.tables.drop(table_name)
        logger.info(f"Dropped existing table {table_name}")
    except Exception:
        pass
    
    files_db.create_table(table_name, df)
    logger.info(f"Created table files.{table_name}")
    
    return table_name

def create_knowledge_base(model_name="text-embedding-3-small"):
    """Create knowledge base with OpenAI embedding model"""
    try:
        logger.info(f"Creating knowledge base 'movies_kb' with {model_name}...")
        server.query("DROP KNOWLEDGE_BASE IF EXISTS movies_kb;").fetch()
        
        # Use f-string formatting to properly inject the API key
        kb_creation_query = server.query(f"""
            CREATE KNOWLEDGE_BASE movies_kb
            USING
                embedding_model = {{
                    "provider": "openai",
                    "model_name": "{model_name}",
                    "api_key": "{openai_api_key}"
                }},
                metadata_columns = ['genre', 'expanded_genres', 'rating'],
                content_columns = ['content'],
                id_column = 'movie_id';
        """)
        kb_creation_query.fetch()
        logger.info("Created knowledge base 'movies_kb'")
        
    except Exception as e:
        logger.error(f"Knowledge base creation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create knowledge base: {str(e)}")

def insert_data_into_knowledge_base(limit=30, min_rating=8.5):
    """Insert data into knowledge base with configurable limits"""
    try:
        with yaspin(text="Inserting data into knowledge base (limited dataset)..."):
            insert_query = server.query(f"""
                INSERT INTO movies_kb
                SELECT movie_id,
                       genre,
                       expanded_genres,
                       rating,
                       content
                FROM files.movies
                WHERE rating >= {min_rating}
                LIMIT {limit}
                USING
                    track_column = movie_id
            """).fetch()
        logger.info("Data inserted successfully into movies_kb")
        
    except Exception as e:
        logger.error(f"Insert error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to insert data into knowledge base: {str(e)}")

def verify_knowledge_base():
    """Check row count in knowledge base and verify initialization"""
    global kb_initialized
    
    try:
        row_count_df = server.query("""
            SELECT COUNT(*) AS cnt
            FROM (SELECT id FROM movies_kb) AS t;
        """).fetch()
        row_count = int(row_count_df.at[0, 'cnt'])
        
        if row_count > 0:
            kb_initialized = True
            logger.info(f"Knowledge base 'movies_kb' is ready with {row_count:,} rows")
        else:
            logger.warning("Knowledge base exists but is empty")
            kb_initialized = False
            
        return row_count
        
    except Exception as e:
        logger.error(f"Error checking knowledge base row count: {e}")
        kb_initialized = False
        raise HTTPException(status_code=500, detail=f"Failed to verify knowledge base: {str(e)}")

async def initialize_mindsdb(limit=30, min_rating=8.5, model_name="text-embedding-3-small"):
    """Initialize MindsDB connection, prepare dataset, and set up knowledge base"""
    global server, kb_initialized
    
    try:
        # Step 1: Connect to MindsDB
        connect_to_mindsdb()
        
        # Step 2: Download and prepare dataset
        df = download_and_prepare_dataset()
        
        # Step 3: Upload dataset to MindsDB
        upload_dataset_to_mindsdb(df)
        
        # Step 4: Create knowledge base with OpenAI model
        create_knowledge_base(model_name=model_name)
        
        # Step 5: Insert data into knowledge base
        insert_data_into_knowledge_base(limit=limit, min_rating=min_rating)
        
        # Step 6: Verify knowledge base
        row_count = verify_knowledge_base()
        
        return {
            "status": "success", 
            "kb_initialized": kb_initialized, 
            "row_count": row_count if kb_initialized else 0
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize MindsDB: {e}")
        server = None
        kb_initialized = False
        raise HTTPException(status_code=500, detail=f"MindsDB initialization failed: {str(e)}")
