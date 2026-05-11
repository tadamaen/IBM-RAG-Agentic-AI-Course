"""Module for processing LinkedIn profile data."""

import json
import logging
from typing import Dict, List, Any, Optional
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from modules.llm_interface import create_watsonx_embedding
import config

logger = logging.getLogger(__name__)

# Function 1: split_profile_data -> Divides the profile data into manageable chunks
def split_profile_data(profile_data: Dict[str, Any]) -> List:
    try:
        profile_json = json.dumps(profile_data)
        document = Document(text = profile_json)
        splitter = SentenceSplitter(chunk_size = config.CHUNK_SIZE)
        nodes = splitter.get_nodes_from_documents([document])
        logger.info(f"Created {len(nodes)} nodes from profile data")
        return nodes
      
    except Exception as e:
        logger.error(f"Error in split_profile_data: {e}")
        return []

# Function 2: create_vector_database -> Creates a vector index from the chunks
def create_vector_database(nodes: List) -> Optional[VectorStoreIndex]:
    try:
        embedding_model = create_watsonx_embedding()
        index = VectorStoreIndex(nodes = nodes,
                                 embed_model = embedding_model,
                                 show_progress = True)
        logger.info("Vector database created successfully")
        return index
      
    except Exception as e:
        logger.error(f"Error in create_vector_database: {e}")
        return None

# [Optional] Function 3: verify_embeddings -> Ensures all embeddings were created properly
def verify_embeddings(index: VectorStoreIndex) -> bool:
    try:
        vector_store = index._storage_context.vector_store
        node_ids = list(index.index_struct.nodes_dict.keys())
        missing_embeddings = False

        for node_id in node_ids:
            embedding = vector_store.get(node_id)
            if embedding is None:
                logger.warning(f"Node ID {node_id} has a None embedding.")
                missing_embeddings = True
              
            else:
                logger.debug(f"Node ID {node_id} has a valid embedding.")
        
        if missing_embeddings:
            logger.warning("Some node embeddings are missing")
            return False
          
        else:
            logger.info("All node embeddings are valid")
            return True
          
    except Exception as e:
        logger.error(f"Error in verify_embeddings: {e}")
        return False
