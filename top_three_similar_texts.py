# INSTALLING AND IMPORTING LIBRARIES
# pip install chromadb==1.0.12
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install sentence-transformers==4.1.0

import chromadb
from chromadb.utils import embedding_functions

# Define the embedding function using SentenceTransformers
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name = "all-MiniLM-L6-v2")
client = chromadb.Client()
collection_name = "my_grocery_collection"

# Main function to interact with the Chroma DB
def main():
    try:
        # Create a collection in the Chroma database with a specified name, distance metric (cosine distance) and embedding function
        collection = client.create_collection(name = collection_name,
                                              metadata = {"description" : "A collection for storing grocery data"},
                                              configuration={"hnsw" : {"space" : "cosine"}, 
                                                             "embedding_function" : ef})
        print(f"Collection created: {collection.name}")

        # Array of grocery-related text items and ids
        texts = ['fresh red apples', 'organic bananas', 'ripe mangoes', 'whole wheat bread', 'farm-fresh eggs', 'natural yogurt', 'frozen vegetables',
                 'grass-fed beef', 'free-range chicken', 'fresh salmon fillet', 'aromatic coffee beans', 'pure honey', 'golden apple', 'red fruit']
        ids = [f"food_{index + 1}" for index, _ in enumerate(texts)]

        # Add documents and their corresponding IDs to the collection
        collection.add(documents = texts,
                       metadatas = [{"source" : "grocery_store", "category" : "food"} for _ in texts],
                       ids = ids)

        # Retrieve all the items stored in the collection
        all_items = collection.get()
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")

        # Function to perform a similarity search in the collection
        def perform_similarity_search(collection, all_items):
            try:
                # Define the query term you want to search for in the collection and perform a query to search for the most similar documents to the 'query_term
                query_term = "apple"
              
                # FOR THE MODIFIED EXERCISE:
                # query_term = ["red", "fresh"]
                # if isinstance(query_term, str):
                #    query_term = [query_term]
                
                results = collection.query(query_texts = [query_term],    # Change from [query_term] to query_term for the modified exercise
                                           n_results = 3)
                print(f"Query results for '{query_term}':")
                print(results)
        
                # Check if no results are returned or if the results array is empty
                if not results or not results['ids'] or len(results['ids'][0]) == 0:
                    print(f'No documents found similar to "{query_term}"')
                    return
        
                print(f'Top 3 similar documents to "{query_term}": ')
                for i in range(min(3, len(results['ids'][0]))):
                    doc_id = results['ids'][0][i]
                    score = results['distances'][0][i]
                    text = results['documents'][0][i]
                    if not text:
                        print(f' - ID: {doc_id}, Text: "Text not available", Score: {score:.3f}')
                    else:
                        print(f' - ID: {doc_id}, Text: "{text}", Score: {score:.3f}')

                # Use this for the modified example instead:
                # for q in range(len(query_term)):
                #     print(f'Top 3 similar documents to "{query_term[q]}": ')
                #     for i in range(min(3, len(results['ids'][q]))):
                #         doc_id = results['ids'][q][i]  
                #         score = results['distances'][q][i]  
                #         text = results['documents'][q][i]
                #         if not text:
                #             print(f' - ID: {doc_id}, Text: "Text not available", Score: {score:.3f}')
                #         else:
                #             print(f' - ID: {doc_id}, Text: "{text}", Score: {score:.3f}')

            except Exception as error:
                print(f"Error in similarity search: {error}")
              
        # Perform Similarity Search
        perform_similarity_search(collection, all_items)
      
    except Exception as error:
        print(f"Error: {error}")

if __name__ == "__main__":
    main()
