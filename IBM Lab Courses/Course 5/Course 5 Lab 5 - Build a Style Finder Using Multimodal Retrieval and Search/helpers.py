import logging
import re

# Set up logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to get all items related to a specific image from the dataset
def get_all_items_for_image(image_url, dataset):
    related_items = dataset[dataset['Image URL'] == image_url]
    logger.info(f"Found {len(related_items)} items related to image URL: {image_url}")
    return related_items

# Function to append alternatives to the user response in a formatted way
def format_alternatives_response(user_response, alternatives, similarity_score, threshold = 0.8):
    if not user_response or any(phrase in user_response for phrase in ["I'm not able to provide",  "I cannot",  "I apologize, but", "I don't feel comfortable"]):
        user_response = "## Fashion Analysis Results\n\nHere are the items detected in your image:"
    if similarity_score >= threshold:
        enhanced_response = user_response + "\n\n## Similar Items Found\n\nHere are some similar items we found:\n"
    else:
        enhanced_response = user_response + "\n\n## Similar Items Found\n\nHere are some visually similar items:\n"

    items_added = 0
    max_items = 10
    
    for item, alts in alternatives.items():
        enhanced_response += f"\n### {item}:\n"
        if alts:
            for alt in alts[:3]:
                if items_added < max_items:
                    enhanced_response += f"- {alt['title']} for {alt['price']} from {alt['source']} ([Buy it here]({alt['link']}))\n"
                    items_added += 1
        else:
            enhanced_response += "- No alternatives found.\n"
    
    return enhanced_response
