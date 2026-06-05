from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import shutil
import io
import unittest
from unittest.mock import patch

FILEPATH = 'structured_restaurant_data.json'
BACKUP_PATH = 'structured_restaurant_data.json.bak'
EXAMPLE_RESTAURANT_PARAGRAPH = 'Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, **casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, it captures the salt-air energy of the coast through its signature beer-battered snapper tacos and zesty octopus ceviche, making it a premier spot for open-air dining near the pier. Price range:'

def load_data(file_path):
    """Load restaurant data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ File {file_path} not found. Starting with empty database.")
        return []
    except json.JSONDecodeError:
        print(f"⚠️ Error decoding {file_path}. Starting with empty database.")
        return []

def save_data(data, file_path):
    """Save restaurant data to JSON file"""
    try:
        # Create backup before saving
        if os.path.exists(file_path):
            shutil.copy(file_path, f"{file_path}.bak")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"❌ Error saving data: {e}")
        return False

# Update your restaurant_data_structure_prompt_generation
def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    EXAMPLE_OUTPUT = """{{"name": "Mar de Cortez",
                          "location": "Santa Monica",
                          "type": "casual taqueria",
                          "food_style": "Baja-style seafood",
                          "rating": 4.2,
                          "price_range": 1,
                          "signatures": ["beer-battered snapper tacos", "zesty octopus ceviche"],
                          "vibe": "salt-air energy",
                          "environment": "a premier sun-drenched spot for open-air dining near the pier.",
                          "shortcomings": []}}"""
    
    base_system_msg = f"""You are a data extraction assistant.
                          Your task is to extract restaurant information from an unstructured restaurant description
                          and convert it into a structured JSON object.

                          Rules:
                          - Return ONLY valid JSON.
                          - Do not include explanations, markdown, or extra text.
                          - Extract the following fields: name, location, type, food_style, rating, price_range, signatures, vibe, environment, shortcomings
                          - Convert the price range into an integer equal to the number of dollar signs.
                            Examples: $ -> 1, $$ -> 2, $$$ -> 3, $$$$ -> 4
                          - signatures must be returned as a list of signature dishes.
                          - shortcomings must be returned as a list.
                          - If no shortcomings are mentioned, return an empty list.
                          - rating should be returned as a numeric value."""

    base_user_prompt = f"""Task: Convert the restaurant description below into the specified JSON format.
                           Restaurant description: {restaurant_paragraph}
                           Example:
                           Input Restaurant Description: Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, **casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, it captures the salt-air energy of the coast through its signature beer-battered snapper tacos and zesty octopus ceviche, making it a premier spot for open-air dining near the pier. Price range: $
                           Output: {EXAMPLE_OUTPUT}"""

    return base_system_msg, base_user_prompt

# Might need to explain why we are using granite here (cheap)
def llm_model(system_msg, prompt_txt, params = None):
    model_id = "ibm/granite-4-h-small"
    project_id = "skills-network"
    credentials = Credentials(url = "https://us-south.ml.cloud.ibm.com")

    # Define the model by ModelInference
    model = ModelInference(model_id = model_id,
                           credentials = credentials,
                           project_id = project_id)

    # Define the messages
    messages = [{"role": "system", "content": system_msg},
                {"role": "user", "content": prompt_txt}]

    # Get the final response output and return it
    response = model.chat(messages = messages)
    return response["choices"][0]["message"]["content"]

def JSON_auto_repair_prompts(response, error_message):
    auto_repair_system_msg = """You are a strict JSON correction engine.
                                Your only task is to fix invalid or malformed JSON so that it becomes valid, schema-compliant JSON.
            
                                Rules:
                                - Output ONLY valid JSON. No explanations, no markdown, no extra text.
                                - Preserve as much of the original content as possible.
                                - Do NOT invent new information unless required to fix structural/schema issues.
                                - Follow the schema implied by the validation error message.
                                - Fix common issues such as:
                                  - Missing commas, brackets, or quotes
                                  - Incorrect data types (e.g., string vs number)
                                  - Extra trailing commas
                                  - Incorrect nesting or structure
                                - Ensure final output is parseable by a standard JSON parser.
                                """

    auto_repair_prompt = f"""You are given:
                             1. A candidate JSON output that failed validation: {response}
                             2. The validation error message: {error_message}

                             Task:
                             - Carefully analyze the error message to identify what is wrong.
                             - Repair the JSON so that it fully conforms to the required format.
                             - Keep all valid data unchanged unless it violates the schema.
                             - Return ONLY the corrected JSON output.
                            
                             Do NOT include any commentary or explanation.
                             Return only valid JSON."""

    return auto_repair_system_msg, auto_repair_prompt

def new_data_entry_process(paragraph, itemId):
    """
    Process a new restaurant paragraph and structure it into a JSON object
    
    Args:
        paragraph (str): The restaurant description paragraph
        itemId (int): The unique ID for this restaurant item
    
    Returns:
        dict: Structured restaurant data with itemId included
    """
    # Define the Restaurant schema for validation
    class Restaurant(BaseModel):
        name: str
        location: str
        type: str
        food_style: str
        rating: Optional[float] = None
        price_range: Optional[int] = None
        signatures: List[str] = Field(default_factory = list)
        vibe: Optional[str] = None
        environment: str
        shortcomings: List[str] = Field(default_factory = list)
    
    # Step 1: Generate prompts for structuring the restaurant data
    system_msg, user_prompt = restaurant_data_structure_prompt_generation(paragraph)
    
    # Step 2: Get initial LLM generation
    output = llm_model(system_msg = system_msg, prompt_txt = user_prompt)
    
    # Step 3: Validation + Auto-repair loop
    max_retries = 3
    retry = 0
    
    while retry < max_retries:
        try:
            restaurant_obj = Restaurant.model_validate_json(output)
            break
        except ValidationError as e:
            error_message = e.json()
            repair_system_msg, repair_prompt = JSON_auto_repair_prompts(response = output, error_message = error_message)
            output = llm_model(system_msg = repair_system_msg, prompt_txt = repair_prompt)
            retry += 1
    
    # Step 4: Final validation and structuring
    try:
        restaurant_obj = Restaurant.model_validate_json(output)
        structured_data = restaurant_obj.model_dump()
        structured_data['itemId'] = itemId
        return structured_data
    except ValidationError:
        return {"error": "unparseable_json", "raw_output": output, "itemId": itemId}


def manage_restaurants(file_path = FILEPATH, backup_path = BACKUP_PATH):
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")
        
        choice = input("\nAction: ")

        # Choice 1: Iterate through the records in the data file and show their names. If name doesn't exist, print 'N/A'.
        if choice == '1':
            print("\n--- Current Listings ---")
            for idx, record in enumerate(data):
                name = record.get('name', 'N/A')
                print(f"{idx + 1}. {name}")
              
        # Choice 2: Get the record index in demand from the user with input(). Check the validity of the input index. If the index is valid, use the  
			  #           helper function show_restaurant_card(res, index); Otherwise, print "invalid index." 
        elif choice == '2':
            try:
                index = int(input("Enter record number: ")) - 1
                if 0 <= index < len(data):
                    restaurant = data[index]
                    # Display detailed record
                    print(f"\n{'='*60}")
                    print(f"🏪 RESTAURANT DETAILS")
                    print(f"{'='*60}")
                    print(f"📛 Name: {restaurant.get('name', 'N/A')}")
                    print(f"📍 Location: {restaurant.get('location', 'N/A')}")
                    print(f"🏷️ Type: {restaurant.get('type', 'N/A')}")
                    print(f"🍽️ Food Style: {restaurant.get('food_style', 'N/A')}")
                    print(f"⭐ Rating: {restaurant.get('rating', 'N/A')}")
                    
                    price_range = restaurant.get('price_range')
                    if price_range:
                        print(f"💰 Price Range: {'$' * price_range} ({price_range}/4)")
                    else:
                        print(f"💰 Price Range: N/A")
                    
                    signatures = restaurant.get('signatures', [])
                    if signatures:
                        print(f"🍜 Signature Dishes: {', '.join(signatures)}")
                    else:
                        print(f"🍜 Signature Dishes: N/A")
                    
                    print(f"🎭 Vibe: {restaurant.get('vibe', 'N/A')}")
                    print(f"🌍 Environment: {restaurant.get('environment', 'N/A')}")
                    
                    shortcomings = restaurant.get('shortcomings', [])
                    if shortcomings:
                        print(f"⚠️ Shortcomings: {', '.join(shortcomings)}")
                    
                    if 'itemId' in restaurant:
                        print(f"🆔 Item ID: {restaurant['itemId']}")
                      
                    print(f"{'='*60}\n")
                  
                else:
                    print("❌ Invalid index.")
    
            except ValueError:
                print("❌ Please enter a valid number.")

        elif choice in ['3', '4', '5']:
            # Strict Security Warning
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            # Choice 3: INSERT DATA
            if choice == '3':
                itemId = 1000000 + len(data) + 1

                print("\n📝 Enter restaurant description (press Enter, then type 'END' on next line to finish):")
                
                lines = []
                while True:
                    line = input()
                    if line == 'END':
                        break
                    lines.append(line)
                
                paragraph = ' '.join(lines).strip()
                
                if paragraph and paragraph != 'END':
                    print("\n⏳ Processing restaurant with AI...")
                    new_restaurant = new_data_entry_process(paragraph, itemId)
                    if 'error' not in new_restaurant:
                        data.append(new_restaurant)
                        save_data(data, file_path)
                        print(f"✅ Restaurant added successfully! (ID: {itemId})")
                        print(f"   Name: {new_restaurant.get('name', 'N/A')}")
                    else:
                        print(f"❌ Failed to process restaurant: {new_restaurant.get('error', 'Unknown error')}")
                else:
                    print("❌ No description provided.")

            # Choice 4: EDIT DATA
            elif choice == '4':
                # First: ask for the input record index.
        				# Second: iterate over the keys of the current record, and ask for new values. If the user doesn't want to update, a simple 
        				#         Enter can skip. Update it only when the input index is valid.
        				# Third: save with save_data() and notify ("Record updated.")
                try:
                    index = int(input("Enter record number to edit: ")) - 1
                    if 0 <= index < len(data):
                        record = data[index]
                        print(f"\n✏️ Editing record #{index + 1}: {record.get('name', 'N/A')}")
                        print("(Press Enter to keep current value)\n")
                        
                        fields = ['name', 'location', 'type', 'food_style', 'rating', 'price_range', 'signatures', 'vibe', 'environment', 'shortcomings']
                        
                        for field in fields:
                            current_value = record.get(field, '')
                            if isinstance(current_value, list):
                                current_value = ', '.join(current_value)
                            
                            new_value = input(f"{field.capitalize()} [{current_value}]: ").strip()
                            
                            if new_value:
                                if field in ['rating', 'price_range']:
                                    try:
                                        if field == 'rating':
                                            record[field] = float(new_value)
                                        else:
                                            record[field] = int(new_value)
                                    except ValueError:
                                        print(f"⚠️ Invalid value for {field}, keeping original.")
                                      
                                elif field in ['signatures', 'shortcomings']:
                                    record[field] = [item.strip() for item in new_value.split(',') if item.strip()]
                                  
                                else:
                                    record[field] = new_value
                        
                        save_data(data, file_path)
                        print("✅ Record updated successfully!")
                      
                    else:
                        print("❌ Invalid index.")
                      
                except ValueError:
                    print("❌ Please enter a valid number.")

            # Choice 5: DELETE DATA
            elif choice == '5':
                # First: ask for the input record index.
				        # Second: use pop() to delete if the index is valid.
				        # Third: save_data() and notify.
                try:
                    index = int(input("Enter record number to delete: ")) - 1
                    if 0 <= index < len(data):
                        deleted_record = data.pop(index)
                        save_data(data, file_path)
                        print(f"✅ Deleted: {deleted_record.get('name', 'Unknown restaurant')}")
                      
                    else:
                        print("❌ Invalid index.")
                      
                except ValueError:
                    print("❌ Please enter a valid number.")

        elif choice == '6':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid input. Please choose 1-6.")

# RUN THE UI
if __name__ == "__main__":
    manage_restaurants()


# FOR TESTING THE UI
class TestRestaurantDatabase(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary clean database for testing."""
        self.test_file = 'structured_restaurant_data_unit_test.json'
        self.test_file_backup = 'structured_restaurant_data_unit_test.json.bak'
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, 'w') as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Clean up the test file after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input):
        """
        Test Scenario: Add a new restaurant.
        Inputs: '3' (Add), 'yes' (Confirm), 'New Burger Joint', '6' (Exit)
        """
        # We mock the sequence of user inputs
        mock_restaurant = 'The Copper Sprout is a high-concept, Modern Appalachian farm-to-table destination that blends an industrial-chic aesthetic with rustic forest charm, featuring reclaimed wood and amber lighting to create a sophisticated yet cozy vibe. Priced in the $$ category, the menu celebrates seasonal foraging and local heritage, headlined by signature dishes like Cast-Iron Smoked Trout with pickled fiddlehead ferns and hand-foraged Wild Mushroom Risotto with aged goat cheese. The experience is designed to be intimate and earthy, making it a premier spot for those seeking high-quality, smokehouse-influenced cuisine in a refined, atmospheric setting.'
        mock_input.side_effect = ['3', 'yes', mock_restaurant, '6']
        
        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        print(data)
        self.assertEqual(len(data), 2)
        self.assertIn("✅ Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ['5', 'yes', 1, '6']
        
        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        print(data)
        self.assertEqual(len(data), 1)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Test Scenario: Try to delete but say 'no' to security warning.
        Inputs: '5' (Delete), 'no' (Cancel), '6' (Exit)
        """
        mock_input.side_effect = ['5', 'no', '6']
        
        manage_restaurants(self.test_file, self.test_file_backup)
        
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 1) # Data should remain unchanged
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())
		
if __name__ == "__main__":
    unittest.main() # Unit Test
	# manage_restaurants(FILEPATH, BACKUP_PATH) # Actual UI Call
