from shared_functions import *

# Global variable to store loaded food items
food_items = []

# Main function for interactive CLI food recommendation system
def main():
    try:
        print("🍽️ Interactive Food Recommendation System")
        print("=" * 50)
        print("Loading food database...")
        
        # Load food data from file
        global food_items
        food_items = load_food_data('./FoodDataSet.json')
        print(f"\n ✅ Loaded {len(food_items)} food items successfully")
        
        # Create and populate search collection
        collection = create_similarity_search_collection("interactive_food_search", {'description': 'A collection for interactive food search'})
        populate_similarity_collection(collection, food_items)
        interactive_food_chatbot(collection)
        
    except Exception as error:
        print(f"❌ Error initializing system: {error}")

# Function to create the interactive CLI chatbot for food recommendations
def interactive_food_chatbot(collection):
    print("\n" + "="*50)
    print("🤖 INTERACTIVE FOOD SEARCH CHATBOT")
    print("="*50)
    print("\nCommands:")
    print("\n• Type any food name or description to search")
    print("\n• 'help' - Show available commands")
    print("\n• 'quit' or 'exit' - Exit the system")
    print("\n• Ctrl+C - Emergency exit")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n🔍 Search for food: ").strip()
            
            if not user_input:
                print("Please enter a search term or 'help' for commands")
                continue
      
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Food Recommendation System!")
                print("\nGoodbye!")
                break
            elif user_input.lower() in ['help', 'h']:
                show_help_menu()
            else:
                handle_food_search(collection, user_input)
                
        except KeyboardInterrupt:
            print("\n👋 System interrupted. Goodbye!")
            break
          
        except Exception as e:
            print(f"❌ Error processing request: {e}")

# Function to display help information for users
def show_help_menu():
    print("\n📖 HELP MENU")
    print("-" * 30)
    print()
    print("\nSearch Examples:")
    print("\n• 'chocolate dessert' - Find chocolate desserts")
    print("\n• 'Italian food' - Find Italian cuisine")
    print("\n• 'sweet treats' - Find sweet desserts")
    print("\n• 'baked goods' - Find baked items")
    print("\n• 'low calorie' - Find lower-calorie options")
    print()
    print("\nCommands:")
    print("\n• 'help' - Show this help menu")
    print("\n• 'quit' - Exit the system")

# Function to handle food similarity search with enhanced display
def handle_food_search(collection, query):
    print(f"\n🔍 Searching for '{query}'...")
    print("\nPlease wait...")
    
    # Perform similarity search
    results = perform_similarity_search(collection, query, 5)
    
    if not results:
        print("\n ❌ No matching foods found.")
        print("\n 💡 Try different keywords like:")
        print("\n • Cuisine types: 'Italian', 'American'")
        print("\n • Ingredients: 'chocolate', 'flour', 'cheese'")
        print("\n • Descriptors: 'sweet', 'baked', 'dessert'")
        return
    
    # Display results with rich formatting
    print(f"\n✅ Found {len(results)} recommendations:")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        # Calculate percentage score
        percentage_score = result['similarity_score'] * 100
        print(f"\n {i}. 🍽️ {result['food_name']}")
        print(f"\n 📊 Match Score: {percentage_score:.1f}%")
        print(f"\n 🏷️ Cuisine: {result['cuisine_type']}")
        print(f"\n 🔥 Calories: {result['food_calories_per_serving']} per serving")
        print(f"\n 📝 Description: {result['food_description']}")
        
        if i < len(results):
            print("   " + "-" * 50)  
    print("=" * 60)
    
    # Provide suggestions for further exploration
    suggest_related_searches(results)

# Function to suggest related searches based on current results
def suggest_related_searches(results):
    if not results:
        return

    cuisines = list(set([r['cuisine_type'] for r in results]))
    
    print("\n💡 Related searches you might like:")
    for cuisine in cuisines[:3]:
        print(f"\n • Try '{cuisine} dishes' for more {cuisine} options")
    
    # Suggest calorie-based searches
    avg_calories = sum([r['food_calories_per_serving'] for r in results]) / len(results)
    if avg_calories > 350:
        print("\n • Try 'low calorie' for lighter options")
    else:
        print("\n • Try 'hearty meal' for more substantial dishes")

if __name__ == "__main__":
    main()
