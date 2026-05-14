from shared_functions import *

# Main function for advanced search demonstrations"""
def main():
    try:
        print("🔬 Advanced Food Search System")
        print("=" * 50)
        print("Loading food database with advanced filtering capabilities...")
        
        # Load food data from JSON file
        food_items = load_food_data('./FoodDataSet.json')
        print(f"\n ✅ Loaded {len(food_items)} food items successfully")
        
        # Create collection specifically for advanced search operations
        collection = create_similarity_search_collection("advanced_food_search", {'description' : 'A collection for advanced search demos'})
        populate_similarity_collection(collection, food_items)
        interactive_advanced_search(collection)
        
    except Exception as error:
        print(f"\n ❌ Error initializing advanced search system: {error}")

# Function for interactive advanced search with filtering options
def interactive_advanced_search(collection):
    print("\n" + "="*50)
    print("🔧 ADVANCED SEARCH WITH FILTERS")
    print("="*50)
    print()
    print("Search Options:")
    print("\n1. Basic similarity search")
    print("\n2. Cuisine-filtered search")  
    print("\n3. Calorie-filtered search")
    print("\n4. Combined filters search")
    print("\n5. Demonstration mode")
    print("\n6. Help")
    print("\n7. Exit")
    print("-" * 50)
    
    while True:
        try:
            choice = input("\n📋 Select option (1-7): ").strip()
            
            if choice == '1':
                perform_basic_search(collection)
            elif choice == '2':
                perform_cuisine_filtered_search(collection)
            elif choice == '3':
                perform_calorie_filtered_search(collection)
            elif choice == '4':
                perform_combined_filtered_search(collection)
            elif choice == '5':
                run_search_demonstrations(collection)
            elif choice == '6':
                show_advanced_help()
            elif choice == '7':
                print("\n 👋 Exiting Advanced Search System. Goodbye!")
                break
            else:
                print("\n ❌ Invalid option. Please select 1-7.")
                
        except KeyboardInterrupt:
            print("\n👋 System interrupted. Goodbye!")
            break
  
        except Exception as e:
            print(f"❌ Error: {e}")

# Function to perform basic similarity search without filters
def perform_basic_search(collection):
    print("\n🔍 BASIC SIMILARITY SEARCH")
    print("-" * 30)
    
    query = input("Enter search query: ").strip()
    if not query:
        print("\n ❌ Please enter a search term")
        return
    
    print(f"\n🔍 Searching for '{query}'...")
    results = perform_similarity_search(collection, query, 5)
    display_search_results(results, "Basic Search Results")

# Function to perform cuisine-filtered similarity search
def perform_cuisine_filtered_search(collection):
    print("\n🍽️ CUISINE-FILTERED SEARCH")
    print("-" * 30)
    
    # Show available cuisines from our dataset
    cuisines = ["Italian", "Thai", "Mexican", "Indian", "Japanese", "French", "Mediterranean", "American", "Health Food", "Dessert"]
    print("\nAvailable cuisines:")
    for i, cuisine in enumerate(cuisines, 1):
        print(f"{i}. {cuisine}")
    
    query = input("\nEnter search query: ").strip()
    cuisine_choice = input("\nEnter cuisine number (or cuisine name): ").strip()
    
    if not query:
        print("❌ Please enter a search term")
        return
    cuisine_filter = None
    if cuisine_choice.isdigit():
        idx = int(cuisine_choice) - 1
        if 0 <= idx < len(cuisines):
            cuisine_filter = cuisines[idx]
    else:
        cuisine_filter = cuisine_choice
    
    if not cuisine_filter:
        print("❌ Invalid cuisine selection")
        return
    
    print(f"\n🔍 Searching for '{query}' in {cuisine_filter} cuisine...")
    results = perform_filtered_similarity_search(collection, query, cuisine_filter=cuisine_filter, n_results = 5)
    display_search_results(results, f"Cuisine-Filtered Results ({cuisine_filter})")

# Function to perform calorie-filtered similarity search"""
def perform_calorie_filtered_search(collection):
    print("\n🔥 CALORIE-FILTERED SEARCH")
    print("-" * 30)
    
    query = input("\nEnter search query: ").strip()
    max_calories_input = input("\nEnter maximum calories (or press Enter for no limit): ").strip()
    
    if not query:
        print("❌ Please enter a search term")
        return
    max_calories = None
    if max_calories_input.isdigit():
        max_calories = int(max_calories_input)
    
    print(f"\n🔍 Searching for '{query}'" + (f" with max {max_calories} calories..." if max_calories else "..."))
    results = perform_filtered_similarity_search(collection, query, max_calories = max_calories, n_results = 5)
    calorie_text = f"under {max_calories} calories" if max_calories else "any calories"
    display_search_results(results, f"Calorie-Filtered Results ({calorie_text})")

# Function to perform search with multiple filters combined
def perform_combined_filtered_search(collection):
    print("\n🎯 COMBINED FILTERS SEARCH")
    print("-" * 30)
    
    query = input("\nEnter search query: ").strip()
    cuisine = input("\nEnter cuisine type (optional): ").strip()
    max_calories_input = input("\nEnter maximum calories (optional): ").strip()
    
    if not query:
        print("❌ Please enter a search term")
        return
    cuisine_filter = cuisine if cuisine else None
    max_calories = int(max_calories_input) if max_calories_input.isdigit() else None
    
    # Build description of applied filters
    filter_description = []
    if cuisine_filter:
        filter_description.append(f"cuisine: {cuisine_filter}")
    if max_calories:
        filter_description.append(f"max calories: {max_calories}")
    
    filter_text = ", ".join(filter_description) if filter_description else "no filters"
    
    print(f"\n🔍 Searching for '{query}' with {filter_text}...")
    results = perform_filtered_similarity_search(collection, query, cuisine_filter = cuisine_filter, max_calories = max_calories, n_results = 5)
    display_search_results(results, f"Combined Filtered Results ({filter_text})")

# Function to run predetermined demonstrations of different search types
def run_search_demonstrations(collection):
    print("\n📊 SEARCH DEMONSTRATIONS")
    print("=" * 40)
    
    demonstrations = [
        {
            "title": "Italian Cuisine Search",
            "query": "creamy pasta",
            "cuisine_filter": "Italian",
            "max_calories": None
        },
        {
            "title": "Low-Calorie Healthy Options",
            "query": "healthy meal",
            "cuisine_filter": None,
            "max_calories": 300
        },
        {
            "title": "Asian Light Dishes",
            "query": "light fresh meal",
            "cuisine_filter": "Japanese",
            "max_calories": 250
        }
    ]
    
    for i, demo in enumerate(demonstrations, 1):
        print(f"\n{i}. {demo['title']}")
        print(f"\nQuery: '{demo['query']}'")
        
        filters = []
        if demo['cuisine_filter']:
            filters.append(f"\nCuisine: {demo['cuisine_filter']}")
        if demo['max_calories']:
            filters.append(f"\nMax Calories: {demo['max_calories']}")
        
        if filters:
            print(f"\nFilters: {', '.join(filters)}")
        
        results = perform_filtered_similarity_search(collection, demo['query'], cuisine_filter = demo['cuisine_filter'], max_calories = demo['max_calories'], n_results = 3)       
        display_search_results(results, demo['title'], show_details = False)
        input("\n⏸️ Press Enter to continue to next demonstration...")

# Function to display search results in a formatted way
def display_search_results(results, title, show_details = True):
    print(f"\n📋 {title}")
    print("=" * 50)
    
    if not results:
        print("❌ No matching results found")
        print("💡 Try adjusting your search terms or filters")
        return
    
    for i, result in enumerate(results, 1):
        score_percentage = result['similarity_score'] * 100
        
        if show_details:
            print(f"\n{i}. 🍽️  {result['food_name']}")
            print(f"\n📊 Similarity Score: {score_percentage:.1f}%")
            print(f"\n🏷️ Cuisine: {result['cuisine_type']}")
            print(f"\n🔥 Calories: {result['food_calories_per_serving']}")
            print(f"\n📝 Description: {result['food_description']}")
        else:
            print(f"\n{i}. {result['food_name']} ({score_percentage:.1f}% match)")
    print("=" * 50)

# Function to display help information for advanced search
def show_advanced_help():
    print("\n📖 ADVANCED SEARCH HELP")
    print("=" * 40)
    print("\nSearch Types:")
    print("\n1. Basic Search - Standard similarity search")
    print("\n2. Cuisine Filter - Search within specific cuisine types")
    print("\n3. Calorie Filter - Search for foods under calorie limits")
    print("\n4. Combined Filters - Use multiple filters together")
    print("\n5. Demonstrations - See predefined search examples")
    print()
    print("\nTips:")
    print("\n• Use descriptive terms: 'creamy', 'spicy', 'light'")
    print("\n• Combine ingredients: 'chicken vegetables'"
    print("\n• Try cuisine names: 'Italian', 'Thai', 'Mexican'")
    print("\n• Filter by calories for dietary goals")

if __name__ == "__main__":
    main()
