import streamlit as st
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import os
import time

def init_genai(api_key):
    """Initializes the Google Generative AI client."""
    genai.configure(api_key=api_key)

    if not api_key:
        st.sidebar.warning("Please enter your Google API Key.")
        return False

    # Optionally verify connectivity
    try:
        genai.configure(api_key=api_key)

        # Set API Key for Langchain Google Generative AI
        os.environ["GOOGLE_API_KEY"] = api_key

        st.sidebar.success("Gemini API Key configured successfully.")
        return True

    except Exception as e:
        st.sidebar.warning(f"Connection issue: {e}")
        return False

def create_nutrition_prompt():
    template = """
    You are an expert nutritionist and professional dietitian.

    Analyze the following food items and provide detailed nutritional information.

    Food Items: {food_items}

    Provide the response in the following structured format:

    1. MACRONUTRIENTS
       - Protein (grams)
       - Carbohydrates (grams)
       - Fat (grams)
       - Dietary Fiber (grams)

    2. MICRONUTRIENTS
       - Important vitamins
       - Important minerals
       - Their health benefits

    3. CALORIE INFORMATION
       - Calories for each food item
       - Total estimated calories
       - Explain the calorie contribution of each item.

    4. HEALTH BENEFITS
       - Benefits of each food item
       - Benefits for the human body
       - Important dietary considerations.

    5. NUTRITIONAL SUMMARY
       - Overall nutritional quality
       - Whether the meal is balanced
       - Suggestions for improving the meal.

    IMPORTANT INSTRUCTIONS:

    - Provide detailed and complete information.
    - Do not unnecessarily shorten the response.
    - Use tables wherever appropriate.
    - Explain nutritional values clearly.
    - Mention that nutritional values are estimates.
    - State assumptions about serving sizes.
    - Never invent exact nutritional values.
    - If the input is not a food item, explain that you
      only provide nutritional information about food.

    Provide a complete response with all sections.
    """

    return PromptTemplate(
        input_variables=["food_items"],
        template=template
    )

def get_nutritional_info(food_items, api_key):
    """Generates the nutritional information response using the AI model."""

    if not api_key:
        st.error("Please enter a valid Google API Key.")
        return None
    
    if not food_items or food_items.strip() == "":
        st.error("Please provide at least one food item.")
        return None
        
    try:
        # Initialize Langchain Chat model
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=api_key,
            temperature=0.4
        )
        
        # Create prompt
        prompt = create_nutrition_prompt()

        # Create Langchain chain 
        chain = prompt | llm | StrOutputParser()
        response = chain.stream({
            "food_items": food_items
        })
            
        return response

    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

def main():
    # 1. Create main Streamlit application title
    st.set_page_config(page_title="NutriGen - Instant Nutritional Info", page_icon="🥗", layout="centered")
    st.title("NutriAI - Instant Nutritional Information")
    st.write("Get detailed macronutrient, micronutrient, and calorie data for your meals using Google Gemini AI.")
    
    # Sidebar for API Key
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Enter your Google Gemini API Key:", type="password")
    
    if api_key:
        init_genai(api_key)
    else:
        st.sidebar.info("Get your API key at https://aistudio.google.com/app/apikey")
        
    # 2. Collect food items input from user
    st.header("What are you eating?")
    
    with st.form("nutrition_form"):
        food_input = st.text_area(
            "Enter food items (separated by commas):",
            placeholder="e.g., 2 scrambled eggs, 1 slice of whole wheat toast, 1 apple",
            height=100
        )
        
        submit_button = st.form_submit_button("Analyze Nutritional Value")
        
    # 3. Generate and Display Response
    if submit_button:
        if not api_key:
             st.warning("⚠️ Please provide your Google API Key in the sidebar to continue.")

        elif not food_input:
             st.warning("⚠️ Please enter some food items.")

        else:
            st.markdown("### 🥗 Nutritional Analysis")

        try:
            result = get_nutritional_info(food_input, api_key)

            if result:
                with st.spinner("Analyzing nutritional content... Please Wait."):

                    st.write_stream(result)

        except Exception as e:

            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                st.error("Gemini API quota exceeded. Please try again when your quota is available.")

            else:
                st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
