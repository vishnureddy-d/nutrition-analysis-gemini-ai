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
    # Using the standard environment variable for Langchain integration
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Optionally list models to verify connectivity
    try:
        models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        st.sidebar.success(f"Connected! Available models: {len(models)}")
    except Exception as e:
        st.sidebar.warning(f"Connection issue: {e}")

def create_nutrition_prompt():
    """Defines the prompt template for nutritional information."""
    template = """
    You are an expert nutritionist and dietitian. Provide detailed nutritional information for the following food items: {food_items}.
    
    Please provide the information in a structured format:
    1. Macronutrients: Protein, Fat, Carbohydrates (in grams) for each item.
    2. Micronutrients: Key Vitamins and Minerals for each item.
    3. Calorie Content: Total calories for each item and roughly estimated total calories.
    4. General summary of the health benefits or dietary considerations.
    
    If the requested input is not a food item or doesn't make sense, kindly explain that you can only provide nutritional data for foods.
    dont stop the response in the midel i need complete and details about it 
    """
    return PromptTemplate(input_variables=["food_items"], template=template)

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
            model="gemini-2.5-flash",
            temperature=0.4,
            max_output_tokens=1024
        )
        
        prompt = create_nutrition_prompt()
        chain = prompt | llm | StrOutputParser()
        
        # Adding a spinner while generating
        with st.spinner('Analyzing nutritional content...'):
            response = chain.invoke({"food_items": food_items})
            
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
            result = get_nutritional_info(food_input, api_key)
            if result:
                st.markdown("### Nutritional Analysis")
                st.markdown(result)

if __name__ == "__main__":
    main()
