import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

# load .env file
load_dotenv()

def analyze_text_with_agent(text_to_analyze: str):
    """
    Analyzes a given text using a Groq model to determine sentiment, theme, and a summary.

    Args:
        text_to_analyze (str): The cleaned text of the tweet.

    Returns:
        dict: A dictionary containing the analysis (sentiment, theme, summary).
    """
    model = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0)


    class Analysis(BaseModel):
        sentiment: str = Field(description="The sentiment of the text. Must be 'Positive', 'Negative', or 'Neutral'.")
        theme: str = Field(description="The main theme or topic of the text in one or two words.")
        summary: str = Field(description="A brief, one-sentence summary of the text.")

    # parser for structure the model's output
    parser = JsonOutputParser(pydantic_object=Analysis)

    # prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert sentiment analyst. Analyze the user's text and provide the output in a structured JSON format. Use the provided format instructions."),
        ("human", "Analyze the following text: ```{text}```\n{format_instructions}")
    ]).partial(format_instructions=parser.get_format_instructions())

    # Create the chain that links the prompt, model, and parser.
    chain = prompt | model | parser

    print("Agent is analyzing the text...")
    try:
        # Invoke the chain with the input text.
        analysis_result = chain.invoke({"text": text_to_analyze})
        return analysis_result
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return None

if __name__ == '__main__':
    # A simple test for our agent
    sample_text = "cant pessimist counter instinct optimistic optimism important"
    
    result = analyze_text_with_agent(sample_text)
    
    if result:
        print("\n--- Agent Analysis Result ---")
        import json
        print(json.dumps(result, indent=2))