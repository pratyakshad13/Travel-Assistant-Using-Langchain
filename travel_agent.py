import os
import torch
import json
import datetime
from typing import Optional
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_core.tools import tool
from langchain_community.llms import HuggingFaceEndpoint , Ollama
from langchain.tools.render import render_text_description
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
from temporary_data import FLIGHT_DATA , HOTEL_DATA
from dotenv import load_dotenv
from flight_booking_backend import FlightBookingBackend
from hotel_booking_backend import HotelBookingBackend
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

mydb = None
def database_flight_init(host , user , password , database):
    global mydb
    mydb = FlightBookingBackend(host , user ,database ,password)
mydb2 = None
def database_hotel_init(host , user , password , database):
    global mydb2
    mydb2 = HotelBookingBackend(host , user ,database ,password)
    

def get_image_caption(img):
    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    image = img
    enc_image = model.encode_image(image)
    return model.answer_question(enc_image, "Describe this image.", tokenizer)

# Define Prompts
@tool
def hotel_search(destination :str , date : str , text : Optional[str] = None):
    """ Search for available hotels based on user query.
        You are equiped with hotels data

    Args:
        destination (str) : In which city you would like to stay
        date (str) : since when you would like to stay
        text (str , optional ): user preference
    Returns:
        str : return response to user query
    """
    prompt = f"""Given the following Hotels search request and available hotels,
    provide relevant hotel options:

    User preference: {text}
   
    Available HOTELS:
    {mydb2.get_hotel_info( destination , date)}

    Please list only the relevant Hotel with their details."""

    llm = Ollama(model="llama3-groq-tool-use", temperature=0.7)
    response = llm.invoke(prompt)
    return str(response)

@tool
def book_hotel(hotel_name: str, passenger_id: str, date : str , number_of_people : int , destination :str, number_of_room :int):
    """You are Hotel booking agent who is expert in hotel ticket booking task
    Args:
        hotel_id (int): _description_
        passenger_id (str): _description_
        date (str): _description_
        number_of_people (int): _description_
        room_type (str): _description_
        stay_time (int): _description_
    """
    response = mydb2.book_hotels( passenger_id, destination, hotel_name, date, number_of_room)
    return f"room booked in {hotel_name} on {date} for {number_of_people} + \n\n\n{response}"
# departure, passenger_id, destination, airport, airline, date, seats
@tool 
def book_flight( departure : str , destination : str , flight_name : str , passenger_id : str ,airport : str,seat : str ):
    """ You are flight booking agent who is expert in flight ticket booking task

    Args:
        departure (str): Departure Airport 
        destination (str): Destination ( where you want to go)
        flight_name (str): perticular flight name you want to travel in
        passenger_id (str): unique passenger_id to uniquely identify passenger
        airport (str) : From which airport you would like to board plane
        seat (str): which seat you would like to book

    Returns:
        str : if all required parameter are provide then return a response regarding ticket information else ask for remaining parameters to book ticket
    """
    # functionality
    
    response = mydb.book_flight(departure , destination ,flight_name , passenger_id , airport, seat)
    return f"ticket booked for {flight_name} form {destination} to {departure} + {response}"
@tool
def search_flights(departure : str , destinatio : str , date: Optional[str] = None , text: Optional[str] = None ) -> str:
    """
    Search for available flights based on user query.
    You are equiped with flights data
    
    Args:
        departure (str) : departure airport / city
        destination (str) : destination airport / city
        date (str , optional) : date of flight
        text (str , optional): User's preference
    Returns:
        str: Available flights matching the search criteria
    """
    prompt = f"""Given the following flight search request and available flights,
    provide relevant flight options:

    User preference: {text}
   
    Available Flights:
    {mydb.get_flight_info(departure , destinatio , (date if date is not None else datetime.now()))}

    Please list only the relevant flights with their details."""

    llm = Ollama(model="llama3-groq-tool-use", temperature=0.7)
    response = llm.invoke(prompt)
    return str(response)



@tool
def image_caption_tool(img_path: str) -> str:
    """
    Generate caption for an uploaded image.
    
    Args:
        img_path (str): Path to the image file
        
    Returns:
        str: Generated caption for the image
    """
    try:
        if img_path is None:
            return "Error: No image file provided"

        # Process image
        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu" if not torch.cuda.is_available() else "cuda"
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        # Handle image file
        image = Image.open(img_path).convert('RGB')
        inputs = processor(image, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=20)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error analyzing image: {str(e)}"  

@tool
def adviser_tool(text : str , chat_history=None):
    """You are a Travel Expert that helps user in their query

    Args:
        text (_type_): User query 
        chat_history (_type_, optional): previous record of chats with user
    
    Returns:
        str : the response to resolve user query
    """
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceEndpoint(
                huggingfacehub_api_token="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",
                repo_id=repo_id,
                task="text-generation",
                temperature=0.7,
                top_p=0.9,
                max_length=512,
                do_sample=True,
                max_new_tokens=512,
                return_full_text=False
            )
    history_text = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" 
                                 for h in (chat_history or [])])
    prompt_template = """
        Act as an experienced travel advisor. Based on this query: "{query}"
        And considering this conversation history: {history}
        
        Provide detailed travel information about:
        1. Popular attractions and landmarks
        2. Best time to visit
        3. Local culture and customs
        4. Food and dining recommendations
        5. Transportation options
        6. Accommodation suggestions
        7. Practical travel tips
        
        Keep the response informative and well-structured.
        """
    response = llm(prompt_template)
    if isinstance(response, dict) and 'generated_text' in response:
        return response['generated_text']
    elif isinstance(response, str):
        return response
    else:
        return "I apologize, but I couldn't generate a proper response. Please try asking your question again."
                
def initialize_chain():
    database_flight_init("localhost" , "root" , "flight_management", "1213" )
    database_flight_init("localhost" , "root" , "flight_management", "1213" )
    tools = [adviser_tool, search_flights, book_flight, hotel_search, book_hotel]
    rendered_tools = render_text_description(tools)
    
    system_prompt = f"""You are an assistant that has access to the following set of tools.
    Here are the names and descriptions for each tool:

    {rendered_tools}
    Given the user input, return the name and input of the tool to use.
    Return your response as a JSON blob with 'name' and 'arguments' keys.
    The value associated with the 'arguments' key should be a dictionary of parameters.
    Only return the JSON object, nothing else."""

    llm = Ollama(model="llama3-groq-tool-use")
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{input}")]
    )
    
    def tool_chain(model_output):
        try:
            if isinstance(model_output, str):
                model_output = json.loads(model_output)
            
            tool_map = {tool.name: tool for tool in tools}
            chosen_tool = tool_map[model_output["name"]] 
            tool_response = chosen_tool.invoke(model_output["arguments"])
            return tool_response
        except json.JSONDecodeError:
            return "I had trouble understanding that. Could you please rephrase your request?"
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    chain = prompt | llm | tool_chain
    return chain