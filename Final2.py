####################################################

"""
    IGNORE THIS CODE
    THIS CODE FILE IS JUST MIXTURE OF ALL THE OTHER FILES IN ONE FILE
"""

####################################################


















import ollama
import streamlit as st
from PIL import Image
from operator import itemgetter
from transformers import pipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_core.tools import tool
from langchain_community.llms import HuggingFaceEndpoint , Ollama
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_message_histories import StreamlitChatMessageHistory 
from langchain.tools.render import render_text_description
from langchain_ollama import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser , JsonOutputParser
from langchain.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from amadeus import Client, ResponseError
from typing import List, Dict , Optional
import pandas as pd
from datetime import datetime, timedelta
import json
import torch
import io
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image


if "messages" not in st.session_state:
    st.session_state.messages = []


FLIGHT_DATA = """
Here’s the list of todays flights departing from Lucknow Airport (LKO):

Flight 6E 2108 - 06:00 AM - to Delhi (IndiGo)
Flight BA 7980 - 06:00 AM - to Delhi (British Airways, operated by IndiGo)
Flight VS 8675 - 06:00 AM - to Delhi (Virgin Atlantic, operated by IndiGo)
Flight 6E 142 - 06:40 AM - to Ahmedabad (IndiGo)
Flight 6E 6164 - 07:15 AM - to Amritsar (IndiGo)
Flight 6E 7462 - 07:25 AM - to Nagpur (IndiGo)
Flight 6E 5050 - 07:45 AM - to Kolkata (IndiGo)
Flight S9 327* - 07:50 AM - to Moradabad (Flybig)
Flight 6E 2026 - 07:50 AM - to Delhi (IndiGo)
Flight VS 8661 - 07:50 AM - to Delhi (Virgin Atlantic, operated by IndiGo)
Flight IX 185 - 07:55 AM - to Ras al-Khaimah (Air India Express)
Flight AI 9159 - 07:55 AM - to Ras al-Khaimah (Air India, operated by Air India Express)
Flight 6E 2238 - 08:00 AM - to Mumbai (IndiGo)
Flight VS 8956 - 08:00 AM - to Mumbai (Virgin Atlantic, operated by IndiGo)
Flight 6E 7439 - 08:15 AM - to Indore (IndiGo)
Flight IX 1552 - 08:25 AM - to Delhi (Air India Express)
Flight AI 9758 - 08:25 AM - to Delhi (Air India, operated by Air India Express)
Flight 6E 3250 - 08:50 AM - to Bengaluru (IndiGo, operated by SmartLynx Latvia)
Flight VS 8951 - 08:50 AM - to Bengaluru (Virgin Atlantic, operated by SmartLynx Latvia)
Flight IX 2816 - 09:00 AM - to Hyderabad (Air India Express)
Flight AI 9767 - 09:00 AM - to Hyderabad (Air India, operated by Air India Express)
Flight WY 266 - 09:05 AM - to Muscat (Oman Air)
Flight 6E 146 - 09:10 AM - to Guwahati (IndiGo)
Flight 6E 523 - 09:15 AM - to Hyderabad (IndiGo)
Flight 6E 6521 - 09:20 AM - to Raipur (IndiGo)
"""
HOTEL_DATA = """
Here is a list of popular hotels in Lucknow along with their approximate ratings:

Taj Mahal Lucknow - 9.0/10
Fairfield by Marriott Lucknow - 7.6/10
Hilton Garden Inn Lucknow - 7.5/10
Hyatt Regency Lucknow - 8.5/10
Renaissance Lucknow Hotel - 8.4/10
Lebua Lucknow - 8.6/10
Clarks Avadh - 7.8/10
The Piccadily Lucknow - 8.3/10
Hotel India Awadh - 7.2/10
Golden Tulip Lucknow - 8.0/10
Hotel Charans Plaza - 7.0/10
The Centrum - 8.1/10
Hotel Savvy Grand - 8.2/10
Hotel Levana Suites - 8.3/10
Hotel Lineage - 7.8/10
FabHotel V Hazratganj - 7.3/10
Hometel Alambagh Lucknow - 7.7/10
Hotel La Place Sarovar Portico - 8.4/10
Ginger Lucknow - 6.5/10
Fortune Park BBD - 8.3/10Nothing
"""

@tool
def hotel_search(text : str):
    """ Search for available hotels based on user query.
        You are equiped with hotels data

    Args:
        text (str): user query
    Returns:
        str : return response to user query
    """
    prompt = f"""Given the following Hotels search request and available hotels,
    provide relevant hotel options:

    User Query: {text}
   
    Available HOTELS:
    {HOTEL_DATA}

    Please list only the relevant Hotel with their details."""

    llm = Ollama(model="llama3-groq-tool-use", temperature=0.7)
    response = llm.invoke(prompt)
    return str(response)

@tool
def book_hotel(hotel_id:int, passenger_id: int, date : str , number_of_people : int , room_type :str , stay_time : int):
    """You are Hotel booking agent who is expert in hotel ticket booking task
    Args:
        hotel_id (int): _description_
        passenger_id (int): _description_
        date (str): _description_
        number_of_people (int): _description_
        room_type (str): _description_
        stay_time (int): _description_
    """
    return f"room booked in {hotel_id} on {date} for {number_of_people}"
@tool 
def book_flight( departure : str , destination : str , flight_id : int , passenger_id : int , seat : str ):
    """ You are flight booking agent who is expert in flight ticket booking task

    Args:
        departure (str): Departure Airport 
        destination (str): Destination ( where you want to go)
        flight_id (int): perticular flight_id you want to travel in
        passenger_id (int): unique passenger_id to uniquely identify passenger
        seat (str): which seat you would like to book

    Returns:
        str : if all required parameter are provide then return a response regarding ticket information else ask for remaining parameters to book ticket
    """
    # functionality
    return f"ticket booked for {flight_id} form {destination} to {departure}"
@tool
def search_flights(text: str) -> str:
    """
    Search for available flights based on user query.
    You are equiped with flights data
    
    Args:
        text (str): User's input
        
    Returns:
        str: Available flights matching the search criteria
    """
    prompt = f"""Given the following flight search request and available flights,
    provide relevant flight options:

    User Query: {text}
   
    Available Flights:
    {FLIGHT_DATA}

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
                



# tools = [adviser_tool , search_flights , book_flight]
# rendered_tools = render_text_description(tools)
# print(rendered_tools)

# system_prompt = f"""You are an assistant that has access to the following set of tools.
# Here are the names and descriptions for each tool:

# {rendered_tools}
# Given the user input, return the name and input of the tool to use.
# Return your response as a JSON blob with 'name' and 'arguments' keys.
# The value associated with the 'arguments' key should be a dictionary of parameters."""

# llm = Ollama(model="llama3-groq-tool-use")

# prompt = ChatPromptTemplate.from_messages(
#     [("system", system_prompt), ("user", "{input}")]
# )

# chain = prompt | llm | JsonOutputParser()

# def tool_chain(model_output):
#     tool_map = {tool.name: tool for tool in tools}
#     chosen_tool = tool_map[model_output["name"]]
#     return itemgetter("arguments") | chosen_tool
# chain = prompt | llm | JsonOutputParser() | tool_chain
# print(chain.invoke({'i want to know about delhi'}))







# msgs = StreamlitChatMessageHistory(key="langchain_messages")
# if len(msgs.messages) == 0:
#     msgs.add_ai_message("I am a Travel Assistant")

# st.title("Chatbot with tools")
# for msg in msgs.messages:
#     st.chat_message(msg.type).write(msg.content)

# if input := st.chat_input("What is up?"):
# # Display user input and save to message history.
#     st.chat_message("user").write(input)
#     msgs.add_user_message(input) 
#     # Invoke chain to get reponse.
#     response = chain.invoke({input})                 
#     # Display AI assistant response and save to message history.
#     st.chat_message("assistant").write(str(response))
#     msgs.add_ai_message(response)


# if "messages" not in st.session_state:
#     st.session_state.messages = []

# def initialize_chain():
#     tools = [adviser_tool, search_flights, book_flight , hotel_search , book_hotel , image_caption_tool]
#     rendered_tools = render_text_description(tools)
    
#     system_prompt = f"""You are an assistant that has access to the following set of tools.
#     Here are the names and descriptions for each tool:

#     {rendered_tools}
#     Given the user input, return the name and input of the tool to use.
#     Return your response as a JSON blob with 'name' and 'arguments' keys.
#     The value associated with the 'arguments' key should be a dictionary of parameters.
#     Only return the JSON object, nothing else."""

#     llm = Ollama(model="llama3-groq-tool-use")
#     prompt = ChatPromptTemplate.from_messages(
#         [("system", system_prompt), ("user", "{input}")]
#     )
    
#     def tool_chain(model_output):
#         try:
#             # If the output is a string, parse it to JSON
#             if isinstance(model_output, str):
#                 model_output = json.loads(model_output)
            
#             tool_map = {tool.name: tool for tool in tools}
#             chosen_tool = tool_map[model_output["name"]]
#             tool_response = chosen_tool.invoke(model_output["arguments"])
#             return tool_response
#         except json.JSONDecodeError:
#             return "I had trouble understanding that. Could you please rephrase your request?"
#         except Exception as e:
#             return f"An error occurred: {str(e)}"
    
#     chain = prompt | llm | tool_chain
#     return chain

# def main():
#     st.title("✈️ Flight Search & Travel Assistant")
    
#     # Initialize the chat history
#     msgs = StreamlitChatMessageHistory()
    
#     # Sidebar for user information
#     with st.sidebar:
#         st.header("User Information")
#         passenger_id = st.text_input("Passenger ID", value="", key="passenger_id")
#         st.write("---")
#         st.header("Quick Links")
#         if st.button("View All Flights"):
#             st.code(FLIGHT_DATA, language="text")
    
#     # Main chat interface
#     st.write("### Chat with your Travel Assistant")
    
#     # Display chat messages
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("How can I help you today?"):
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Generate and display assistant response
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     chain = initialize_chain()
#                     response = chain.invoke({'input': prompt})
                    
#                     # Display the response
#                     st.markdown(response)
                    
#                     # Add assistant response to chat history
#                     st.session_state.messages.append({"role": "assistant", "content": response})
                    
#                     # If it's a flight booking response, show additional options
#                     if "ticket booked" in str(response).lower() or "room booked" in str(response).lower():
#                         st.success("✅ Flight booked successfully!")
#                         st.download_button(
#                             label="Download Ticket",
#                             data=str(response),
#                             file_name="ticket.txt",
#                             mime="text/plain"
#                         )
#                 except Exception as e:
#                     st.error(f"An error occurred: {str(e)}")
#                     st.info("Please try rephrasing your request or contact support if the issue persists.")

#     # Additional features section
#     with st.expander("Additional Features"):
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Upload Travel Documents")
#             uploaded_file = st.file_uploader("Upload your documents", type=['png', 'jpg', 'pdf'])
#             if uploaded_file:
#                 st.success("Document uploaded successfully!")
        
#         with col2:
#             st.subheader("Quick Actions")
#             if st.button("Search Popular Destinations"):
#                 with st.spinner("Fetching destinations..."):
#                     response = adviser_tool("suggest popular tourist destinations")
#                     st.markdown(response)

# if __name__ == "__main__":
#     main()
    

    
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

def initialize_chain():
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
            
            # If the tool is image_caption_tool and we have an uploaded image
            if chosen_tool.name == "image_caption_tool" and st.session_state.uploaded_image is not None:
                model_output["arguments"]["image"] = st.session_state.uploaded_image
            
            tool_response = chosen_tool.invoke(model_output["arguments"])
            return tool_response
        except json.JSONDecodeError:
            return "I had trouble understanding that. Could you please rephrase your request?"
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    chain = prompt | llm | tool_chain
    return chain

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
    




def main():
    st.title("✈️ Flight Search & Travel Assistant")
    # Initialize the chat history
    msgs = StreamlitChatMessageHistory()
    
    # Create two columns: one for chat and one for image upload
    chat_col, image_col = st.columns([3, 2])
    
    with image_col:
        st.subheader("Image Upload")
        uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            context = get_image_caption(image)
            # uploaded = True
            # st.markdown(context)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert image to bytes for processing
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format)
            st.session_state.uploaded_image = img_byte_arr.getvalue()
            
            st.info("You can now ask questions about this image in the chat!")
        
        # Add a button to clear the uploaded image
        if st.session_state.uploaded_image and st.button("Clear Image"):
            st.session_state.uploaded_image = None
            st.experimental_rerun()
    
    with chat_col:
        st.subheader("Chat with your Travel Assistant")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about flights, hotels, or the uploaded image..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        chain = initialize_chain()
                        response = chain.invoke({'input': prompt})
                        
                        # Display the response
                        st.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # If it's a flight booking response, show additional options
                        if "ticket booked" in str(response).lower() or "room booked" in str(response).lower():
                            st.success("✅ Flight booked successfully!")
                            st.download_button(
                                label="Download Ticket",
                                data=str(response),
                                file_name="ticket.txt",
                                mime="text/plain"
                            )
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.info("Please try rephrasing your request or contact support if the issue persists.")
    
    # Sidebar for user information
    with st.sidebar:
        st.header("User Information")
        passenger_id = st.text_input("Passenger ID", value="", key="passenger_id")
        st.write("---")
        st.header("Quick Links")
        if st.button("View All Flights"):
            st.code(FLIGHT_DATA, language="text")

if __name__ == "__main__":
    main()
