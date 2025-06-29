# Travel Assistant Chatbot

This project is a multimodal travel assistant chatbot built with Streamlit, LangChain, and various NLP tools to assist users with flight and hotel bookings, travel guidance, and general travel-related queries. The chatbot can search for flights and hotels, book tickets, and generate captions for uploaded images. 

## Features

- **Flight Search**: Search for available flights based on user queries.
- **Hotel Search**: Search for hotels in a specific location with details like ratings and amenities.
- **Booking Assistant**: Book flights and hotel rooms by providing necessary details such as dates, number of people, and room type.
- **Image Captioning**: Generate captions for uploaded images, useful for understanding photos or landmarks.
- **Travel Advisory**: Offers information on popular attractions, cultural highlights, dining options, and practical travel tips.

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend Models**: [LangChain](https://www.langchain.com/), [HuggingFace](https://huggingface.co/), [Ollama](https://ollama.com/)
- **NLP Models Used**:
  - `llama3-groq-tool-use` from Ollama for general-purpose responses
  - Salesforce BLIP for image captioning
  - Mixtral-8x7B-Instruct for in-depth travel advice and recommendations

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pratyakshad13/Travel-Assistant-Using-Langchain.git
   cd Travel-Assistant-Using-Langchain
   pip install requirements.txt
   ```
   
2. **Run Backend on different instance :
  In travel_agent.py when intializing database pass you information of your database 
  and make sure that your database instance is running 
  ```bash
    python fast_api_backend.py
  ```
3.**Run Front-end :**
   ```bash
   streamlit run frontend.py
  ```
