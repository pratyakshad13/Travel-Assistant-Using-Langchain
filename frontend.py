from operator import itemgetter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory 
import streamlit as st
from temporary_data import FLIGHT_DATA
from PIL import Image
import io
from travel_agent import get_image_caption
import requests

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

def get_ollama_response(input_text):
    response = requests.post(
        "http://localhost:8000/agent/invoke",
        json={'input': {'input': input_text}})
    print(response)
    return response.json()['output']

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
                        response = get_ollama_response(prompt)
                        print
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
  