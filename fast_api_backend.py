import uvicorn
import os
from fastapi import FastAPI
from langserve import add_routes
from travel_agent import initialize_chain , image_caption_tool
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="Langchain LLM API",
    version="1.0",
    description="A Multi-LLM API with Langchain"
)
                
# Initialize LLM Models
model = initialize_chain()

# Create API Routes with Langchain Pipelines
add_routes(app, model, path="/agent")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)