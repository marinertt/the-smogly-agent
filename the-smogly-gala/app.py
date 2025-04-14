# Import necessary libraries
import gradio as gr
from smolagents import CodeAgent, HfApiModel
from langchain.memory import ConversationBufferMemory

# Import our custom tools from their modules
from tools import DuckDuckGoSearchTool, WeatherTool, NewsTool, HubStatsTool
from retriever import load_guest_dataset

# Initialize Memory
memory = ConversationBufferMemory(return_messages=True)

# Initialize the Hugging Face model
model = HfApiModel()

# Initialize the web search tool
search_tool = DuckDuckGoSearchTool()

# Initialize news tool
news_tool = NewsTool()

# Initialize the weather tool
weather_tool = WeatherTool()

# Initialize the Hub stats tool
hub_stats_tool = HubStatsTool()

# Load the guest dataset and initialize the guest info tool
guest_info_tool = load_guest_dataset()

# Create Alfred with all the tools
alfred = CodeAgent(
    tools=[
        guest_info_tool, 
        news_tool, 
        weather_tool, 
        hub_stats_tool, 
        search_tool
        ], 
    model=model,
    add_base_tools=True,  
    planning_interval=3,
    memory=memory
)

def ask_alfred(prompt):
    return alfred.run(prompt)

iface = gr.Interface(fn=ask_alfred, inputs="text", outputs="text", title="🎩 Alfred the Gala Agent")

if __name__ == "__main__":
    iface.launch()