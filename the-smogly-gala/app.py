# Import necessary libraries
import random
from smolagents import CodeAgent, HfApiModel

# Import our custom tools from their modules
from tools import DuckDuckGoSearchTool, WeatherTool, NewsTool, HubStatsTool
from retriever import load_guest_dataset
from langchain.memory import ConversationBufferMemory


# Initialize Memory
memory = ConversationBufferMemory(return_messages=True)

# Initialize the Hugging Face model
model = HfApiModel()

# Initialize the web search tool
search_tool = DuckDuckGoSearchTool()

# Initialize news tool
news_tool = NewsTool()

# Initialize the weather tool
weather_info_tool = WeatherTool()

# Initialize the Hub stats tool
hub_stats_tool = HubStatsTool()

# Load the guest dataset and initialize the guest info tool
guest_info_tool = load_guest_dataset()

# Create Alfred with all the tools
alfred = CodeAgent(
    tools=[guest_info_tool, news_tool, weather_info_tool, hub_stats_tool, search_tool], 
    model=model,
    add_base_tools=True,  # Add any additional base tools
    planning_interval=3,   # Enable planning every 3 steps
    memory=memory
)