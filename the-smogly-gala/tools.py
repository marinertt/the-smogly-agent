from smolagents import Tool, DuckDuckGoSearchTool
from huggingface_hub import list_models
import os
import requests


class NewsTool(Tool):
    name = "news_tool"
    description = "Fetches the latest news headlines related to a specific topic."

    inputs = {
        "topic": {
            "type": "string",
            "description": "The news topic to search for, like 'AI', 'Ukraine', or 'climate change'.",
        }
    }
    output_type = "string"

    def __init__(self):
        self.api_key = os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError("Please set the NEWS_API_KEY environment variable.")

    def forward(self, topic: str):
        url = (
            f"https://newsapi.org/v2/everything?q={topic}&sortBy=publishedAt"
            f"&language=en&pageSize=3&apiKey={self.api_key}"
        )

        response = requests.get(url)
        if response.status_code != 200:
            return f"Failed to fetch news for '{topic}'."

        articles = response.json().get("articles", [])
        if not articles:
            return f"No news found for '{topic}'."

        news_lines = [
            f"- {a['title']} ({a['source']['name']})\n  {a['url']}" for a in articles
        ]
        return f"🗞️ Latest news on '{topic}':\n" + "\n\n".join(news_lines)


class WeatherTool(Tool):
    name = "weather_tool"
    description = "Get the current weather for a given city."

    inputs = {
        "location": {
            "type": "string",
            "description": "City name to get the weather for, e.g., 'Paris'",
        }
    }
    output_type = "string"

    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        if not self.api_key:
            raise ValueError("Please set the OPENWEATHER_API_KEY environment variable.")

    def forward(self, location: str):
        url = (
            f"http://api.openweathermap.org/data/2.5/weather?q={location}"
            f"&appid={self.api_key}&units=metric"
        )
        response = requests.get(url)
        if response.status_code != 200:
            return f"Failed to get weather for {location}."

        data = response.json()
        weather = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]

        return (
            f"Weather in {location}:\n"
            f"- Condition: {weather}\n"
            f"- Temperature: {temp}°C (feels like {feels_like}°C)"
        )


class HubStatsTool(Tool):
    name = "hub_stats"
    description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    inputs = {
        "author": {
            "type": "string",
            "description": "The username of the model author/organization to find models from.",
        }
    }
    output_type = "string"

    def forward(self, author: str):
        try:
            # List models from the specified author, sorted by downloads
            models = list(
                list_models(author=author, sort="downloads", direction=-1, limit=1)
            )

            if models:
                model = models[0]
                return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
            else:
                return f"No models found for author {author}."
        except Exception as e:
            return f"Error fetching models for {author}: {str(e)}"


# Initialize the tool
hub_stats_tool = HubStatsTool()
weather_tool = WeatherTool()
search_tool = DuckDuckGoSearchTool()
news_tool = NewsTool()
