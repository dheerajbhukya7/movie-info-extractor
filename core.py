from itertools import chain
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# 1. Initialize model
model = ChatMistralAI(
    model="open-mistral-7b",  # or "mistral-small-latest"
    temperature=0.3,
    api_key=os.getenv("MISTRAL_API_KEY"),
)

# SCHEMA
class MovieInfo(BaseModel):
    title: str
    release_year: Optional[int] = None
    genres: Optional[List[str]] = None
    director: Optional[str] = None
    cast: Optional[List[str]] = None
    rating: Optional[float] = None
    summary: Optional[str] = None

# checks the output of the model and parses it into the MovieInfo
parser = PydanticOutputParser(pydantic_object=MovieInfo)

# 2. Prompt Template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a movie information extractor. You will be given
        a text description of a movie, and your task is to extract the
        following information from it:

        {format_instructions}""",
    ),
    ("human", "{paragraph}"),
])


# 4. Input Text
para = input("Enter the movie text to analyze: ")

final_prompt = prompt.invoke(
    {
        "paragraph": para,
        "format_instructions": parser.get_format_instructions()
    }
)

response = model.invoke(final_prompt)

print(response.content)