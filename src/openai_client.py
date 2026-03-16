import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
