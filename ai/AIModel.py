from abc import ABC, abstractmethod


# AIModel is an abstract base class for AI models.
class AIModel(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate content based on the provided prompt.

        Args:
            prompt (str): The input text to generate content from.
            **kwargs: Additional parameters for the generation process.

        Returns:
            str: The generated content.
        """
        pass

    @abstractmethod
    def get_model(self):
        """
        Returns the AI model information.

        Returns:
            object: The AI model information.
        """
        pass


import google.generativeai as genai
import openai
import os


class GenAIModel(AIModel):
    """
        Concrete implementation of AIModel using Google's Gemini API.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Initializes the GenAI with the specified model name.

        Args:
            model_name (str): The name of the generative model to use.
        """

        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self._model_name = model_name
        self._model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate content based on the provided prompt.

        Args:
            prompt (str): The input text to generate content from.
            **kwargs: Additional parameters for the generation process.

        Returns:
            str: The generated content.
        """
        try:
            response = self._model.generate_content(prompt, **kwargs)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Failed to generate content: {str(e)}")

    def get_model(self):
        """
        Returns the initialized generative model.

        Returns:
            google.generativeai.GenerativeModel: The generative model instance.
        """
        return {
            "model_name": self._model_name,
            "model_instance": self._model
        }


class OpenAIModel(AIModel):
    """
        Concrete implementation of AIModel using OpenAI's API.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initializes the OpenAI model with the specified model name.

        Args:
            model_name (str): The name of the OpenAI model to use.
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self._api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self._api_key
        self._model_name = model_name
        self._client = openai.OpenAI(api_key=self._api_key)

    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generates text using the OpenAI API.

        Args:
            prompt (str): The input prompt.
            **kwargs: Additional parameters for OpenAI (e.g., `temperature`, `max_tokens`).

        Returns:
            str: The generated text.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating text with OpenAIModel: {e}")
            return f"Error: {e}"

    def get_model(self):
        """
        Returns the initialized OpenAI model information.

        Returns:
            dict: The OpenAI model information.
        """
        return {
            "model_name": self._model_name,
            "model_instance": None  # OpenAI does not expose a direct instance like GenAI
        }


# factory function to create AIModel instances
def get_ai_model(model_type: str = "gemini", model_name: str = "gemini-2.0-flash") -> AIModel:
    """
    Factory function to create an instance of AIModel based on the specified type.

    Args:
        model_type (str): The type of AI model to create ('gemini' or 'openai').
        model_name (str): The name of the model to use.

    Returns:
        AIModel: An instance of the specified AI model.
    """
    if model_type == "gemini":
        return GenAIModel(model_name)
    elif model_type == "openai":
        return OpenAIModel(model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
