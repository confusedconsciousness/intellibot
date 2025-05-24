# this class uses chain of responsibility pattern to chain multiple AI models together
# if one fails, it will try the next one
from ai.AIModel import get_ai_model

class AIModelChain:
    def __init__(self, models):
        """
        Initializes the AIModelChain with a list of AI models.

        Args:
            models (list): A list of AIModel instances to be used in the chain.
        """
        self.models = models

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate content by passing the prompt through the chain of AI models.

        Args:
            prompt (str): The input text to generate content from.
            **kwargs: Additional parameters for the generation process.

        Returns:
            str: The generated content from the first successful model.
        """
        for model in self.models:
            try:
                return model.generate(prompt, **kwargs)
            except Exception as e:
                print(f"Model {model.get_model()} failed with error: {e}")
        raise RuntimeError("All models in the chain failed to generate content.")


def get_default_ai_model_chain() -> AIModelChain:
    """
    Returns the list of AI models in the chain.

    Returns:
        list: The list of AIModel instances.
    """
    return AIModelChain([
        get_ai_model("gemini", "gemini-2.0-flash"),
        get_ai_model("openai", "gpt-3.5-turbo")
    ])
