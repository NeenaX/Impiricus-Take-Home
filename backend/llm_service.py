import logging
from typing import List, Dict
from transformers import pipeline

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM answer generation"""

    def __init__(self, provider: str = "transformers", model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize LLM service

        Args:
            provider: "transformers" (default), "template" if LLM not loaded
            model: Hugging Face model name (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
        """
        self.provider = provider.lower()
        self.model = model
        self.client = None

        if self.provider == "transformers":
            try:
                logger.info(f"Loading local transformer model: {self.model}")
                self.client = pipeline(
                    "text-generation",
                    model=self.model,
                    dtype="auto",
                    device_map="auto",
                )
                logger.info(f"Successfully loaded model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to load transformers model: {e}")
                logger.warning("Falling back to template-based responses.")
                self.provider = "template"
                self.client = None

        elif self.provider == "template":
            logger.info("Using no LLM loaded, using template answers")
            self.client = None

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate_answer(
        self,
        query: str,
        context: str,
        courses: List[Dict],
        max_tokens: int = 500,
    ) -> str:
        """
        Generate an answer using the LLM or nothing if generation fails
        """
        if self.client is None or self.provider == "template":
            return

        try:
            prompt = (
                "You are a helpful academic advisor at Brown University.\n"
                "Use the course catalog information to answer the student's question clearly and concisely.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )

            response = self.client(
                prompt,
                max_new_tokens=min(max_tokens, 300),
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

            answer = response[0]["generated_text"].split("Answer:")[-1].strip()
            logger.info(f"Generated answer using {self.model}")
            return answer or "I'm sorry, I couldn’t generate an appropriate answer."

        except Exception as e:
            logger.error(f"Error generating answer with {self.model}: {e}")
            logger.info("Falling back to template-based response.")
            return

