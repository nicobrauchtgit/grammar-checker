import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from openai import RateLimitError
from fastapi import HTTPException, status
from dotenv import load_dotenv
from src.api_schemas import GrammarCheckResponse


class OpenAIGrammarChecker:
    def __init__(self, model: str = "gpt-4o-mini", max_retry_attempts: int = 3):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.max_retry_attempts = max_retry_attempts

        if not self.api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OPENAI_API_KEY is not set"
            )

        self.chat_openai = ChatOpenAI(api_key=self.api_key, model=model)
        self.structured_llm = self.chat_openai.with_structured_output(
            self.GrammarErrorResponse)

    class GrammarErrorResponse(BaseModel):
        """A list of all errors found in the provided text."""
        class GrammarError(BaseModel):
            """ Structure of a single grammar error. """
            original_sentence: str = Field(
                description="Entire original sentence from start till '.' .")
            corrected_sentence: str = Field(
                description="The corrected sentence with the error fixed")
            error_type: str = Field(
                description="The type of error found in the sentence")

        grammar_errors: list[GrammarError] = Field(
            description="A list of grammar errors found in the text")

    async def check_grammar(self, text: str) -> list[GrammarCheckResponse]:
        for attempt in range(self.max_retry_attempts):
            try:
                error_response = await self.structured_llm.ainvoke(input=text)

                return [
                    GrammarCheckResponse(
                        original_sentence=error.original_sentence,
                        corrected_sentence=error.corrected_sentence,
                        error_type=error.error_type
                    )
                    for error in error_response.grammar_errors
                ]
            except RateLimitError as e:
                if attempt < self.max_retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Rate limit exceeded: {e}"
                    )
            except Exception as parse_error:
                if attempt < self.max_retry_attempts - 1:
                    print(f"Parse attempt {
                          attempt + 1} failed: {str(parse_error)}")
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to parse LLM output after multiple attempts"
                    ) from parse_error

if __name__ == "__main__":
    grammar_checker = OpenAIGrammarChecker()
    result = grammar_checker.check_grammar(
        "The new cafe in town is really really popular. Everyone seems to love the the unique ambiance and cozy atmosphere. The menu offers a variety of options, and the staff are friendly and welcoming. It's a great place to relax and unwind, whether you're meeting friends or just enjoying some alone time. Many people find the the coffee to be exceptional, and the pastries are always fresh and delicious."
    )
    for r in result[0]:
        print(r.sentence_with_error)
