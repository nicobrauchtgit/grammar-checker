from fastapi import FastAPI, HTTPException, Depends
from src.api_schemas import GrammarCheckRequest, GrammarCheckResponse
from src.grammer_checker import OpenAIGrammarChecker
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_grammar_checker():
    return OpenAIGrammarChecker()


@app.get("/", response_model=str)
async def initial_call() -> str:
    """
    Initial endpoint to check the API status.
    """
    return "Check your grammar at /check-grammar or the api documentation at /docs"


@app.post("/check-grammar", response_model=list[GrammarCheckResponse])
async def check_grammar(
    request: GrammarCheckRequest,
    grammar_checker: OpenAIGrammarChecker = Depends(get_grammar_checker)
) -> list[GrammarCheckResponse]:
    """
    Endpoint to check grammar of the provided text.
    """
    request_text = request.text

    if len(request_text) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        return await grammar_checker.check_grammar(request_text)
    except Exception as e:
        logger.error(f"Error checking grammar: {e}")
        raise HTTPException(status_code=500, detail=str(e))
