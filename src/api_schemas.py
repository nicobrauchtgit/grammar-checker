from pydantic import BaseModel

class GrammarCheckRequest(BaseModel):
    text: str

class GrammarCheckResponse(BaseModel):
    original_sentence: str
    corrected_sentence: str
    error_type: str
