from pydantic import BaseModel
from typing import Dict, Any


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]
