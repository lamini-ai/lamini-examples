from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple, Union
from pydantic import ValidationError, validator
import logging

logger = logging.getLogger(__name__)


class GeneratePayload_InValue(BaseModel):
    prompt: str


class GeneratePayload_OutType(BaseModel):
    out_type: Dict[str, str]


class GeneratePayload(BaseModel, smart_union=True):
    prompt: List[str]  # further type checking with @validator
    out_type: Dict[str, str]
    model_name: str
    prompt_template: Optional[str] = None
    max_tokens: Optional[int] = None

    @validator("prompt")
    def check_type(cls, v):
        try:
            for d in v:
                GeneratePayload_InValue(prompt=d)
        except ValidationError as e:
            err_msg = (
                "Prompt must be a list of strings"
            )
            raise TypeError(err_msg)

        return v

    @validator("out_type")
    def check_type_out_type(cls, v):
        valid_types = {
            "string",
            "str",
            "integer",
            "int",
            "float",
            "bool",
            "boolean",
        }
        valid_type_str = "string, str, integer, int, float, boolean, bool"
        for key, val in v.items():
            if val not in valid_types:
                raise TypeError("Each value must be one of: " + valid_type_str)

        return v
