from typing import Any, Dict, Mapping

from ..utils.typing import Schema


def validate(data: Mapping[str, Any], schema: Schema) -> Dict[str, Any]:
    """Validate data against a schema and return the validated dict.
    
    Args:
        data: Raw data to validate
        schema: Schema class (e.g., Pydantic model)
        
    Returns:
        Validated data as dict
        
    Raises:
        ValidationError: If data doesn't match schema
    """
    model = schema(**data)
    return model.model_dump()