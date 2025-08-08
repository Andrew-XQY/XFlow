try:
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "Pydantic is required for built-in validation schemas. "
        "Install with: pip install xflow-py[validation]"
    )

class TrainerConfig(BaseModel):
    learning_rate: float = Field(..., gt=0)
    epochs: int = Field(1, gt=0)

class DataConfig(BaseModel):
    batch_size: int = Field(..., gt=0)
    
class ModelConfig(BaseModel):
    model_name: str = Field(..., min_length=1)
    num_layers: int = Field(1, ge=1)
