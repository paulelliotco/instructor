from typing import Annotated, Optional
from pydantic import AfterValidator, BaseModel, Field
import pytest
import instructor
from itertools import product
import google.generativeai as genai

from .util import models, modes


def uppercase_validator(v):
    if v is None:
        return v  # Return None as is to avoid AttributeError
    if v.islower():
        raise ValueError("Name must be ALL CAPS")
    return v


class UserDetailWithOptional(BaseModel):
    name: Optional[Annotated[str, AfterValidator(uppercase_validator)]] = Field(
        None, description="The name of the user"
    )
    age: int


def test_none_value_validator():
    """Test that the validator handles None values correctly."""
    # Create a model with a None value for name
    user = UserDetailWithOptional(name=None, age=25)
    assert user.name is None  # Should not raise AttributeError


@pytest.mark.parametrize("model, mode", product(models[:1], modes[:1]))  # Limit to one model/mode for speed
def test_create_with_completion_optional_field(model, mode):
    """Test that create_with_completion handles optional fields correctly."""
    client = instructor.from_gemini(genai.GenerativeModel(model), mode=mode)
    
    try:
        # This should not raise AttributeError
        response, raw_response = client.chat.completions.create_with_completion(
            response_model=UserDetailWithOptional,
            messages=[
                {"role": "user", "content": "Extract age is 25"},
            ],
            max_retries=1,
        )
        # We don't assert on the actual values since we're just testing that no exception is raised
    except AttributeError as e:
        if "'NoneType' object has no attribute 'upper'" in str(e):
            pytest.fail(f"AttributeError still occurs: {e}")
        else:
            # If it's a different AttributeError, re-raise it
            raise
