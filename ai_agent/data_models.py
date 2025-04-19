from typing import List
from pydantic import BaseModel, Field


class Ingredient(BaseModel):
    name: str = Field(description="Name of the ingredient in polish language")
    quantity: str = Field(
        description="Quantity of the ingredient. Consider only the number or fraction, text is not needed"
    )
    unit: str = Field(description="Unit of measurement, if applicable")


class IngredientsOutput(BaseModel):
    ingredients: List[Ingredient] = Field(description="List of ingredients in the recipe")


class IngredientNameToCommonName(BaseModel):
    original_name: str = Field(description="Original name of the ingredient")
    target_name: str = Field(
        description="Common name of the ingredient in Polish language, might be the same as original name"
    )


class IngredientNamesOutput(BaseModel):
    ingredient_names: list[IngredientNameToCommonName] = Field(
        description="List of ingredient names in Polish language"
    )
