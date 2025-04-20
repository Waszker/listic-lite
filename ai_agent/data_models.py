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


class ConsolidatedIngredientOutput(BaseModel):
    name: str = Field(description="Unified ingredient name in Polish.")
    quantity: float = Field(description="The final calculated quantity needed for shopping.")
    unit: str = Field(description="The final unit for shopping (e.g., 'szt.', 'opak.', 'g', 'ml').")


class UnitConversionOutput(BaseModel):
    quantity: float = Field(description="The numeric quantity after conversion.")
    unit: str = Field(description="The standard unit ('g' or 'ml').")
    explanation: str = Field(description="A brief explanation of the conversion logic.")
