from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cvxpy as cp
import numpy as np
from food_data import FOOD_DATA
from preferences import PREFERENCES
from typing import List, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/request_meal")
def request_meal(foods: List[str]):
    foods = foods
    meal_kcal_target = PREFERENCES.get("kcalories")
    carbs_percent_target = PREFERENCES.get("carbs_percent")
    protein_percent_target = PREFERENCES.get("protein_percent")
    fat_percent_target = PREFERENCES.get("fat_percent")

    # Prepare per-gram vectors
    A = np.column_stack([np.array([FOOD_DATA[food]["kcal"], FOOD_DATA[food]["carbs_g"], FOOD_DATA[food]["protein_g"], FOOD_DATA[food]["fat_g"]]) / 100.0 for food in foods])

    # Compute targets in grams
    b = np.array([meal_kcal_target, (meal_kcal_target * (carbs_percent_target / 100.0)) / 4.0, (meal_kcal_target * (protein_percent_target / 100.0)) / 4.0, (meal_kcal_target * (fat_percent_target / 100.0)) / 9.0])

    # Calories, Carb, Protein Fat weights
    row_weights = np.array([2.0, 1.0, 1.0, 1.0])



    amounts_g = np.array(x.value).flatten()
    achieved = A.dot(amounts_g)

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)