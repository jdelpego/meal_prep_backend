from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from scipy.optimize import lsq_linear
from food_data import FOOD_DATA
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MealRequest(BaseModel):
    target_kcal: float
    target_carbs_percent: float
    target_protein_percent: float
    target_fat_percent: float
    foods: List[str]

@app.post("/optimize_meal")
def optimize_meal(request: MealRequest):
    foods = request.foods
    # Prepare per-gram vectors
    food_data_list = []
    for food in foods:
        food_data = np.array([
            FOOD_DATA[food]["kcal"],
            FOOD_DATA[food]["carbs_g"],
            FOOD_DATA[food]["protein_g"],
            FOOD_DATA[food]["fat_g"]
        ])
        food_data_list.append(food_data / 100.0)
    A = np.column_stack(food_data_list)

    # Compute targets
    carbs_g_target = (request.target_kcal * (request.target_carbs_percent / 100.0)) / 4.0
    protein_g_target = (request.target_kcal * (request.target_protein_percent / 100.0)) / 4.0
    fat_g_target = (request.target_kcal * (request.target_fat_percent / 100.0)) / 9.0
    b = np.array([request.target_kcal, carbs_g_target, protein_g_target, fat_g_target])

    # Normalize and weight
    A_s = A / b[:, None]
    b_s = b / b
    row_weights = np.array([1.0, 5.0, 5.0, 5.0])
    WA = row_weights[:, None] * A_s
    Wb = row_weights * b_s

    lower_bounds = np.array([FOOD_DATA[food]["min_g"] for food in foods])
    upper_bounds = np.array([FOOD_DATA[food]["max_g"] for food in foods])

    res = lsq_linear(WA, Wb, bounds=(lower_bounds, upper_bounds), lsmr_tol='auto', verbose=0)
    amounts_g = res.x

    achieved = A.dot(amounts_g)

    carbs_percent = (achieved[1] * 4.0) / achieved[0] * 100.0
    protein_percent = (achieved[2] * 4.0) / achieved[0] * 100.0
    fat_percent = (achieved[3] * 9.0) / achieved[0] * 100.0

    result = {
        "foods": [{"name": food, "grams": round(amount, 1)} for food, amount in zip(foods, amounts_g)],
        "achieved_kcal": round(achieved[0], 1),
        "achieved_carbs_percent": round(carbs_percent, 1),
        "achieved_protein_percent": round(protein_percent, 1),
        "achieved_fat_percent": round(fat_percent, 1)
    }

    # Add only the specified longevity-related micronutrients
    desired_micronutrients = [
        "magnesium_mg", "zinc_mg", "selenium_ug", "potassium_mg", "calcium_mg", "iron_mg", "copper_mg", "manganese_mg", "iodine_ug",
        "vitamin_d_ug", "vitamin_k_ug", "vitamin_a_ug", "vitamin_e_mg", "vitamin_c_mg",
        "thiamin_mg", "riboflavin_mg", "niacin_mg", "pantothenic_acid_mg", "vitamin_b6_mg", "biotin_ug", "folate_ug", "vitamin_b12_ug",
        "choline_mg", "epa_g", "dha_g"
    ]
    micronutrients = {}
    for key in desired_micronutrients:
        if key in FOOD_DATA[foods[0]]:
            total = sum(amounts_g[i] * FOOD_DATA[foods[i]][key] / 100 for i in range(len(foods)))
            micronutrients[key] = round(total, 1)
    result.update(micronutrients)

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)