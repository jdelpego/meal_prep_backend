from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from scipy.optimize import lsq_linear
from food_data import FOOD_DATA
from typing import List

def calculate_scores(achieved_carbs_percent, achieved_protein_percent, achieved_fat_percent,
                     target_carbs_percent, target_protein_percent, target_fat_percent, micronutrients):
    # Macro score: 100% if exactly matching targets, decreases with deviation
    macro_deviation = (abs(achieved_carbs_percent - target_carbs_percent) +
                       abs(achieved_protein_percent - target_protein_percent) +
                       abs(achieved_fat_percent - target_fat_percent)) / 3
    macro_score = max(0, 100 - macro_deviation)
    
    # Micro score: Average of how much each micronutrient meets or exceeds DV (capped at 100% per nutrient)
    micro_scores = []
    for nutrient in micronutrients.values():
        total = nutrient["total"]
        dv = nutrient["daily_value"]
        if dv > 0:
            score = min(1.0, total / dv)
        else:
            score = 1.0  # If no DV, consider it met
        micro_scores.append(score)
    micro_score = (sum(micro_scores) / len(micro_scores)) * 100 if micro_scores else 0
    
    return round(macro_score, 1), round(micro_score, 1)

# Daily Values (RDAs) for an average adult (approximate, based on FDA/USDA guidelines)
DAILY_VALUES = {
    "magnesium_mg": 400,
    "zinc_mg": 11,
    "selenium_ug": 55,
    "potassium_mg": 4700,
    "calcium_mg": 1000,
    "iron_mg": 18,
    "copper_mg": 0.9,
    "manganese_mg": 2.3,
    "iodine_ug": 150,
    "vitamin_d_ug": 15,
    "vitamin_k_ug": 120,
    "vitamin_a_ug": 900,
    "vitamin_e_mg": 15,
    "vitamin_c_mg": 90,
    "thiamin_mg": 1.2,
    "riboflavin_mg": 1.3,
    "niacin_mg": 16,
    "pantothenic_acid_mg": 5,
    "vitamin_b6_mg": 1.7,
    "biotin_ug": 30,
    "folate_ug": 400,
    "vitamin_b12_ug": 2.4,
    "choline_mg": 550,
    "epa_g": 0.25,  # Approximate for omega-3 (EPA+DHA total ~0.5g/day)
    "dha_g": 0.25
}

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

    # Add only the specified longevity-related micronutrients with totals, DVs, and percentages
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
            dv = DAILY_VALUES[key]
            percentage = (total / dv) * 100 if dv > 0 else 0
            micronutrients[key] = {
                "total": round(total, 1),
                "daily_value": dv,
                "percentage": round(percentage, 1)
            }
    result["micronutrients"] = micronutrients

    # Calculate macro and micro scores
    macro_score, micro_score = calculate_scores(carbs_percent, protein_percent, fat_percent,
                                                 request.target_carbs_percent, request.target_protein_percent, request.target_fat_percent,
                                                 micronutrients)
    result["macro_score"] = macro_score
    result["micro_score"] = micro_score

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)