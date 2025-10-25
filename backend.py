from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cvxpy as cp
import numpy as np
from food_data import FOOD_DATA
from preferences import PREFERENCES
from typing import List, Optional

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
STANDARD_DAILY_KCAL = PREFERENCES.get("kcalories_daily", 2000)
DEFAULT_DAILY_VALUES = {
    "magnesium_mg": 400,
    "zinc_mg": 11,
    "selenium_ug": 55,
    "potassium_mg": 4700,
    "vitamin_d_ug": 15,
    "vitamin_k2_ug": 90,
    "folate_ug": 400,
    "vitamin_b12_ug": 2.4,
    "omega3_epa_dha_g": 1.1,
    "vitamin_c_mg": 90,
    "vitamin_e_mg": 15,
    "choline_mg": 550,
}

DAILY_VALUES = DEFAULT_DAILY_VALUES.copy()
for key, value in PREFERENCES.items():
    if key.endswith(("_mg", "_ug", "_g")):
        DAILY_VALUES[key] = value

TRACKED_NUTRIENTS = {nutrient for data in FOOD_DATA.values() for nutrient in data.keys()}
CALORIE_WEIGHT = 5.0
MACRO_WEIGHT = 1.0

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MealRequest(BaseModel):
    foods: List[str]
    target_kcal: Optional[float] = None
    target_carbs_percent: Optional[float] = None
    target_protein_percent: Optional[float] = None
    target_fat_percent: Optional[float] = None

@app.post("/optimize_meal")
def optimize_meal(request: MealRequest):
    foods = request.foods
    if not foods:
        raise HTTPException(status_code=400, detail="At least one food must be provided.")

    unknown_foods = [food for food in foods if food not in FOOD_DATA]
    if unknown_foods:
        raise HTTPException(status_code=400, detail=f"Unknown foods requested: {', '.join(unknown_foods)}")

    protein_count = sum(1 for food in foods if FOOD_DATA[food].get("category") == "protein")
    carb_count = sum(1 for food in foods if FOOD_DATA[food].get("category") == "carb")
    if protein_count > 2:
        raise HTTPException(status_code=400, detail="A meal can include at most two primary protein sources.")
    if carb_count > 2:
        raise HTTPException(status_code=400, detail="A meal can include at most two primary carbohydrate sources.")

    meal_kcal_target = PREFERENCES.get("kcalories", request.target_kcal or STANDARD_DAILY_KCAL)
    carbs_percent_target = PREFERENCES.get("carbs_percent", request.target_carbs_percent or 0.0)
    protein_percent_target = PREFERENCES.get("protein_percent", request.target_protein_percent or 0.0)
    fat_percent_target = PREFERENCES.get("fat_percent", request.target_fat_percent or 0.0)

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

    # Compute targets in grams
    carbs_g_target = (meal_kcal_target * (carbs_percent_target / 100.0)) / 4.0
    protein_g_target = (meal_kcal_target * (protein_percent_target / 100.0)) / 4.0
    fat_g_target = (meal_kcal_target * (fat_percent_target / 100.0)) / 9.0
    b = np.array([meal_kcal_target, carbs_g_target, protein_g_target, fat_g_target])

    # Weighted macro objective (calorie errors dominate, macros next)
    b_safe = np.where(b != 0, b, 1.0)
    row_weights = np.array([CALORIE_WEIGHT, MACRO_WEIGHT, MACRO_WEIGHT, MACRO_WEIGHT])

    lower_bounds = np.array([FOOD_DATA[food].get("min_g", 0.0) for food in foods])
    upper_bounds = np.array([FOOD_DATA[food].get("max_g", 1000.0) for food in foods])
    percent_caps = np.array([FOOD_DATA[food].get("percent_cap", 1.0) for food in foods])
    min_percents = np.array([FOOD_DATA[food].get("min_percent", 0.0) for food in foods])

    if np.sum(min_percents) > 1.0 + 1e-6:
        raise HTTPException(status_code=400, detail="Sum of minimum percent requirements exceeds 100% of the meal.")

    if np.any(min_percents - percent_caps > 1e-9):
        raise HTTPException(status_code=400, detail="A food's minimum percent exceeds its cap.")

    x = cp.Variable(len(foods))
    total_mass = cp.sum(x)
    constraints = [
        x >= lower_bounds,
        x <= upper_bounds,
        total_mass >= 1e-3,
    ]
    for idx in range(len(foods)):
        constraints.append(x[idx] <= percent_caps[idx] * total_mass)
        constraints.append(x[idx] >= min_percents[idx] * total_mass)

    macro_residual = (A @ x - b) / b_safe
    macro_penalty = cp.sum_squares(row_weights * macro_residual)

    objective = cp.Minimize(macro_penalty)
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP, warm_start=True)
    except cp.SolverError:
        problem.solve(solver=cp.SCS)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise HTTPException(status_code=500, detail="Optimization failed to converge.")

    amounts_g = np.array(x.value).flatten()
    achieved = A.dot(amounts_g)

    carbs_percent = (achieved[1] * 4.0) / achieved[0] * 100.0 if achieved[0] else 0.0
    protein_percent = (achieved[2] * 4.0) / achieved[0] * 100.0 if achieved[0] else 0.0
    fat_percent = (achieved[3] * 9.0) / achieved[0] * 100.0 if achieved[0] else 0.0

    total_mass_value = amounts_g.sum()
    food_breakdown = []
    for food, amount in zip(foods, amounts_g):
        share_percent = (amount / total_mass_value) * 100 if total_mass_value > 0 else 0.0
        food_breakdown.append({
            "name": food,
            "grams": round(amount, 1),
            "percent_of_meal": round(share_percent, 1)
        })

    result = {
        "foods": food_breakdown,
        "targets": {
            "kcal": round(meal_kcal_target, 1),
            "carbs_percent": round(carbs_percent_target, 1),
            "protein_percent": round(protein_percent_target, 1),
            "fat_percent": round(fat_percent_target, 1)
        },
        "achieved": {
            "kcal": round(achieved[0], 1),
            "carbs_g": round(achieved[1], 1),
            "protein_g": round(achieved[2], 1),
            "fat_g": round(achieved[3], 1),
            "carbs_percent": round(carbs_percent, 1),
            "protein_percent": round(protein_percent, 1),
            "fat_percent": round(fat_percent, 1)
        }
    }

    # Add micronutrients with totals, daily values, and percentages
    scaling_factor = meal_kcal_target / STANDARD_DAILY_KCAL
    desired_micronutrients = [
        key for key in DAILY_VALUES.keys()
        if key in TRACKED_NUTRIENTS
        and key not in {"kcalories", "kcalories_daily", "carbs_percent", "protein_percent", "fat_percent"}
    ]
    micronutrients = {}
    for key in desired_micronutrients:
        daily_value = DAILY_VALUES.get(key, 0.0)
        effective_dv = daily_value * scaling_factor if daily_value else 0.0
        total = sum(amounts_g[i] * FOOD_DATA[foods[i]].get(key, 0.0) / 100 for i in range(len(foods)))
        percentage = (total / effective_dv) * 100 if effective_dv else 0.0
        micronutrients[key] = {
            "total": round(total, 1),
            "daily_value": round(effective_dv, 1),
            "percentage": round(percentage, 1)
        }
    result["micronutrients"] = micronutrients

    # Calculate macro and micro scores
    macro_score, micro_score = calculate_scores(carbs_percent, protein_percent, fat_percent,
                                                 carbs_percent_target, protein_percent_target, fat_percent_target,
                                                 micronutrients)
    result["macro_score"] = macro_score
    result["micro_score"] = micro_score

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)