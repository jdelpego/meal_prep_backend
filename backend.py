from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from scipy.optimize import lsq_linear
from food_data import FOOD_DATA

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
    protein_food: str
    carb_food: str
    veggie_food: str

@app.post("/optimize_meal")
def optimize_meal(request: MealRequest):
    # Prepare per-gram vectors
    protein_data = np.array([
        FOOD_DATA[request.protein_food]["kcal"],
        FOOD_DATA[request.protein_food]["carbs_g"],
        FOOD_DATA[request.protein_food]["protein_g"],
        FOOD_DATA[request.protein_food]["fat_g"]
    ])
    carb_data = np.array([
        FOOD_DATA[request.carb_food]["kcal"],
        FOOD_DATA[request.carb_food]["carbs_g"],
        FOOD_DATA[request.carb_food]["protein_g"],
        FOOD_DATA[request.carb_food]["fat_g"]
    ])
    veggie_data = np.array([
        FOOD_DATA[request.veggie_food]["kcal"],
        FOOD_DATA[request.veggie_food]["carbs_g"],
        FOOD_DATA[request.veggie_food]["protein_g"],
        FOOD_DATA[request.veggie_food]["fat_g"]
    ])
    
    protein_per_g = protein_data / 100.0
    carb_per_g = carb_data / 100.0
    veggie_per_g = veggie_data / 100.0
    A = np.column_stack((protein_per_g, carb_per_g, veggie_per_g))

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

    lower_bounds = np.array([
        FOOD_DATA[request.protein_food]["min_g"],
        FOOD_DATA[request.carb_food]["min_g"],
        FOOD_DATA[request.veggie_food]["min_g"]
    ])
    upper_bounds = np.array([
        FOOD_DATA[request.protein_food]["max_g"],
        FOOD_DATA[request.carb_food]["max_g"],
        FOOD_DATA[request.veggie_food]["max_g"]
    ])

    res = lsq_linear(WA, Wb, bounds=(lower_bounds, upper_bounds), lsmr_tol='auto', verbose=0)
    amounts_g = res.x

    protein_g, carb_g, veggie_g = amounts_g
    achieved = A.dot(amounts_g)

    carbs_percent = (achieved[1] * 4.0) / achieved[0] * 100.0
    protein_percent = (achieved[2] * 4.0) / achieved[0] * 100.0
    fat_percent = (achieved[3] * 9.0) / achieved[0] * 100.0

    result = {
        f"{request.protein_food}_g": round(protein_g, 1),
        f"{request.carb_food}_g": round(carb_g, 1),
        f"{request.veggie_food}_g": round(veggie_g, 1),
        "achieved_kcal": round(achieved[0], 1),
        "achieved_carbs_percent": round(carbs_percent, 1),
        "achieved_protein_percent": round(protein_percent, 1),
        "achieved_fat_percent": round(fat_percent, 1)
    }

    # Add micronutrients
    for key in FOOD_DATA[request.protein_food].keys():
        if key not in ["kcal", "carbs_g", "protein_g", "fat_g", "fiber_g", "sugar_g"]:
            result[key] = round(
                protein_g * FOOD_DATA[request.protein_food][key] / 100 +
                carb_g * FOOD_DATA[request.carb_food][key] / 100 +
                veggie_g * FOOD_DATA[request.veggie_food][key] / 100,
                1
            )

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)