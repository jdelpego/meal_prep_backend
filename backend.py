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

# Food data per 100g: [kcal, carbs_g, protein_g, fat_g]
chicken = np.array([
    FOOD_DATA["chicken"]["kcal"],
    FOOD_DATA["chicken"]["carbs_g"],
    FOOD_DATA["chicken"]["protein_g"],
    FOOD_DATA["chicken"]["fat_g"]
])
rice = np.array([
    FOOD_DATA["white_rice"]["kcal"],
    FOOD_DATA["white_rice"]["carbs_g"],
    FOOD_DATA["white_rice"]["protein_g"],
    FOOD_DATA["white_rice"]["fat_g"]
])
avocado = np.array([
    FOOD_DATA["avocado"]["kcal"],
    FOOD_DATA["avocado"]["carbs_g"],
    FOOD_DATA["avocado"]["protein_g"],
    FOOD_DATA["avocado"]["fat_g"]
])

# Bounds in grams
min_chicken_g = 0.0
max_chicken_g = 10000.0
min_rice_g = 0.0
max_rice_g = 10000.0
min_avocado_g = 0.0
max_avocado_g = 10000.0

class MealRequest(BaseModel):
    target_kcal: float
    target_carbs_percent: float
    target_protein_percent: float
    target_fat_percent: float

@app.post("/optimize_meal")
def optimize_meal(request: MealRequest):
    # Prepare per-gram vectors
    chicken_per_g = chicken / 100.0
    rice_per_g = rice / 100.0
    avocado_per_g = avocado / 100.0
    A = np.column_stack((chicken_per_g, rice_per_g, avocado_per_g))

    # Compute targets
    carbs_g_target = (request.target_kcal * request.target_carbs_percent) / 4.0
    protein_g_target = (request.target_kcal * request.target_protein_percent) / 4.0
    fat_g_target = (request.target_kcal * request.target_fat_percent) / 9.0
    b = np.array([request.target_kcal, carbs_g_target, protein_g_target, fat_g_target])

    # Normalize and weight
    A_s = A / b[:, None]
    b_s = b / b
    row_weights = np.array([1.0, 5.0, 5.0, 5.0])
    WA = row_weights[:, None] * A_s
    Wb = row_weights * b_s

    lower_bounds = np.array([min_chicken_g, min_rice_g, min_avocado_g])
    upper_bounds = np.array([max_chicken_g, max_rice_g, max_avocado_g])

    res = lsq_linear(WA, Wb, bounds=(lower_bounds, upper_bounds), lsmr_tol='auto', verbose=0)
    amounts_g = res.x

    chicken_g, rice_g, avocado_g = amounts_g
    achieved = A.dot(amounts_g)

    carbs_percent = (achieved[1] * 4.0) / achieved[0] * 100.0
    protein_percent = (achieved[2] * 4.0) / achieved[0] * 100.0
    fat_percent = (achieved[3] * 9.0) / achieved[0] * 100.0

    result = {
        "chicken_g": round(chicken_g, 1),
        "rice_g": round(rice_g, 1),
        "avocado_g": round(avocado_g, 1),
        "achieved_kcal": round(achieved[0], 1),
        "achieved_carbs_percent": round(carbs_percent, 1),
        "achieved_protein_percent": round(protein_percent, 1),
        "achieved_fat_percent": round(fat_percent, 1)
    }

    # Add micronutrients
    for key in FOOD_DATA["chicken"].keys():
        if key not in ["kcal", "carbs_g", "protein_g", "fat_g", "fiber_g", "sugar_g"]:
            result[key] = round(
                chicken_g * FOOD_DATA["chicken"][key] / 100 +
                rice_g * FOOD_DATA["white_rice"][key] / 100 +
                avocado_g * FOOD_DATA["avocado"][key] / 100,
                1
            )

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)