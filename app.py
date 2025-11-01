from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from scipy.optimize import lsq_linear
from food_data import FOOD_DATA
from presets import PRESETS
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MealRequest(BaseModel):
    foods: list[str]
    kcalories: int = 700
    carbs_percent: int = 40
    protein_percent: int = 30
    fat_percent: int = 30

@app.post("/optimize_meal_prep")
def optimize_meal_prep(request: MealRequest):
    
    # Extract parameters from request
    foods = request.foods
    kcalories = request.kcalories
    carbs_percent = request.carbs_percent
    protein_percent = request.protein_percent
    fat_percent = request.fat_percent
    
    columns = []
    for food in foods:
        column = []
        for category in PRESETS['weights'].keys():
            if category == "vegetable_g":
                column.append(1.0 if FOOD_DATA[food]['category'] == "vegetable" else 0.0)
            else:
                column.append(FOOD_DATA[food][category] / 100.0)
        columns.append(column)
    A = np.column_stack(columns)
    
    # Use parameters instead of PRESETS['targets']
    target_kcalories = kcalories
    target_carbs_g = (target_kcalories * (carbs_percent / 100.0)) / 4.0
    target_protein_g = (target_kcalories * (protein_percent / 100.0)) / 4.0
    target_fat_g = (target_kcalories * (fat_percent / 100.0)) / 9.0
    target_vegetable_g = target_kcalories * PRESETS['targets']['vegetable_g_calorie_ratio']

    targets = []
    for category in PRESETS['weights'].keys():
        if category == "kcalories":
            targets.append(target_kcalories)
        elif category == "carbs_g":
            targets.append(target_carbs_g)
        elif category == "protein_g":
            targets.append(target_protein_g)
        elif category == "fat_g":
            targets.append(target_fat_g)
        elif category == "vegetable_g":
            targets.append(target_vegetable_g)
        else:
            # Micronutrients - scale by calorie ratio
            targets.append(PRESETS['daily_values']['micronutrients'][category] * (target_kcalories / PRESETS['daily_values']['kcalories']))
        
    b = np.array(targets)

    # Normalize weights by target magnitude for fair comparison
    # Without this, small targets (like fat_g=23g) get ignored vs large targets (carbs_g=70g)
    base_weights = [weight for weight in PRESETS['weights'].values()]
    
    normalized_weights = []
    for weight, target in zip(base_weights, targets):
        if target > 1:  # Only normalize non-zero, non-trivial targets
            # Divide by sqrt(target) for gentler normalization
            # This balances between absolute and percentage-based errors
            normalized_weights.append(weight / (target ** 0.5))
        else:
            normalized_weights.append(weight)
    
    W_sqrt = np.diag(np.sqrt(np.array(normalized_weights)))
    
    # Set minimum and maximum bounds for all foods
    # Min: 10g prevents trace amounts (0.5g of broccoli is pointless)
    # Max: 400g prevents unrealistic single-food dominance
    min_amount = 10  # grams
    max_amount = 400  # grams
    
    lower_bounds = np.full(len(foods), min_amount)
    upper_bounds = np.full(len(foods), max_amount)
    
    # Solve weighted least squares with bounds
    result_obj = lsq_linear(W_sqrt @ A, W_sqrt @ b, bounds=(lower_bounds, upper_bounds))
    x = result_obj.x
    
    # Build targets_dict using the parameters passed to the function
    targets_dict = {
        "kcalories": kcalories,
        "carbs_percent": carbs_percent,
        "protein_percent": protein_percent,
        "fat_percent": fat_percent,
        "vegetable_g_calorie_ratio": PRESETS['targets']['vegetable_g_calorie_ratio']
    }
    for category, _ in PRESETS['daily_values']['micronutrients'].items():
        targets_dict[category] = round(PRESETS['daily_values']['micronutrients'][category] * (target_kcalories / PRESETS['daily_values']['kcalories']), 2)
        
    results_dict = {}
    for i, food in enumerate(foods):
        for category in PRESETS['weights'].keys():
            if category == "vegetable_g":
                if FOOD_DATA[food]['category'] == "vegetable":
                    if category not in results_dict:
                        results_dict[category] = 0
                    results_dict[category] += x[i]
            else:
                if category not in results_dict:
                    results_dict[category] = 0
                results_dict[category] += FOOD_DATA[food][category] * x[i] / 100.0

    # Calculate actual vegetable weight from optimized solution (do this FIRST)
    total_vegetable_g = sum([x[i] for i, food in enumerate(foods) if FOOD_DATA[food]['category'] == "vegetable"])
    total_meal_weight = sum(x)
    
    results_dict['vegetable_g'] = total_vegetable_g
    results_dict['vegetable_weight_percent'] = (total_vegetable_g / total_meal_weight) * 100.0
    
    # Now calculate percentages (using vegetable_g that was just set)
    results_dict['carbs_percent'] = (results_dict['carbs_g'] * 4.0) / results_dict['kcalories'] * 100.0
    results_dict['protein_percent'] = (results_dict['protein_g'] * 4.0) / results_dict['kcalories'] * 100.0
    results_dict['fat_percent'] = (results_dict['fat_g'] * 9.0) / results_dict['kcalories'] * 100.0
    results_dict['vegetable_calorie_ratio'] = results_dict['vegetable_g'] / results_dict['kcalories']
    

    # Round all results to 2 decimal places
    targets_dict = {k: round(v, 2) for k, v in targets_dict.items()}
    results_dict = {k: round(v, 2) for k, v in results_dict.items()}
    
    macro_score = max(0.0, 100.0 - (2.0 / 3.0) * sum(abs(results_dict[k] - targets_dict[k]) for k in ("carbs_percent", "protein_percent", "fat_percent")))
    micros = PRESETS["daily_values"]["micronutrients"]
    micro_score = sum(
    100.0 if not targets_dict.get(k) else min(100.0, results_dict.get(k, 0.0) * 100.0 / targets_dict[k])
    for k in micros) / len(micros)


    # Result in grams
    result = {
        "recipe": {food: round(x[i], 1) for i, food in enumerate(foods)},
        "nutrition_targets": targets_dict,
        "nutrition_results": results_dict,
        "scores": {"macro": round(macro_score, 2), "micro": round(micro_score, 2)},
    }
       
    print(result)
    return result

@app.post("/find_missing_ingredient")
def find_missing_ingredient(request: MealRequest):
    """Suggests missing ingredients to fill micronutrient gaps."""
    # 1️⃣ Run optimizer
    nutrition = optimize_meal_prep(request)
    
    if(nutrition['scores']['macro'] > 90.0 and len(request.foods) > 4):
        return {"ingredients": []} 
    
    results = nutrition["nutrition_results"]
    targets = nutrition["nutrition_targets"]

    # 2️⃣ Find micronutrient gaps (>5%)
    micronutrients = list(PRESETS["daily_values"]["micronutrients"].keys())
    micro_gap = {}
    for m in micronutrients:
        target = targets.get(m, 0)
        current = results.get(m, 0)
        if target > 0:
            gap = (target - current) / target
            if gap > 0.05:
                micro_gap[m] = gap

    if not micro_gap:
        return {"ingredients": []}

    # 3️⃣ Build nutrient matrix for all foods
    food_names = list(FOOD_DATA.keys())
    nutrient_matrix = np.array([[FOOD_DATA[f][m] for m in micronutrients] for f in food_names])
    gap_vector = np.array([micro_gap.get(m, 0.0) for m in micronutrients])

    # 4️⃣ Compute cosine similarity
    sims = cosine_similarity(nutrient_matrix, gap_vector.reshape(1, -1)).flatten()

    # 5️⃣ Exclude already used foods
    candidate_scores = [(f, float(sims[i])) for i, f in enumerate(food_names) if f not in request.foods]
    candidate_scores.sort(key=lambda x: x[1], reverse=True)

    # 6️⃣ Return top 3 names only
    top_ingredients = [f for f, _ in candidate_scores[:1]]

    print(top_ingredients)
    return {"ingredients": top_ingredients}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)