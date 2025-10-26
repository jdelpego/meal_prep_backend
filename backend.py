from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from scipy.optimize import lsq_linear
from food_data import FOOD_DATA
from presets import PRESETS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/optimize_meal_prep")
def optimize_meal_prep(foods: list[str]):
    
    columns = []
    for food in foods:
        column = [FOOD_DATA[food][category]/100.0 for category, _ in PRESETS['weights'].items() if category != "vegetable_g"]
        column.insert(4, 1.0 if FOOD_DATA[food]['category'] == "vegetable" else 0.0)
        columns.append(column)
    A = np.column_stack(columns)
    
    # Compute targets in grams
    target_kcalories = PRESETS['targets']["kcalories"]
    target_carbs_g = (target_kcalories * (PRESETS['targets']["carbs_percent"] / 100.0)) / 4.0
    target_protein_g = (target_kcalories * (PRESETS['targets']["protein_percent"] / 100.0)) / 4.0
    target_fat_g = (target_kcalories * (PRESETS['targets']["fat_percent"] / 100.0)) / 9.0
    target_vegetable_g = target_kcalories * (PRESETS['targets']['vegetable_g_calorie_ratio'])


    targets = [
        target_kcalories,
        target_carbs_g,
        target_protein_g,
        target_fat_g,
        target_vegetable_g
    ]
    for category, _ in PRESETS['daily_values']['micronutrients'].items():
        targets.append(PRESETS['daily_values']['micronutrients'][category] * (target_kcalories / PRESETS['daily_values']['kcalories']))
        
    b = np.array(targets)

    # Calories, Carb, Protein, Fat weights
    base_weights = np.array([weight for _, weight in PRESETS['weights'].items()])
    W_sqrt = np.diag(np.sqrt(base_weights))
    
    # Solve weighted least squares with bounds
    result_obj = lsq_linear(W_sqrt @ A, W_sqrt @ b, bounds=(0, np.inf))
    x = result_obj.x
    
    targets_dict = {label: round(target, 2) for label, target in PRESETS['targets'].items()}
    for category, _ in PRESETS['daily_values']['micronutrients'].items():
        targets_dict[category] = round(PRESETS['daily_values']['micronutrients'][category] * (target_kcalories / PRESETS['daily_values']['kcalories']), 2)
        
    results_dict = {}
    for i, food in enumerate(foods):
        for category, _ in PRESETS['weights'].items():
            if category != "vegetable_g":
                if category not in results_dict:
                    results_dict[category] = 0
                results_dict[category] += FOOD_DATA[food][category] * x[i] / 100.0
            else:
                if FOOD_DATA[food]['category'] == "vegetable":
                    if category not in results_dict:
                        results_dict[category] = 0
                    results_dict[category] += x[i]

    results_dict['carbs_percent'] = (results_dict['carbs_g'] * 4.0) / results_dict['kcalories'] * 100.0
    results_dict['protein_percent'] = (results_dict['protein_g'] * 4.0) / results_dict['kcalories'] * 100.0
    results_dict['fat_percent'] = (results_dict['fat_g'] * 9.0) / results_dict['kcalories'] * 100.0
    results_dict['vegetable_calorie_ratio'] = results_dict['vegetable_g'] / results_dict['kcalories']
        
    # Calculate actual vegetable weight from optimized solution
    total_vegetable_g = sum([x[i] for i, food in enumerate(foods) if FOOD_DATA[food]['category'] == "vegetable"])
    total_meal_weight = sum(x)
    
    results_dict['vegetable_g'] = total_vegetable_g
    results_dict['vegetable_weight_percent'] = (total_vegetable_g / total_meal_weight) * 100.0

    # Round all results to 2 decimal places
    results_dict = {k: round(v, 2) for k, v in results_dict.items()}
    
    # Result in grams
    result = {
        "recipe": {food: round(x[i], 1) for i, food in enumerate(foods)},
        "nutrition_targets": targets_dict,
        "nutrition_results": results_dict
    }
       
    print(result)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)