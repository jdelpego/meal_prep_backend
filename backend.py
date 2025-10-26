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
    
    # Build constraint matrix A where each column is a food's nutritional profile
    # Order matches PRESETS['weights'] dictionary order
    columns = []
    for food in foods:
        column = []
        for category in PRESETS['weights'].keys():
            if category == "vegetable_g":
                # Special case: 1.0 if vegetable, 0.0 otherwise
                column.append(1.0 if FOOD_DATA[food]['category'] == "vegetable" else 0.0)
            else:
                # Regular nutritional values per 100g
                column.append(FOOD_DATA[food][category] / 100.0)
        columns.append(column)
    A = np.column_stack(columns)
    
    # Compute targets in grams (order matches PRESETS['weights'] dictionary order)
    target_kcalories = PRESETS['targets']["kcalories"]
    target_carbs_g = (target_kcalories * (PRESETS['targets']["carbs_percent"] / 100.0)) / 4.0
    target_protein_g = (target_kcalories * (PRESETS['targets']["protein_percent"] / 100.0)) / 4.0
    target_fat_g = (target_kcalories * (PRESETS['targets']["fat_percent"] / 100.0)) / 9.0
    target_vegetable_g = target_kcalories * (PRESETS['targets']['vegetable_g_calorie_ratio'])

    # Build targets vector in same order as weights
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
    
    targets_dict = {label: round(target, 2) for label, target in PRESETS['targets'].items()}
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