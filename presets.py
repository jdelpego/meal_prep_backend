PRESETS = {
    "targets": {
        "kcalories": 700,
        "carbs_percent": 40,
        "protein_percent": 30,
        "fat_percent": 30,
        "vegetable_g_calorie_ratio": 0.20,  # 140g vegetables for 700 kcal meal (20% ratio)
    },
    
    "daily_values": {
        "kcalories":2000,
        "micronutrients": {
            "fiber_g": 25,
            "magnesium_mg": 350,
            "potassium_mg": 4700,
            "selenium_ug": 55,
            "zinc_mg": 10,
            "vitamin_d_ug": 25,
            "vitamin_k2_ug": 100,
            "folate_ug": 400,
            "vitamin_b12_ug": 2.4,
            "omega3_epa_dha_g": 1.5,
            "vitamin_c_mg": 90,
            "vitamin_e_mg": 15,
            "choline_mg": 400,
        }
    },
    "weights": {
        # Macronutrients (highest priority - these define the meal structure)
        "kcalories": 150,      # Must hit calorie target
        "carbs_g": 150,        # Equal weight for all macros (normalized by target size in code)
        "protein_g": 150,      # Equal weight for all macros (normalized by target size in code)
        "fat_g": 150,          # Equal weight for all macros (normalized by target size in code)
        
        # Vegetables (medium priority - important for health)
        "vegetable_g": 20,     # Allow flexibility, but still important
        
        # Micronutrients (lower priority - nice to have, but don't break macros)
        # High-priority micros (harder to get from typical foods)
        "omega3_epa_dha_g": 5, # Hard to get without fish
        "vitamin_d_ug": 5,     # Hard to get from food
        "magnesium_mg": 4,     # Important for many functions
        "potassium_mg": 4,     # Important electrolyte
        
        # Medium-priority micros
        "fiber_g": 3,          # Usually hit naturally with vegetables
        "selenium_ug": 3,      # Usually adequate
        "zinc_mg": 3,          # Usually adequate
        "folate_ug": 3,        # Abundant in vegetables
        "vitamin_b12_ug": 3,   # Easy with animal products
        "vitamin_c_mg": 3,     # Abundant in vegetables
        
        # Lower-priority micros (often naturally met or less critical)
        "vitamin_k2_ug": 1,    # Rare in foods, usually low
        "vitamin_e_mg": 2,     # Usually adequate
        "choline_mg": 2,       # Usually adequate with eggs
    }
}