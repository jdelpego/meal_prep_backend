# Mealy 🍽️

**Smart Meal Optimization for Balanced Nutrition**

Mealy is an intelligent meal planning API that uses advanced mathematical optimization to create perfectly balanced meals based on your calorie and macronutrient goals.

## What It Does

Mealy solves a complex nutritional puzzle: given a list of available foods, it calculates the exact portions needed to hit your target calories, macronutrients (carbs, protein, fat), and micronutrients—all while ensuring realistic, practical serving sizes.

### How It Works

1. **Select Your Foods**: Choose from a diverse database of whole foods including proteins (chicken, salmon, tofu), carbs (rice, oats, quinoa), healthy fats (avocado, olive oil, almonds), and vegetables
2. **Set Your Goals**: Specify your target calories and macro percentages (or use smart defaults)
3. **Get Optimized Meals**: Receive precise portion sizes that balance all your nutritional targets simultaneously

The optimization engine uses **weighted least squares** with intelligent constraints to find the ideal combination of foods that:
- Hits your calorie target accurately
- Balances macronutrients according to your percentages (e.g., 40% carbs, 30% protein, 30% fat)
- Includes adequate vegetables for micronutrients
- Provides realistic portions (10g–400g per ingredient)
- Considers 13+ essential micronutrients for longevity

## Key Benefits

### 🎯 **Precision Nutrition**
- Achieve macro targets within 1-2% accuracy
- No more guessing or manual calculations
- Mathematical optimization ensures the best possible balance

### 🥗 **Prevents Extreme Results**
- Smart bounds prevent unrealistic portions (no more 878g of carrots!)
- Normalized weighting ensures fair balancing across all nutrients
- Minimum 10g per ingredient prevents trace amounts
- Maximum 400g per ingredient keeps portions realistic

### 🧮 **Science-Based Algorithm**
- Uses normalized weights (`w/t²`) to equalize percentage errors
- Treats a 1% error in fat the same as a 1% error in carbs
- Considers both macronutrients and micronutrients simultaneously
- Decoupled vegetable constraint prevents circular dependencies

### 🔧 **Flexible & Customizable**
- Set custom calorie targets (default: 700 kcal per meal)
- Adjust macro percentages to your diet (e.g., 50/25/25 for low-fat)
- Choose from diverse food options including plant-based proteins
- Works with any combination of available foods

### ⚡ **Fast & Reliable**
- RESTful API built with FastAPI
- Instant optimization results
- CORS-enabled for easy frontend integration
- Consistent, reproducible results

## Technical Highlights

### Optimization Approach
The app uses **scipy's least squares solver** (`lsq_linear`) with:
- **Matrix A**: Nutritional content per 100g for each food
- **Vector b**: Your target nutrient values
- **Weight normalization**: `weight / (target²)` for fair percentage-based penalties
- **Bounds**: 10–400g per ingredient for practical portions

### Nutrient Tracking
Each meal is optimized across 17+ nutritional parameters:
- **Macros**: Calories, carbs, protein, fat, fiber
- **Minerals**: Magnesium, potassium, selenium, zinc
- **Vitamins**: D, K2, folate, B12, C, E
- **Other**: Omega-3 EPA/DHA, choline

### Food Database
Curated whole-food database with per-100g nutrition data sourced from USDA:
- **Proteins**: Chicken, salmon, eggs, beef, tofu
- **Carbs**: White rice, sweet potato, oats, quinoa
- **Fats**: Avocado, almonds, olive oil
- **Vegetables**: Broccoli, spinach, carrots, kale, bell peppers, cauliflower, tomato
- **Fruits**: Banana, blueberries

## API Usage

### Endpoint: `POST /optimize_meal_prep`

**Request Body**:
```json
{
  "foods": ["chicken", "white_rice", "broccoli", "avocado"],
  "kcalories": 700,
  "carbs_percent": 40,
  "protein_percent": 30,
  "fat_percent": 30
}
```

**Response**:
```json
{
  "portions": {
    "chicken": 120.5,
    "white_rice": 85.3,
    "broccoli": 150.0,
    "avocado": 45.2
  },
  "totals": {
    "kcalories": 698.4,
    "carbs_g": 69.8,
    "protein_g": 52.3,
    "fat_g": 23.1,
    "fiber_g": 12.4,
    ...
  }
}
```

### Endpoint: `POST /recommend_ingredients`

Get food recommendations based on your current selection to improve nutritional balance.

## Running Mealy

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
./run.sh
```

The API will be available at `http://localhost:8001`

### Requirements
- Python 3.11+
- FastAPI
- NumPy
- SciPy
- Scikit-learn
- Pydantic

## Use Cases

- **Meal Prep Planning**: Calculate exact portions for weekly meal prep
- **Macro Tracking**: Hit specific macro targets for fitness goals
- **Dietary Balance**: Ensure adequate micronutrient intake
- **Recipe Development**: Create nutritionally optimized meal combinations
- **Nutrition Education**: Understand how foods combine to meet nutritional needs

## Future Enhancements

- Additional food database entries
- Meal planning for multiple meals per day
- Cost optimization alongside nutrition
- Allergen and dietary restriction filtering
- Meal variety scoring to prevent monotony

---

**Built with ❤️ for optimal nutrition**

*Mealy - Because balanced nutrition shouldn't require a PhD in mathematics*
