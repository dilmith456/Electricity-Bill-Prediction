from src.pipelines.retraining_pipeline import append_new_data
import pandas as pd
import joblib

# -------------------------
# Load Model & Preprocessor
# -------------------------
model = joblib.load("artifacts/model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.joblib")


# -------------------------
# Sri Lankan Bill Calculation (DYNAMIC)
# -------------------------
def calculate_bill(units):
    units = float(units)

    tariff_df = pd.read_csv("notebook/tariff_rates.csv")

    total_energy = 0
    fixed_charge = 0

    for _, row in tariff_df.iterrows():
        min_u = row["min_units"]
        max_u = row["max_units"]

        if units > min_u:
            consumed = min(units, max_u) - min_u
            total_energy += consumed * row["unit_price"]
            fixed_charge = row["fixed_charge"]

    subtotal = total_energy + fixed_charge

    total = subtotal + (subtotal * 0.025)

    return round(total, 2)


# -------------------------
# User Input Function
# -------------------------
def get_user_input():

    print("\n--- Enter Household Details ---")

    household_size = int(input("Household Size: "))
    past_units = float(input("Past Month Units: "))
    ac_hours = float(input("This Month AC Usage (In Hours): "))
    fan_hours = float(input("This Month Fan Usage (In Hours): "))
    tv_hours = float(input("TV Usage (hours/day): "))                 
    fridge_hours = float(input("Fridge Usage (hours/day ~24): "))     

    location_type = input("Location Type (Urban/Rural): ").strip().capitalize()
    income_level = input("Income Level (Low/Middle/High): ").strip().capitalize()

    climate_zone = input("Climate Zone (DryHot/Cool/WetWarm): ").strip()

    # Normalize climate_zone
    if climate_zone.lower() == "dryhot":
        climate_zone = "DryHot"
    elif climate_zone.lower() == "wetwarm":
        climate_zone = "WetWarm"
    elif climate_zone.lower() == "cool":
        climate_zone = "Cool"

    data = pd.DataFrame([{
        "household_size": household_size,
        "inflation_rate": 5,
        "electricity_tariff_rate": 50,
        "past_month_units": past_units,
        "ac_hours": ac_hours,
        "fan_hours": fan_hours,
        "tv_hours": tv_hours,             
        "fridge_hours": fridge_hours,     
        "location_type": location_type,
        "climate_zone": climate_zone,
        "income_level": income_level
    }])

    return data


# -------------------------
# Prediction Function
# -------------------------
def predict():

    try:
        user_data = get_user_input()

        # Transform
        transformed_data = preprocessor.transform(user_data)

        # Base prediction
        base_prediction = float(model.predict(transformed_data)[0])

        # Extract inputs
        ac = user_data["ac_hours"].values[0]
        fan = user_data["fan_hours"].values[0]
        tv = user_data["tv_hours"].values[0]             
        fridge = user_data["fridge_hours"].values[0]     
        past_units = user_data["past_month_units"].values[0]

        # -------------------------
        # FINAL PREDICTION (UPDATED)
        # -------------------------
        predicted_units = (
            (base_prediction * 0.7) +
            (past_units * 0.2) +
            (ac * 2.6) +
            (fan * 1.3) +
            (tv * 0.15) +          
            (fridge * 0.1)         
        )

        predicted_units = max(20, min(predicted_units, 300))
        predicted_units = round(predicted_units, 2)

        # Bill calculation
        bill = calculate_bill(predicted_units)

        # Output
        print("\n--- RESULT ---")
        print(f"Predicted Next Month Units: {predicted_units} units")
        print(f"Estimated Electricity Bill (LKR): {bill}")

        # -------------------------
        # SAVE NEW DATA
        # -------------------------
        new_record = {
            "household_id": 999,
            "date": pd.Timestamp.now().date(),
            "location_type": user_data["location_type"].values[0],
            "climate_zone": user_data["climate_zone"].values[0],
            "household_size": user_data["household_size"].values[0],
            "income_level": user_data["income_level"].values[0],
            "inflation_rate": user_data["inflation_rate"].values[0],
            "electricity_tariff_rate": user_data["electricity_tariff_rate"].values[0],
            "past_month_units": past_units,
            "ac_hours": ac,
            "fan_hours": fan,
            "tv_hours": tv,                 
            "fridge_hours": fridge,         
            "next_month_units": predicted_units
        }

        append_new_data(new_record)

    except Exception as e:
        print("\nError:", e)


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    predict()