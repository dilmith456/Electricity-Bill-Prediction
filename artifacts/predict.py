import pandas as pd
import joblib

# -------------------------
# Load Model & Preprocessor
# -------------------------
model = joblib.load("../artifacts/model.pkl")
preprocessor = joblib.load("../artifacts/preprocessor.joblib")


# -------------------------
# Sri Lankan Bill Calculation
# -------------------------
def calculate_bill(units):
    units = float(units)

    if units <= 60:
        energy = units * 25
        fixed = 300

    elif units <= 90:
        energy = (60 * 25) + (units - 60) * 30
        fixed = 400

    elif units <= 120:
        energy = (60 * 25) + (30 * 30) + (units - 90) * 50
        fixed = 1000

    elif units <= 180:
        energy = (60 * 25) + (30 * 30) + (30 * 50) + (units - 120) * 75
        fixed = 1500

    else:
        energy = units * 75
        fixed = 2000

    subtotal = energy + fixed

    # SSCL TAX (2.5%)
    sscl = subtotal * 0.025

    total = subtotal + sscl

    return round(total, 2)


# -------------------------
# User Input Function
# -------------------------
def get_user_input():

    print("\n--- Enter Household Details ---")

    household_size = int(input("Household Size: "))
    past_units = float(input("Past Month Units: "))
    ac_hours = float(input("AC Usage (hours/day): "))
    fan_hours = float(input("Fan Usage (hours/day): "))

    location_type = input("Location Type (Urban/Rural): ").strip().capitalize()
    income_level = input("Income Level (Low/Middle/High): ").strip().capitalize()

    climate_zone = input("Climate Zone (DryHot/Cool/WetWarm): ").strip()

    # Normalize climate_zone properly
    if climate_zone.lower() == "dryhot":
        climate_zone = "DryHot"
    elif climate_zone.lower() == "wetwarm":
        climate_zone = "WetWarm"
    elif climate_zone.lower() == "cool":
        climate_zone = "Cool"

    # Fixed system values
    inflation_rate = 5
    electricity_tariff_rate = 50

    data = pd.DataFrame([{
        "household_size": household_size,
        "inflation_rate": inflation_rate,
        "electricity_tariff_rate": electricity_tariff_rate,
        "past_month_units": past_units,
        "ac_hours": ac_hours,
        "fan_hours": fan_hours,
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

        # Transform input
        transformed_data = preprocessor.transform(user_data)

        # Base model prediction
        prediction = model.predict(transformed_data)
        base_prediction = float(prediction[0])

        # Extract inputs
        ac_hours = user_data["ac_hours"].values[0]
        fan_hours = user_data["fan_hours"].values[0]
        past_units = user_data["past_month_units"].values[0]

        # -------------------------
        # ✅ FINAL FIX: Reweighted Prediction
        # -------------------------
        predicted_units = (
            (base_prediction * 0.7) +   # reduce model dominance
            (past_units * 0.2) +        # stabilize with past usage
            (ac_hours * 3.0) +          # strong AC impact
            (fan_hours * 1.5)           # moderate fan impact
        )

        # Clamp realistic range
        predicted_units = max(20, min(predicted_units, 300))
        predicted_units = round(predicted_units, 2)

        # Calculate bill
        bill = calculate_bill(predicted_units)

        # Output
        print("\n--- RESULT ---")
        print(f"Predicted Next Month Units: {predicted_units} units")
        print(f"Estimated Electricity Bill (LKR): {bill}")

    except Exception as e:
        print("\nError:", e)


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    predict()