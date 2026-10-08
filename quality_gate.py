
import json
import sys

# Minimum acceptable accuracy
MINIMUM_ACCURACY = 0.85

print("Reading model evaluation metrics...")

# Read metrics.json
with open("metrics.json", "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

print("Model Accuracy :", round(accuracy, 4))
print("Required Accuracy:", MINIMUM_ACCURACY)

# Quality Gate
if accuracy < MINIMUM_ACCURACY:
    print("QUALITY GATE FAILED")
    print("Model performance is below the required threshold.")
    sys.exit(1)

print("QUALITY GATE PASSED")
print("Model performance satisfies the required threshold.")
sys.exit(0)
