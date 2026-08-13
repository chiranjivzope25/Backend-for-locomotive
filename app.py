import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI(
    title="Locomotive Axle Lock Early Warning System",
    description="Two-Stage Kinematic & Physical Sensor Fusion API",
    version="2.0"
)

# -------------------------------------------------------------
# 1. LOAD ARTIFACTS WITH ERROR HANDLING
# -------------------------------------------------------------
try:
    model_kinematic = joblib.load(r"c:\Users\CHIRNAJIV ZOPE\Downloads\Chirag Industry\Axle lock cases in locomotives\models\axle_lock_xgb.joblib")
    transformer_kinematic = joblib.load(r"c:\Users\CHIRNAJIV ZOPE\Downloads\Chirag Industry\Axle lock cases in locomotives\models\power_transformer.joblib")
    
    model_phy = joblib.load(r"c:\Users\CHIRNAJIV ZOPE\Downloads\Chirag Industry\Axle lock cases in locomotives\models\phy_axle_lock_xgb.joblib")
    transformer_phy = joblib.load(r"c:\Users\CHIRNAJIV ZOPE\Downloads\Chirag Industry\Axle lock cases in locomotives\models\phy_power_transformer.joblib")
    print("All ML models and transformers loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

# -------------------------------------------------------------
# 2. PYDANTIC SCHEMAS FOR DATA VALIDATION
# -------------------------------------------------------------
class KinematicInput(BaseModel):
    v_loco_kmh: float = Field(..., example=80.0)
    axle1_speed_rads: float = Field(..., example=55.1)
    axle2_speed_rads: float = Field(..., example=55.2)
    axle3_speed_rads: float = Field(..., example=55.0)
    axle4_speed_rads: float = Field(..., example=54.8)
    axle1_slip_ratio: float = Field(..., example=0.0)

class PhysicalInput(BaseModel):
    axle1_bearing_temp_c: float = Field(..., example=105.4)
    axle1_vibration_g: float = Field(..., example=3.8)
    axle1_motor_current_amp: float = Field(..., example=520.0)
    
    # ✅ FIXED: Removed the stray ',c' syntax error
    axle2_bearing_temp_c: float = Field(..., example=45.0)
    axle2_vibration_g: float = Field(..., example=0.3)
    axle2_motor_current_amp: float = Field(..., example=300.0)
    
    axle3_bearing_temp_c: float = Field(..., example=46.2)
    axle3_vibration_g: float = Field(..., example=0.35)
    axle3_motor_current_amp: float = Field(..., example=305.0)
    
    axle4_bearing_temp_c: float = Field(..., example=44.8)
    axle4_vibration_g: float = Field(..., example=0.28)
    axle4_motor_current_amp: float = Field(..., example=298.0)

class DualModelRequest(BaseModel):
    data_axel: KinematicInput
    data_phy: PhysicalInput

# -------------------------------------------------------------
# 3. ENDPOINTS
# -------------------------------------------------------------
@app.get("/")
def home():
    return {
        "status": "Online",
        "system": "Locomotive Axle Lock Dual-Model Inference Service"
    }

@app.post("/predict")
def predict(request: DualModelRequest):
    try:
        # Convert incoming JSON payload to DataFrames
        df_kinematic = pd.DataFrame([request.data_axel.dict()])
        df_phy = pd.DataFrame([request.data_phy.dict()])
        
        # 1. Transform Features
        x_scaled_kin = transformer_kinematic.transform(df_kinematic)
        x_scaled_phy = transformer_phy.transform(df_phy)
        
        # 2. Get Predictions & Probabilities
        pred_kin = int(model_kinematic.predict(x_scaled_kin)[0])
        prob_kin = float(model_kinematic.predict_proba(x_scaled_kin)[0][1])
        
        pred_phy = int(model_phy.predict(x_scaled_phy)[0])
        prob_phy = float(model_phy.predict_proba(x_scaled_phy)[0][1])
        
        # ✅ 3. SAFETY RULE OVERRIDE FOR AXLE LOCK
        # If Train is moving (>15 km/h) AND any axle speed drops below 5 rad/s, force Lock Anomaly = 1
        speeds = [
            request.data_axel.axle1_speed_rads,
            request.data_axel.axle2_speed_rads,
            request.data_axel.axle3_speed_rads,
            request.data_axel.axle4_speed_rads
        ]
        if request.data_axel.v_loco_kmh > 15.0 and any(s < 5.0 for s in speeds):
            pred_kin = 1
            prob_kin = max(prob_kin, 0.99)

        # 4. Severity Mapping
        if pred_kin == 1 and pred_phy == 1:
            alert_status = "CRITICAL: AXLE LOCK & MECHANICAL SEIZURE CONFIRMED"
            color = "red"
            risk_level = "HIGH"
        elif pred_phy == 1:
            alert_status = "WARNING: HIGH BEARING TEMP / VIBRATION DETECTED"
            color = "orange"
            risk_level = "MEDIUM"
        elif pred_kin == 1:
            alert_status = "CAUTION: WHEEL SLIP OR KINEMATIC LOCK DETECTED"
            color = "yellow"
            risk_level = "LOW-MEDIUM"
        else:
            alert_status = "SYSTEM NORMAL"
            color = "green"
            risk_level = "NORMAL"
            
        return {
            "overall_status": alert_status,
            "display_color": color,
            "risk_level": risk_level,
            "model_outputs": {
                "kinematic_model": {
                    "prediction": pred_kin,
                    "confidence_score": round(prob_kin, 4)
                },
                "physical_model": {
                    "prediction": pred_phy,
                    "confidence_score": round(prob_phy, 4)
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Inference Error: {str(e)}"
        )