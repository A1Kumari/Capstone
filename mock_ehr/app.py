from fastapi import FastAPI

mock_ehr = FastAPI(title="Mock EHR API")

DB = {
    "P1008": {
        "patient": {
            "patient_id"        : "P1008",
            "name"              : "Thomas Wright",
            "primary_dx"        : "Type 2 Diabetes Mellitus",
            "discharge_approved": True,
        },
        "med_orders": {
            "med_orders"      : ["Metformin", "Lisinopril", "Atorvastatin", "Insulin"],
            "counseling_notes": ["Insulin"],
        },
        "labs": {
            "labs": [
                {"name": "HbA1c",       "value": 9.2,  "status": "abnormal"},
                {"name": "Creatinine",  "value": 1.1,  "status": "normal"},
                {"name": "Blood Sugar", "value": 210.0, "status": "abnormal"},
            ]
        },
        "allergies": {
            "allergies": ["Penicillin"]
        },
        "careplan": {
            "follow_up_required": True,
            "details"           : "Endocrinology clinic in 3 months",
        },
    },
    "P1019": {
        "patient": {
            "patient_id"        : "P1019",
            "name"              : "Sarah Connor",
            "primary_dx"        : "Hypertension",
            "discharge_approved": True,
        },
        "med_orders": {
            "med_orders"      : ["Lisinopril", "Amlodipine"],
            "counseling_notes": [],
        },
        "labs": {
            "labs": [
                {"name": "Sodium",      "value": 138.0, "status": "normal"},
                {"name": "Potassium",   "value": 4.1,   "status": "normal"},
            ]
        },
        "allergies": {
            "allergies": ["Sulfa", "Lisinopril"]
        },
        "careplan": {
            "follow_up_required": True,
            "details"           : "Check blood pressure in 2 weeks",
        },
    },
}

_EMPTY = {
    "patient"    : {"patient_id": "", "primary_dx": "Unknown", "discharge_approved": False},
    "med_orders" : {"med_orders": [], "counseling_notes": []},
    "labs"       : {"labs": []},
    "allergies"  : {"allergies": []},
    "careplan"   : {"follow_up_required": False},
}


@mock_ehr.get("/ehr/patient/{patient_id}")
def get_patient(patient_id: str):
    return DB.get(patient_id, {}).get("patient", _EMPTY["patient"])

@mock_ehr.get("/ehr/med_orders/{patient_id}")
def get_med_orders(patient_id: str):
    return DB.get(patient_id, {}).get("med_orders", _EMPTY["med_orders"])

@mock_ehr.get("/ehr/labs/{patient_id}")
def get_labs(patient_id: str):
    return DB.get(patient_id, {}).get("labs", _EMPTY["labs"])

@mock_ehr.get("/ehr/allergies/{patient_id}")
def get_allergies(patient_id: str):
    return DB.get(patient_id, {}).get("allergies", _EMPTY["allergies"])

@mock_ehr.get("/ehr/careplan/{patient_id}")
def get_careplan(patient_id: str):
    return DB.get(patient_id, {}).get("careplan", _EMPTY["careplan"])
