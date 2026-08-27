"""Bridge the trained Keras model to structured agronomic guidance."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.predict import predict_crop_disease  # noqa: E402

KNOWLEDGE: dict[str, dict] = {
    "Tomato___healthy": {
        "severity": "None",
        "symptoms": ["Uniform green foliage", "No lesions, spots, or wilting"],
        "treatment": ["No chemical treatment required", "Continue balanced irrigation and nutrition"],
        "prevention": ["Scout weekly", "Maintain airflow between plants", "Sanitize tools after each field visit"],
    },
    "Tomato___Early_blight": {
        "severity": "High",
        "symptoms": [
            "Concentric brown target-spot lesions on older leaves",
            "Yellowing around leaf spots",
            "Defoliation starting from the base",
        ],
        "treatment": [
            "Remove heavily infected leaves",
            "Apply a labeled chlorothalonil or copper fungicide",
            "Improve nitrogen balance to reduce plant stress",
        ],
        "prevention": ["Mulch soil to reduce splash", "Rotate away from solanaceous crops", "Stake plants for airflow"],
    },
    "Tomato___Late_blight": {
        "severity": "Critical",
        "symptoms": ["Water-soaked lesions", "White fungal growth on leaf undersides", "Rapid collapse in humid weather"],
        "treatment": ["Destroy infected plants", "Apply systemic + protectant fungicide program", "Stop overhead irrigation"],
        "prevention": ["Use certified disease-free transplants", "Avoid dense canopies", "Monitor cool, wet forecasts"],
    },
    "Tomato___Leaf_Mold": {
        "severity": "Moderate",
        "symptoms": ["Pale spots on upper leaf surface", "Olive-green mold on undersides"],
        "treatment": ["Increase ventilation in greenhouses", "Apply sulfur or copper sprays as labeled"],
        "prevention": ["Reduce humidity", "Space plants", "Remove lower leaves"],
    },
    "Tomato___Septoria_leaf_spot": {
        "severity": "High",
        "symptoms": ["Small circular spots with dark margins and gray centers", "Tiny pycnidia in lesion centers"],
        "treatment": ["Defoliate severely spotted leaves", "Rotate protectant fungicides"],
        "prevention": ["Avoid working plants when wet", "Stake and prune", "Three-year crop rotation"],
    },
    "Tomato___Bacterial_spot": {
        "severity": "High",
        "symptoms": ["Dark greasy specks on leaves and fruit", "Shot-hole tearing of foliage"],
        "treatment": ["Copper + mancozeb program during wet weather", "Remove infested debris"],
        "prevention": ["Use pathogen-free seed", "Avoid overhead irrigation", "Sanitize transplant trays"],
    },
    "Tomato___Target_Spot": {
        "severity": "Moderate",
        "symptoms": ["Brown lesions with concentric rings", "Fruit spotting in humid conditions"],
        "treatment": ["Apply strobilurin or SDHI fungicides as labeled", "Remove lower canopy"],
        "prevention": ["Mulch", "Improve airflow", "Avoid late-season nitrogen excess"],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "severity": "Critical",
        "symptoms": ["Upward leaf curling", "Yellowing of leaf margins", "Stunted plants with reduced fruit set"],
        "treatment": ["Rogue infected plants", "Control whiteflies with approved insecticides and oils"],
        "prevention": ["Reflective mulch", "Insect-proof netting for nurseries", "Resistant cultivars"],
    },
    "Tomato___Tomato_mosaic_virus": {
        "severity": "High",
        "symptoms": ["Mosaic mottling", "Leaf distortion", "Uneven fruit ripening"],
        "treatment": ["No chemical cure; remove infected plants", "Disinfect tools with 10% bleach"],
        "prevention": ["Do not smoke near plants", "Use virus-indexed seed", "Wash hands before handling"],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "severity": "Moderate",
        "symptoms": ["Stippled yellow leaves", "Fine webbing on undersides"],
        "treatment": ["Release predatory mites if available", "Apply horticultural oil or miticide as labeled"],
        "prevention": ["Avoid dusty, water-stressed canopies", "Scout undersides weekly"],
    },
    "Potato___healthy": {
        "severity": "None",
        "symptoms": ["Dark green, turgid leaflets with no lesions"],
        "treatment": ["Maintain even soil moisture", "Hill soil to protect tubers"],
        "prevention": ["Certified seed potatoes", "Rotate at least three years"],
    },
    "Potato___Early_blight": {
        "severity": "High",
        "symptoms": ["Bullseye lesions on lower leaves", "Premature defoliation"],
        "treatment": ["Protectant fungicide at first symptoms", "Remove volunteer potatoes"],
        "prevention": ["Adequate potassium", "Avoid overhead irrigation late in the day"],
    },
    "Potato___Late_blight": {
        "severity": "Critical",
        "symptoms": ["Water-soaked lesions", "White sporulation", "Tuber rot"],
        "treatment": ["Destroy outbreaks", "Systemic fungicides during blight-favorable weather"],
        "prevention": ["Blight-resistant varieties", "Kill vines before harvest", "Store only dry, sound tubers"],
    },
    "Corn___healthy": {
        "severity": "None",
        "symptoms": ["Uniform green leaves and intact midribs"],
        "treatment": ["Continue fertility program", "Scout for insects"],
        "prevention": ["Hybrid diversity", "Balanced nitrogen"],
    },
    "Corn___Common_rust": {
        "severity": "Moderate",
        "symptoms": ["Cinnamon-brown pustules on both leaf surfaces"],
        "treatment": ["Foliar fungicide if rust exceeds economic threshold before tasseling"],
        "prevention": ["Resistant hybrids", "Avoid late planting in rust-prone regions"],
    },
    "Corn___Northern_Leaf_Blight": {
        "severity": "High",
        "symptoms": ["Long cigar-shaped gray-green lesions"],
        "treatment": ["Apply fungicide at VT–R1 if lesions are advancing"],
        "prevention": ["Residue management", "Crop rotation", "Ht-gene hybrids"],
    },
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
        "severity": "High",
        "symptoms": ["Rectangular gray lesions limited by veins"],
        "treatment": ["Strobilurin or triazole fungicide at tasseling when pressure is high"],
        "prevention": ["Reduce corn-on-corn residue", "Choose GLS-tolerant hybrids"],
    },
    "Apple___healthy": {
        "severity": "None",
        "symptoms": ["Glossy green leaves without scab lesions"],
        "treatment": ["Standard orchard nutrition and irrigation"],
        "prevention": ["Prune for light penetration", "Sanitation of fallen leaves"],
    },
    "Apple___Apple_scab": {
        "severity": "High",
        "symptoms": ["Olive-black velvety leaf spots", "Russeted fruit lesions"],
        "treatment": ["Protectant fungicides from green tip through petal fall"],
        "prevention": ["Resistant cultivars", "Shred fallen leaves", "Open canopy pruning"],
    },
    "Apple___Black_rot": {
        "severity": "High",
        "symptoms": ["Frog-eye leaf spots", "Mummified fruit"],
        "treatment": ["Prune cankers", "Apply captan or similar labeled products"],
        "prevention": ["Remove mummies", "Avoid wounding trunks"],
    },
    "Apple___Cedar_apple_rust": {
        "severity": "Moderate",
        "symptoms": ["Bright orange leaf spots", "Aecia on leaf undersides"],
        "treatment": ["Fungicide at pink through petal fall"],
        "prevention": ["Remove nearby juniper galls where practical", "Plant rust-resistant apples"],
    },
    "Grape___healthy": {
        "severity": "None",
        "symptoms": ["Intact green canopy and bloom"],
        "treatment": ["Canopy management only"],
        "prevention": ["Shoot thinning", "Leaf pulling around clusters"],
    },
    "Grape___Black_rot": {
        "severity": "High",
        "symptoms": ["Tan leaf spots with dark borders", "Shriveled black berries"],
        "treatment": ["Mancozeb or captan from bloom to veraison as labeled"],
        "prevention": ["Remove mummies", "Open the fruiting zone"],
    },
    "Grape___Esca_(Black_Measles)": {
        "severity": "High",
        "symptoms": ["Tiger-stripe leaf patterns", "Berry spotting", "Vine decline"],
        "treatment": ["Remove symptomatic wood", "Protect pruning wounds"],
        "prevention": ["Delay pruning in wet weather", "Avoid large cuts near the trunk"],
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "severity": "Moderate",
        "symptoms": ["Angular brown leaf spots", "Premature defoliation"],
        "treatment": ["Protectant sprays during prolonged leaf wetness"],
        "prevention": ["Improve airflow", "Avoid excess vigor"],
    },
    "Pepper,_bell___healthy": {
        "severity": "None",
        "symptoms": ["Thick, dark green leaves"],
        "treatment": ["Maintain calcium to prevent blossom-end rot"],
        "prevention": ["Even watering", "Mulch"],
    },
    "Pepper,_bell___Bacterial_spot": {
        "severity": "High",
        "symptoms": ["Water-soaked leaf spots", "Raised fruit scabs"],
        "treatment": ["Copper sprays during wet periods", "Rogue severely infected plants"],
        "prevention": ["Hot-water seed treatment", "Drip irrigation"],
    },
    "Cherry___healthy": {
        "severity": "None",
        "symptoms": ["Uniform foliage without shot-hole symptoms"],
        "treatment": ["Standard orchard care"],
        "prevention": ["Prune after harvest in dry weather"],
    },
    "Cherry___Powdery_mildew": {
        "severity": "Moderate",
        "symptoms": ["White powdery growth on young leaves", "Leaf curling"],
        "treatment": ["Sulfur or potassium bicarbonate as labeled"],
        "prevention": ["Open canopy", "Avoid excessive nitrogen"],
    },
    "Strawberry___healthy": {
        "severity": "None",
        "symptoms": ["Glossy trifoliate leaves"],
        "treatment": ["Keep crowns dry"],
        "prevention": ["Renew beds every 3–4 years"],
    },
    "Strawberry___Leaf_scorch": {
        "severity": "Moderate",
        "symptoms": ["Purple spots that coalesce into scorched areas"],
        "treatment": ["Remove old leaves after harvest", "Apply labeled fungicide if severe"],
        "prevention": ["Wide plant spacing", "Avoid overhead irrigation at dusk"],
    },
    "Squash___Powdery_mildew": {
        "severity": "Moderate",
        "symptoms": ["White powder on upper leaf surface", "Premature senescence"],
        "treatment": ["Sulfur, potassium bicarbonate, or labeled systemic fungicide"],
        "prevention": ["Resistant varieties", "Do not crowd plants"],
    },
    "Soybean___healthy": {
        "severity": "None",
        "symptoms": ["Uniform trifoliate color"],
        "treatment": ["Maintain potassium fertility"],
        "prevention": ["Rotate with non-legumes"],
    },
}


def parse_class_label(label: str) -> tuple[str, str, bool]:
    if "___" in label:
        crop, disease = label.split("___", 1)
    elif "_" in label:
        crop, disease = label.split("_", 1)
    else:
        crop, disease = "Unknown crop", label
    crop = crop.replace(",", " ").replace("_", " ").strip().title()
    disease_key = disease.strip()
    is_healthy = disease_key.lower() == "healthy"
    disease_name = "Healthy" if is_healthy else disease_key.replace("_", " ").strip().title()
    return crop, disease_name, is_healthy


def _fallback_guidance(disease_name: str, is_healthy: bool) -> dict:
    if is_healthy:
        return {
            "severity": "None",
            "symptoms": ["No disease symptoms detected on the submitted leaf"],
            "treatment": ["Continue current crop management"],
            "prevention": ["Keep scouting", "Irrigate at the soil line", "Rotate crops"],
        }
    return {
        "severity": "Moderate",
        "symptoms": [f"Visual patterns consistent with {disease_name}"],
        "treatment": [
            "Confirm diagnosis with a local extension officer",
            "Remove badly affected leaves",
            "Apply a labeled crop-specific protectant if weather favors spread",
        ],
        "prevention": [
            "Improve airflow and drainage",
            "Avoid working plants when foliage is wet",
            "Sanitize tools and rotate away from the same family next season",
        ],
    }


def analyze_image(image_path: str) -> dict:
    raw = predict_crop_disease(image_path)
    top1 = raw["top1"]
    label = top1["label"]
    crop, disease, is_healthy = parse_class_label(label)
    guide = KNOWLEDGE.get(label) or _fallback_guidance(disease, is_healthy)
    return {
        "crop": crop,
        "disease": disease,
        "class_label": label,
        "is_healthy": is_healthy,
        "severity": guide["severity"],
        "confidence": top1["confidence"],
        "top3": raw["top3"],
        "symptoms": guide["symptoms"],
        "treatment": guide["treatment"],
        "prevention": guide["prevention"],
    }
