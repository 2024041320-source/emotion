
from pathlib import Path
from contextlib import asynccontextmanager
import pickle
import re
import traceback

import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "Artifacts" / "BiGRU_Model.keras"
tokenizer_path = BASE_DIR / "Artifacts" / "tokenizer.pkl"
static_path = BASE_DIR / "static"

max_sequence_length = 50

emotion_labels = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]

EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


# ==========================================
# 2. TEXT PREPROCESSING
# ==========================================

def preprocess_text(text: str) -> str:

    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# 3. REQUEST AND RESPONSE SCHEMAS
# ==========================================

class TextInput(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={
            "example": "I feel so happy and excited"
        }
    )


class PredictionResponse(BaseModel):

    text: str
    predicted_emotion: str
    confidence: float
    all_probabilites: dict[str, float]


class HealthResponse(BaseModel):

    status: str
    model_loaded: bool


# ==========================================
# 4. MODEL AND TOKENIZER STORAGE
# ==========================================

dl_model = {}


# ==========================================
# 5. LIFESPAN MANAGEMENT
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    try:

        print("Starting Emotion Detection API...", flush=True)

        # Check whether required files exist
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found: {tokenizer_path}"
            )

        print("Loading TensorFlow BiGRU model...", flush=True)

        dl_model["BiGRU"] = load_model(str(model_path))

        print("Loading tokenizer...", flush=True)

        with open(tokenizer_path, "rb") as file:
            dl_model["Tokenizer"] = pickle.load(file)

        print(
            "Model and tokenizer loaded successfully!",
            flush=True
        )

        yield

    except Exception:

        print(
            "ERROR DURING APPLICATION STARTUP:",
            flush=True
        )

        traceback.print_exc()

        raise

    finally:

        dl_model.clear()

        print("Application shutdown completed.", flush=True)


# ==========================================
# 6. CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Emotion Detection API",
    description="Emotion Detection using BiGRU and TensorFlow",
    version="1.0.0",
    lifespan=lifespan
)


# ==========================================
# 7. ENABLE CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 8. MOUNT STATIC FILES
# ==========================================

app.mount(
    "/static",
    StaticFiles(directory=str(static_path)),
    name="static"
)


# ==========================================
# 9. API ENDPOINTS
# ==========================================

# Homepage

@app.get("/", include_in_schema=False)
def server_ui():

    return FileResponse(
        static_path / "index.html"
    )


# Health Check

@app.get("/health", response_model=HealthResponse)
def health_check():

    model_loaded = (
        "BiGRU" in dl_model
        and "Tokenizer" in dl_model
    )

    return HealthResponse(
        status="Server is running",
        model_loaded=model_loaded
    )


# ==========================================
# 10. EMOTION PREDICTION
# ==========================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_emotion(text_input: TextInput):

    # Get loaded model and tokenizer

    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer_model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again later."
        )

    # Step 1: Preprocess input text

    cleaned_text = preprocess_text(text_input.text)

    if not cleaned_text:

        raise HTTPException(
            status_code=400,
            detail="Please enter valid text."
        )

    # Step 2: Convert text into sequences

    tokenized_text = tokenizer_model.texts_to_sequences(
        [cleaned_text]
    )

    # Step 3: Padding

    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    # Step 4: Model prediction

    probabilities = BiGRU_model.predict(
        padded_sequence,
        verbose=0
    )[0]

    # Step 5: Find highest probability emotion

    top_emotion_index = int(
        np.argmax(probabilities)
    )

    predicted_emotion = emotion_labels[
        top_emotion_index
    ]

    # Step 6: Calculate all emotion probabilities

    all_probabilites = {
        label: float(prob)
        for prob, label in zip(
            probabilities,
            emotion_labels
        )
    }

    # Step 7: Return prediction response

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=predicted_emotion,
        confidence=float(
            probabilities[top_emotion_index]
        ),
        all_probabilites=all_probabilites
    )
