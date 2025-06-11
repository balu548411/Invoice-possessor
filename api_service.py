import os
import io
import uuid
import json
import uvicorn
import tempfile
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path

from src.config import MODEL_DIR, INFERENCE_CONFIG
from src.inference.predictor import DocumentPredictor, format_results_as_json


# Initialize FastAPI app
app = FastAPI(
    title="Document Parsing API",
    description="API for intelligent document parsing",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL_PATH = os.path.join(MODEL_DIR, "model_best.pth")
TEMP_DIR = Path(tempfile.gettempdir()) / "document_parser"
TEMP_DIR.mkdir(exist_ok=True)

# Initialize predictor (lazy loading)
predictor = None


def get_predictor():
    """Get or initialize the document predictor."""
    global predictor
    if predictor is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=500, 
                detail=f"Model weights not found at {MODEL_PATH}"
            )
        predictor = DocumentPredictor(
            MODEL_PATH, 
            confidence_threshold=INFERENCE_CONFIG['confidence_threshold']
        )
    return predictor


class PredictionResult(BaseModel):
    """Response model for predictions."""
    data: Dict[str, Any]
    meta: Dict[str, Any]


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Document Parsing API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if model can be loaded
        predictor = get_predictor()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/parse", response_model=PredictionResult)
async def parse_document(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Form(None)
):
    """
    Parse a document image and return structured data.
    
    Args:
        file: Uploaded document image file
        confidence_threshold: Optional confidence threshold for predictions
    
    Returns:
        Structured data extracted from the document
    """
    # Get predictor instance
    predictor = get_predictor()
    
    # Override confidence threshold if provided
    if confidence_threshold is not None:
        predictor.confidence_threshold = confidence_threshold
    
    try:
        # Read uploaded file
        content = await file.read()
        
        # Save to temporary file
        temp_path = TEMP_DIR / f"{uuid.uuid4()}.jpg"
        with open(temp_path, "wb") as temp_file:
            temp_file.write(content)
        
        # Run prediction
        results = predictor.predict(str(temp_path))
        
        # Format results as JSON
        output = format_results_as_json(results)
        
        # Clean up
        os.unlink(temp_path)
        
        return output
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize")
async def visualize_document(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Form(None)
):
    """
    Visualize document parsing results on the input image.
    
    Args:
        file: Uploaded document image file
        confidence_threshold: Optional confidence threshold for predictions
    
    Returns:
        Image with visualized predictions
    """
    # Get predictor instance
    predictor = get_predictor()
    
    # Override confidence threshold if provided
    if confidence_threshold is not None:
        predictor.confidence_threshold = confidence_threshold
    
    try:
        # Read uploaded file
        content = await file.read()
        
        # Convert to numpy array
        image = Image.open(io.BytesIO(content))
        image = np.array(image)
        
        # Run prediction and visualization
        results = predictor.predict(image)
        visualized = predictor.visualize_prediction(image, results)
        
        # Save to temporary file
        output_path = TEMP_DIR / f"{uuid.uuid4()}_viz.jpg"
        Image.fromarray(visualized).save(output_path)
        
        # Return the visualization
        return FileResponse(
            output_path,
            media_type="image/jpeg",
            filename="document_visualization.jpg"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
async def model_info():
    """Get model information."""
    # Get predictor instance
    predictor = get_predictor()
    
    # Return model info
    return {
        "model_path": MODEL_PATH,
        "confidence_threshold": predictor.confidence_threshold,
        "entity_classes": list(predictor.id_to_class.values()),
        "device": str(predictor.device)
    }


if __name__ == "__main__":
    # Run API server
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True) 