from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, Request
from sqlalchemy import create_engine, Column, String, Float, DateTime, ForeignKey, LargeBinary, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from typing import List, Optional
import uuid
import os
import time
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi.responses import Response,JSONResponse
import json
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from functools import wraps
from tempfile import NamedTemporaryFile
import shutil
import requests

load_dotenv()

app = FastAPI()

PORT = int(os.getenv("PORT", 10000))
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://dashboard.deadcow.xyz", "https://deadcow.xyz"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    max_age=3600,
)


def get_db_engine():
    DATABASE_URL = os.getenv("DATABASE_URL")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    # Configure connection arguments based on environment
    if ENVIRONMENT == "production":
        connect_args = {
            "sslmode": "require",
            "connect_timeout": 30
        }
    else:
        # Local development - no SSL
        connect_args = {
            "connect_timeout": 30
        }
    
    # Create engine with appropriate config
    return create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_size=5,
        max_overflow=10,
        echo=True
    )

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = get_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class NodeRegistration(BaseModel):
    device_uuid: str  # Unique identifier from the device
    device_name: str | None = None
    device_model: str | None = None

class NodeInfo(BaseModel):
    node_id: str
    device_uuid: str
    device_name: Optional[str] = None
    device_model: Optional[str] = None
    status: str
    current_model_id: Optional[str] = None
    last_heartbeat: datetime
    created_at: datetime

    class Config:
        from_attributes = True

class ModelInfo(BaseModel):
    model_id: str
    name: str
    filename: str
    status: str
    node_id: Optional[str] = None
    owner_address: str
    created_at: datetime

    class Config:
        from_attributes = True


class NodesResponse(BaseModel):
    nodes: List[NodeInfo]

    class Config:
        from_attributes = True


class InferenceRequest(Base):
    __tablename__ = "inference_requests"
    
    request_id = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey("models.model_id"))
    node_id = Column(String, ForeignKey("nodes.node_id"))
    input_data = Column(LargeBinary)
    status = Column(String)  # pending, processing, completed, failed
    result = Column(LargeBinary, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

class TextInferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    temperature: float = Field(default=0.7, ge=0, le=1)
    max_tokens: int = Field(default=100, ge=1, le=1000)

# Database Models
class Node(Base):
    __tablename__ = "nodes"
    
    node_id = Column(String, primary_key=True)
    device_uuid = Column(String, unique=True)  # Add unique constraint
    device_name = Column(String, nullable=True)
    device_model = Column(String, nullable=True)
    status = Column(String)  # available, busy
    current_model_id = Column(String, nullable=True)
    last_heartbeat = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelResponse(BaseModel):
    model_id: str
    name: str
    filename: str
    status: str
    node_id: str | None
    owner_address: str
    created_at: datetime

class ModelUploadResponse(BaseModel):
    model_id: str
    name: str
    node_id: str
    status: str
    owner_address: str

class NoModelResponse(BaseModel):
    status: str = "no_model"

class ModelReadyResponse(BaseModel):
    status: str = "model_ready"
    model_id: str
    download_url: str

class Model(Base):
    __tablename__ = "models"
    
    model_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    filename = Column(String)
    status = Column(String)  # uploaded, deployed, failed
    node_id = Column(String, ForeignKey("nodes.node_id"), nullable=True)
    owner_address = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    file_size = Column(Float, nullable=True)

Base.metadata.create_all(bind=engine)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )


# Dependency
def get_db():
    db = SessionLocal()
    try:
        # Test the connection
        db.execute(text("SELECT 1"))
        yield db
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

def handle_db_error(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            print(f"Database error in {func.__name__}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
    return wrapper

def construct_node_url(node_id: str) -> str:
    """Construct the URL for a node using its ID."""
    base_domain = "https://api.deadcow.xyz"  # Use HTTPS in production
    return f"{base_domain}/nodes/{node_id}/download_model"

# Pydantic models
class NodeCreate(BaseModel):
    node_id: str
    status: str
    current_model_id: str | None = None

class NodeResponse(BaseModel):
    node_id: str
    device_uuid: str
    device_name: Optional[str] = None
    device_model: Optional[str] = None
    status: str
    current_model_id: Optional[str] = None
    last_heartbeat: datetime
    created_at: datetime

    class Config:
        from_attributes = True

class NodeListResponse(BaseModel):
    nodes: List[NodeResponse]

    class Config:
        from_attributes = True

from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, HTTPException

class NodeRegistration(BaseModel):
    device_uuid: str
    device_name: Optional[str] = None
    device_model: Optional[str] = None

class RegistrationResponse(BaseModel):
    node_id: str
    status: str
    device_uuid: str
    device_name: Optional[str] = None
    device_model: Optional[str] = None
    node_url: Optional[str] = None


@app.post("/nodes/register", response_model=RegistrationResponse)
@handle_db_error
async def register_node(registration: NodeRegistration, db: Session = Depends(get_db)):
    """Register a new node"""
    try:
        print(f"Checking for existing device: {registration.device_uuid}")
        
        # First try to find existing node
        existing_node = db.query(Node)\
            .filter(Node.device_uuid == registration.device_uuid)\
            .first()
        
        if existing_node:
            print(f"Found existing node: {existing_node.node_id}")
            # Update existing node
            existing_node.last_heartbeat = datetime.utcnow()
            existing_node.status = "available"
            existing_node.device_name = registration.device_name
            existing_node.device_model = registration.device_model
            existing_node.node_url = construct_node_url(existing_node.node_id)

            db.commit()
            
            return RegistrationResponse(
                node_id=existing_node.node_id,
                status="already_registered",
                device_uuid=existing_node.device_uuid,
                device_name=existing_node.device_name,
                device_model=existing_node.device_model,
                node_url=registration.node_url
            )
        
        # If no existing node, create new one
        node_id = str(uuid.uuid4())
        print(f"Creating new node: {node_id}")
        
        new_node = Node(
            node_id=node_id,
            device_uuid=registration.device_uuid,
            device_name=registration.device_name,
            device_model=registration.device_model,
            status="available",
            last_heartbeat=datetime.utcnow(),
            node_url=construct_node_url(node_id)
        )
        
        try:
            db.add(new_node)
            db.commit()
            print(f"Successfully created node: {node_id}")
            
            return RegistrationResponse(
                node_id=node_id,
                status="registered",
                device_uuid=registration.device_uuid,
                device_name=registration.device_name,
                device_model=registration.device_model,
                node_url=construct_node_url(node_id)
            )
            
        except Exception as e:
            print(f"Error creating node: {str(e)}")
            db.rollback()
            raise
            
    except Exception as e:
        print(f"Registration error: {str(e)}")
        db.rollback()
        # Check if it's a unique violation
        if "UniqueViolation" in str(e):
            # Try one more time to get the existing node
            existing_node = db.query(Node)\
                .filter(Node.device_uuid == registration.device_uuid)\
                .first()
            if existing_node:
                return {
                    "node_id": existing_node.node_id,
                    "status": "already_registered",
                    "device_uuid": existing_node.device_uuid,
                    "device_name": existing_node.device_name,
                    "device_model": existing_node.device_model
                }
        
        raise HTTPException(
            status_code=500,
            detail=f"Registration failed: {str(e)}"
        )
    
@app.post("/models/upload", response_model=ModelUploadResponse)
@handle_db_error
async def upload_model(
    model_file: UploadFile = File(...),
    name: str = Form(...),
    owner_address: str = Form(...),
    db: Session = Depends(get_db)
):
    """Handle model upload and distribution"""
    model_id = str(uuid.uuid4())
    
    try:
        # Create a temporary file
        with NamedTemporaryFile(delete=False) as temp_file:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(model_file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Store file path instead of content
        app.state.model_uploads = getattr(app.state, 'model_uploads', {})
        app.state.model_uploads[model_id] = {
            'file_path': temp_file_path,
            'filename': model_file.filename,
            'expiry': datetime.utcnow() + timedelta(minutes=5)
        }
        
        # Get any available node
        node = db.query(Node)\
            .filter(Node.status == "available")\
            .first()
    
        if not node:
            # Clean up temp file
            os.unlink(temp_file_path)
            raise HTTPException(status_code=503, detail="No nodes available")

        db_model = Model(
            model_id=model_id,
            name=name,
            filename=model_file.filename,
            status="deployed",
            node_id=node.node_id,
            owner_address=owner_address,
            file_size=os.path.getsize(temp_file_path) / (1024 * 1024)  # Size in MB
        )
        
        db.add(db_model)
        db.commit()
        
        return ModelUploadResponse(
            model_id=model_id,
            name=name,
            node_id=node.node_id,
            status="assigned",
            owner_address=owner_address
        )
        
    except Exception as e:
        # Clean up on error
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        app.state.model_uploads.pop(model_id, None)
        raise HTTPException(status_code=500, detail=str(e))

    # Read model file
    # file_size = model_file.size / (1024 * 1024)  # Convert bytes to MB
    # print(f"Uploading file of size: {file_size:.2f}MB")

    # try:
    #     model_id = str(uuid.uuid4())
        
    #     # Read file in chunks to avoid memory issues
    #     chunk_size = 1024 * 1024  # 1MB chunks
    #     content = bytearray()
        
    #     while True:
    #         chunk = await model_file.read(chunk_size)
    #         if not chunk:
    #             break
    #         content.extend(chunk)
        
    #     app.state.model_uploads = getattr(app.state, 'model_uploads', {})
    #     app.state.model_uploads[model_id] = {
    #         'content': bytes(content),
    #         'filename': model_file.filename,
    #         'expiry': datetime.utcnow() + timedelta(minutes=5)
    #     }
        
    #     node = db.query(Node).first()

    # # Create model record
    #     db_model = Model(
    #     model_id=model_id,
    #     name=name,
    #     filename=model_file.filename,
    #     status="deployed",  # Will be changed to "downloaded" when node gets it
    #     # node_id=available_node.node_id,
    #     node_id=node.node_id,
    #     owner_address=owner_address,
    #     file_size=file_size
    #     )
    
    #     db.add(db_model)
    #     db.commit()
    #     return ModelUploadResponse(
    #     model_id=model_id,
    #     name=name,
    #     # node_id=available_node.node_id,
    #     node_id=node.node_id,
    #     status="assigned",
    #     owner_address=owner_address
    # )
    # model_data = await model_file.read()
    
    # file_content = await model_file.read()

    # app.state.model_uploads = getattr(app.state, 'model_uploads', {})
    # app.state.model_uploads[model_id] = {
    #     'content': file_content,
    #     'filename': model_file.filename,
    #     'expiry': datetime.utcnow() + timedelta(minutes=5)  # Clean up after 5 minutes
    # }
    
    # Find available node
    # five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
    # available_node = db.query(Node)\
    #     .filter(Node.status == "available")\
    #     .filter(Node.last_heartbeat >= five_minutes_ago)\
    #     .first()
    
    # if not available_node:
    #     raise HTTPException(status_code=503, detail="No available nodes")
    

@app.get("/models/owner/{owner_address}", response_model=List[ModelInfo])
@handle_db_error
async def get_owner_models(owner_address: str, db: Session = Depends(get_db)):
    """Get all models owned by a specific address"""
    models = db.query(Model)\
        .filter(Model.owner_address == owner_address)\
        .order_by(Model.created_at.desc())\
        .all()
    
    return models

# @app.get("/models/{model_id}/download")
# async def get_model(model_id: str, db: Session = Depends(get_db)):
#     """Download model directly from node"""
#     model = db.query(Model).filter(Model.model_id == model_id).first()
#     if not model:
#         raise HTTPException(status_code=404, detail="Model not found")
    
#     model_upload = getattr(app.state, 'model_uploads', {}).get(model_id)
#     if not model_upload or datetime.utcnow() > model_upload['expiry']:
#         raise HTTPException(status_code=404, detail="Model file no longer available for download. Please upload again.")
    
#     content = model_upload['content']
#     filename = model_upload['filename']
#     app.state.model_uploads.pop(model_id, None)
    
#     return Response(
#         content=content,
#         media_type="application/octet-stream",
#         headers={
#             "Content-Disposition": f"attachment; filename={filename}"
#         }
#     )

@app.get("/models/{model_id}/download")
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """Download model directly from node"""
    # First check if model exists in temporary storage
    model_upload = getattr(app.state, 'model_uploads', {}).get(model_id)
    if not model_upload:
        raise HTTPException(
            status_code=404, 
            detail="Model file not found in temporary storage"
        )
    
    # Check if model has expired
    if datetime.utcnow() > model_upload['expiry']:
        # Clean up expired model
        try:
            os.unlink(model_upload['file_path'])
        except:
            pass
        app.state.model_uploads.pop(model_id, None)
        raise HTTPException(
            status_code=404, 
            detail="Model file has expired. Please upload again."
        )

    file_path = model_upload['file_path']
    filename = model_upload['filename']
    
    async def file_streamer():
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):  # 8KB chunks
                yield chunk
        # Clean up after streaming
        os.unlink(file_path)
        app.state.model_uploads.pop(model_id, None)
    
    return Response(
        file_streamer(),
        media_type='application/octet-stream',
        headers={
            'Content-Disposition': f'attachment; filename="{filename}"'
        }
    )
    # Verify model exists in database
    # model = db.query(Model).filter(Model.model_id == model_id).first()
    # if not model:
    #     raise HTTPException(
    #         status_code=404, 
    #         detail="Model not found in database"
    #     )
    
    # try:
    #     content = model_upload['content']
    #     filename = model_upload['filename']
        
    #     # Remove from temporary storage after successful access
    #     app.state.model_uploads.pop(model_id, None)
        
    #     return Response(
    #         content=content,
    #         media_type="application/octet-stream",
    #         headers={
    #             "Content-Disposition": f'attachment; filename="{filename}"',
    #             "Content-Length": str(len(content))
    #         }
    #     )

@app.post("/nodes/{node_id}/heartbeat")
@handle_db_error
async def node_heartbeat(node_id: str, db: Session = Depends(get_db)):
    """Handle node heartbeat"""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node.last_heartbeat = datetime.utcnow()
    db.commit()
    return {"status": "ok"}

@app.get("/nodes/available", response_model=NodesResponse)
@handle_db_error
async def get_available_nodes(db: Session = Depends(get_db)):
    """Get list of available nodes"""
    five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
    nodes = db.query(Node)\
        .filter(Node.status == "available")\
        .filter(Node.last_heartbeat >= five_minutes_ago)\
        .all()
    
    return NodesResponse(nodes=[
            NodeInfo.from_orm(node) for node in nodes
        ])

@app.get("/nodes/{node_id}/check_assignment")
@handle_db_error
async def check_node_assignment(node_id: str, db: Session = Depends(get_db)):
    """Check node assignment"""
    
    # Update heartbeat and get node
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    # Update heartbeat
    node.last_heartbeat = datetime.utcnow()
    db.commit()
    
    # Check if there's an assigned model that hasn't been downloaded
    assigned_model = db.query(Model)\
        .filter(Model.node_id == node_id)\
        .filter(Model.status == "deployed")\
        .first()
    
    if not assigned_model:
        return NoModelResponse()
    
    model_upload = getattr(app.state, 'model_uploads', {}).get(assigned_model.model_id)
    if not model_upload:
        # Model file no longer available
        return NoModelResponse()
    
    return ModelReadyResponse(
            model_id=assigned_model.model_id,
            download_url=f"/models/{assigned_model.model_id}/download"
    )

@app.post("/models/{model_id}/downloaded")
async def mark_model_downloaded(model_id: str, node_id: str, db: Session = Depends(get_db)):
    """Mark a model as successfully downloaded"""
    model = db.query(Model).filter(Model.model_id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model.status = "downloaded"
    db.commit()
    
    return {"status": "success"}

@app.post("/models/{model_id}/text", 
          description="Text-to-text inference endpoint for language models",
          response_model=dict)
async def run_text_inference(
    model_id: str,
    request: TextInferenceRequest,
    owner_address: str,
    db: Session = Depends(get_db)
):
    """Run text-to-text inference on a language model"""
    # Find model and its node
    model = db.query(Model)\
        .filter(Model.model_id == model_id)\
        .filter(Model.owner_address == owner_address)\
        .first()
    
    if not model:
        raise HTTPException(
            status_code=404, 
            detail="Model not found or you don't have access to it"
        )

    if not model.node_id:
        raise HTTPException(
            status_code=404, 
            detail="Model not assigned to any node"
        )
    
    # Check if node is available
    node = db.query(Node).filter(Node.node_id == model.node_id).first()
    if not node or node.status != "available":
        raise HTTPException(status_code=503, detail="Node not available")
    
    try:
        # Create inference request
        request_id = str(uuid.uuid4())
        inference_request = InferenceRequest(
            request_id=request_id,
            model_id=model_id,
            node_id=node.node_id,
            input_data=json.dumps({
                "prompt": request.prompt,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }).encode(),
            status="pending"
        )
        
        # Mark node as busy
        node.status = "busy"
        db.add(inference_request)
        db.commit()
        
        # Wait for result (with timeout)
        start_time = time.time()
        while time.time() - start_time < 30:  # 30 second timeout
            db.refresh(inference_request)
            if inference_request.status == "completed":
                return {
                    "text": inference_request.result.decode(),
                    "model_id": model_id,
                    "node_id": node.node_id,
                    "request_id": request_id,
                    "prompt_tokens": len(request.prompt.split()),
                    "completion_tokens": len(inference_request.result.decode().split()),
                    "total_tokens": len(request.prompt.split()) + len(inference_request.result.decode().split())
                }
            elif inference_request.status == "failed":
                raise HTTPException(
                    status_code=500, 
                    detail="Text generation failed"
                )
            
            await asyncio.sleep(0.5)
        
        # If we get here, it timed out
        raise HTTPException(
            status_code=504, 
            detail="Text generation timeout"
        )
            
    except Exception as e:
        # Make sure to reset node status on error
        node.status = "available"
        db.commit()
        raise HTTPException(
            status_code=500, 
            detail=f"Text generation error: {str(e)}"
        )

@app.post("/models/upload_and_assign", response_model=ModelUploadResponse)
@handle_db_error
async def upload_and_assign_model(
    model_file: UploadFile = File(...),
    name: str = Form(...),
    owner_address: str = Form(...),
    db: Session = Depends(get_db)
):
    """Upload model and assign to a node"""
    model_id = str(uuid.uuid4())
    
    try:
        # Create a temporary file
        with NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(model_file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Store file path in temporary storage
        app.state.model_uploads = getattr(app.state, 'model_uploads', {})
        app.state.model_uploads[model_id] = {
            'file_path': temp_file_path,
            'filename': model_file.filename,
            'expiry': datetime.utcnow() + timedelta(minutes=5)
        }
        
        # Get any available node
        node = db.query(Node)\
            .filter(Node.status == "available")\
            .first()
        
        if not node:
            os.unlink(temp_file_path)
            raise HTTPException(status_code=503, detail="No nodes available")
      
        notify_node_to_download(node.node_id, model_id)
        
        # Directly return the response without storing in the database
        return ModelUploadResponse(
            model_id=model_id,
            name=name,
            node_id=node.node_id,
            status="assigned",
            owner_address=owner_address
        )
        
    except Exception as e:
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        app.state.model_uploads.pop(model_id, None)
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def setup_upload_cleanup():
    async def cleanup_old_uploads():
        while True:
            await asyncio.sleep(300)  # Run every 5 minutes
            now = datetime.utcnow()
            if hasattr(app.state, 'model_uploads'):
                expired = [
                    model_id for model_id, data 
                    in app.state.model_uploads.items() 
                    if now > data['expiry']
                ]
                for model_id in expired:
                    app.state.model_uploads.pop(model_id, None)
    
    asyncio.create_task(cleanup_old_uploads())

# def notify_node_to_download(node_id: str, model_id: str, db: Session):
#     """Notify the node to download the model"""
#     # Implement a notification mechanism, e.g., WebSocket, HTTP request, etc.
#     # For example, send an HTTP request to the node's endpoint
#     node = db.query(Node).filter(Node.node_id == node_id).first()
#     if not node:
#         print(f"Node {node_id} not found or URL not set")
#         raise HTTPException(status_code=404, detail="Node not found or URL not set")
#     try:
#         response = requests.post(node.node_url, json={"model_id": model_id})
#         response.raise_for_status()
#     except Exception as e:
#         print(f"Failed to notify node {node_id}: {str(e)}")
def notify_node_to_download(node_id: str, model_id: str):
         """Notify the node to download the model"""
         try:
             node_url = f"http://localhost:5000/nodes/{node_id}/download_model"
             response = requests.post(node_url, json={"model_id": model_id})
             response.raise_for_status()
             print(f"Successfully notified node {node_id} to download model {model_id}")
         except requests.exceptions.RequestException as e:
             print(f"Failed to notify node {node_id}: {str(e)}")