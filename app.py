from fastapi import FastAPI, UploadFile, HTTPException, Depends
from sqlalchemy import create_engine, Column, String, Float, DateTime, ForeignKey, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from typing import List
import uuid
import os
import time
from datetime import datetime, timedelta  # Added timedelta here
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.responses import Response

load_dotenv()

app = FastAPI()

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class NodeRegistration(BaseModel):
    device_uuid: str  # Unique identifier from the device
    device_name: str | None = None
    device_model: str | None = None

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

class Model(Base):
    __tablename__ = "models"
    
    model_id = Column(String, primary_key=True)
    model_data = Column(LargeBinary)  # Store model binary directly in DB
    filename = Column(String)
    status = Column(String)  # uploaded, deployed, failed
    node_id = Column(String, ForeignKey("nodes.node_id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
class NodeCreate(BaseModel):
    node_id: str
    status: str
    current_model_id: str | None = None

class NodeResponse(BaseModel):
    node_id: str
    status: str
    current_model_id: str | None

@app.post("/nodes/register")
async def register_node(registration: NodeRegistration, db: Session = Depends(get_db)):
    """Register a new iPhone node"""
    # Check if device already exists
    existing_node = db.query(Node)\
        .filter(Node.device_uuid == registration.device_uuid)\
        .first()
    
    if existing_node:
        # Update last heartbeat and return existing node_id
        existing_node.last_heartbeat = datetime.utcnow()
        existing_node.status = "available"  # Reset status to available
        db.commit()
        return {"node_id": existing_node.node_id, "status": "already_registered"}
    
    # Create new node if device doesn't exist
    node_id = str(uuid.uuid4())
    db_node = Node(
        node_id=node_id,
        device_uuid=registration.device_uuid,
        device_name=registration.device_name,
        device_model=registration.device_model,
        status="available",
        last_heartbeat=datetime.utcnow()
    )
    
    db.add(db_node)
    db.commit()
    db.refresh(db_node)
    
    return {
        "node_id": node_id,
        "status": "new_registration",
        "device_uuid": registration.device_uuid,
        "device_model": registration.device_model,
        "device_name": registration.device_name
    }
@app.post("/models/upload")
async def upload_model(
    model_file: UploadFile,
    db: Session = Depends(get_db)
):
    """Handle model upload and distribution"""
    model_id = str(uuid.uuid4())
    
    # Read model file
    model_data = await model_file.read()
    
    # Find available node
    five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
    available_node = db.query(Node)\
        .filter(Node.status == "available")\
        .filter(Node.last_heartbeat >= five_minutes_ago)\
        .first()
    
    if not available_node:
        raise HTTPException(status_code=503, detail="No available nodes")
    
    # Create model record
    db_model = Model(
        model_id=model_id,
        model_data=model_data,
        filename=model_file.filename,
        status="deployed",  # Will be changed to "downloaded" when node gets it
        node_id=available_node.node_id
    )
    
    db.add(db_model)
    db.commit()
    
    return {
        "model_id": model_id,
        "node_id": available_node.node_id,
        "status": "assigned"
    }
    
@app.get("/models/{model_id}/download")
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """Download model directly from database"""
    model = db.query(Model).filter(Model.model_id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return Response(
        content=model.model_data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename={model.filename}"
        }
    )

@app.post("/nodes/{node_id}/heartbeat")
async def node_heartbeat(node_id: str, db: Session = Depends(get_db)):
    """Handle node heartbeat"""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node.last_heartbeat = datetime.utcnow()
    db.commit()
    return {"status": "ok"}

@app.get("/nodes/available")
async def get_available_nodes(db: Session = Depends(get_db)):
    """Get list of available nodes"""
    try:
        five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
        nodes = db.query(Node)\
            .filter(Node.status == "available")\
            .filter(Node.last_heartbeat >= five_minutes_ago)\
            .all()
        
        return {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "device_name": node.device_name,
                    "device_model": node.device_model,
                    "status": node.status,
                    "last_heartbeat": node.last_heartbeat
                } for node in nodes
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )

@app.get("/nodes/{node_id}/check_assignment")
async def check_node_assignment(node_id: str, db: Session = Depends(get_db)):
    """Check if there's a model assigned to this node"""
    
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
        return {"status": "no_model"}
    
    return {
        "status": "model_ready",
        "model_id": assigned_model.model_id,
        "download_url": f"/models/{assigned_model.model_id}/download"
    }

@app.post("/models/{model_id}/downloaded")
async def mark_model_downloaded(model_id: str, node_id: str, db: Session = Depends(get_db)):
    """Mark a model as successfully downloaded"""
    model = db.query(Model).filter(Model.model_id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model.status = "downloaded"
    db.commit()
    
    return {"status": "success"}