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

# Database Models
class Node(Base):
    __tablename__ = "nodes"
    
    node_id = Column(String, primary_key=True)
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
async def register_node(db: Session = Depends(get_db)):
    """Register a new iPhone node"""
    node_id = str(uuid.uuid4())
    db_node = Node(
        node_id=node_id,
        status="available",
        last_heartbeat=datetime.utcnow()
    )
    db.add(db_node)
    db.commit()
    db.refresh(db_node)
    return {"node_id": node_id}

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
        status="deployed",
        node_id=available_node.node_id
    )
    
    # Update node status
    available_node.status = "busy"
    available_node.current_model_id = model_id
    
    db.add(db_model)
    db.commit()
    
    return {
        "model_id": model_id,
        "node_id": available_node.node_id,
        "download_url": f"/models/{model_id}/download"
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
    nodes = db.query(Node)\
        .filter(Node.status == "available")\
        .filter(Node.last_heartbeat >= datetime.utcnow().timestamp() - 300)\
        .all()
    return {"nodes": nodes}