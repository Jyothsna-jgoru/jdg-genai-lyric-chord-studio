from __future__ import annotations

import os

os.environ.setdefault("JDG_MODEL_AUTOLOAD", "false")
os.environ.setdefault("JDG_ALLOW_MODEL_DOWNLOAD", "false")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base, get_db
from app.main import app
from app.ml.model import model_service


@pytest.fixture
def client(tmp_path, monkeypatch):
    database = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{database}", connect_args={"check_same_thread": False})
    TestingSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    Base.metadata.create_all(engine)

    def override_db():
        session = TestingSession()
        try: yield session
        finally: session.close()

    app.dependency_overrides[get_db] = override_db
    monkeypatch.setattr(model_service, "generate", lambda instruction, controls: (None, 0.0))
    monkeypatch.setattr(model_service, "base_loaded", False)
    monkeypatch.setattr(model_service, "adapter_loaded", False)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def song_payload():
    return {
        "project_name": "Test Song",
        "lyrics": "[Verse]\nMorning light arrives again\n[Chorus]\nWe rise together now",
        "controls": {"key": "C", "scale": "major", "genre": "pop", "mood": "hopeful", "tempo": 96,
                     "time_signature": "4/4", "difficulty": "beginner", "chord_density": 2,
                     "variation": .35, "seed": 42, "decoding_method": "greedy", "num_beams": 4,
                     "temperature": .8, "top_k": 40},
    }

