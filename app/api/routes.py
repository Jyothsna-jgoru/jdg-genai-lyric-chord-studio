from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.base import get_db
from app.ml.model import model_service
from app.schemas.studio import (
    GenerateRequest, GenerateResponse, LyricAnalysisRequest, LyricAnalysisResponse, ProjectDetail,
    ProjectPayload, ProjectRename, ProjectSummary, RegenerateRequest, TimelineEvent, TimelineRequest,
    TransposeRequest, ValidateRequest,
)
from app.services.exports import chord_sheet_text, json_export, midi_export, pdf_export, safe_filename
from app.services.generation import generate_song, playback_timeline, transpose_result
from app.services.lyrics import analyze_lyrics
from app.services.projects import (
    ProjectConflictError, create_project, delete_project, duplicate_project, get_project,
    list_projects, rename_project, update_project,
)
from app.theory.engine import theory


router = APIRouter(prefix=settings.api_prefix)


@router.get("/health")
def health():
    return {"status": "ok", "application": settings.app_name, "model": model_service.health()}


@router.get("/model")
def model_info():
    return model_service.health()


@router.get("/evaluation")
def evaluation_summary():
    if not settings.evaluation_path.exists():
        return {"available": False, "message": "No evaluation has been run yet."}
    return {"available": True, **json.loads(settings.evaluation_path.read_text(encoding="utf-8"))}


@router.post("/lyrics/analyze", response_model=LyricAnalysisResponse)
def analyze(request: LyricAnalysisRequest):
    return analyze_lyrics(request.lyrics)


@router.post("/chords/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    return generate_song(request)


@router.post("/chords/regenerate-section", response_model=GenerateResponse)
def regenerate(request: RegenerateRequest):
    if not any(section.name.lower() == request.section_name.lower() for section in (request.sections or analyze_lyrics(request.lyrics).sections)):
        raise HTTPException(status_code=404, detail="Section not found")
    return generate_song(GenerateRequest(**request.model_dump(exclude={"section_name"})), only_section=request.section_name)


@router.post("/chords/validate")
def validate(request: ValidateRequest):
    valid, reason = theory.validate_chord(request.chord, request.difficulty)
    return {"valid": valid, "reason": reason}


@router.post("/transpose", response_model=GenerateResponse)
def transpose(request: TransposeRequest):
    try: return transpose_result(request.result, request.target_key)
    except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/playback/timeline", response_model=list[TimelineEvent])
def timeline(request: TimelineRequest):
    try: return playback_timeline(request.result)
    except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/projects", response_model=ProjectDetail, status_code=status.HTTP_201_CREATED)
def project_create(payload: ProjectPayload, db: Session = Depends(get_db)):
    try: return create_project(db, payload)
    except ProjectConflictError as exc: raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/projects", response_model=list[ProjectSummary])
def project_list(db: Session = Depends(get_db)):
    return list_projects(db)


@router.get("/projects/{project_id}", response_model=ProjectDetail)
def project_open(project_id: int, db: Session = Depends(get_db)):
    project = get_project(db, project_id)
    if not project: raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.put("/projects/{project_id}", response_model=ProjectDetail)
def project_update(project_id: int, payload: ProjectPayload, db: Session = Depends(get_db)):
    try: project = update_project(db, project_id, payload)
    except ProjectConflictError as exc: raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not project: raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.patch("/projects/{project_id}/rename", response_model=ProjectDetail)
def project_rename(project_id: int, payload: ProjectRename, db: Session = Depends(get_db)):
    try: project = rename_project(db, project_id, payload.name)
    except ProjectConflictError as exc: raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not project: raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.post("/projects/{project_id}/duplicate", response_model=ProjectDetail, status_code=status.HTTP_201_CREATED)
def project_duplicate(project_id: int, db: Session = Depends(get_db)):
    project = duplicate_project(db, project_id)
    if not project: raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def project_delete(project_id: int, confirm: bool = Query(False), db: Session = Depends(get_db)):
    if not confirm: raise HTTPException(status_code=400, detail="Deletion requires confirm=true")
    if not delete_project(db, project_id): raise HTTPException(status_code=404, detail="Project not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/exports/{kind}")
def export_result(kind: Literal["txt", "json", "pdf", "midi"], result: GenerateResponse):
    try:
        if kind == "txt": content, media = chord_sheet_text(result).encode(), "text/plain; charset=utf-8"
        elif kind == "json": content, media = json_export(result), "application/json"
        elif kind == "pdf": content, media = pdf_export(result), "application/pdf"
        else: content, media = midi_export(result), "audio/midi"
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Export failed validation: {exc}") from exc
    return Response(content=content, media_type=media, headers={"Content-Disposition": f'attachment; filename="{safe_filename(result.project_name, kind if kind != "midi" else "mid")}"'})

