from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Project
from app.schemas.studio import ProjectDetail, ProjectPayload, ProjectSummary


class ProjectConflictError(ValueError):
    """Raised when a project name would violate the uniqueness constraint."""


def _detail(project: Project) -> ProjectDetail:
    return ProjectDetail(
        id=project.id, name=project.name, lyrics=project.original_lyrics,
        generation_source=project.generation_source, created_at=project.created_at, updated_at=project.updated_at,
        result=json.loads(project.validated_output), manual_chord_edits=json.loads(project.manual_chord_edits),
    )


def create_project(db: Session, payload: ProjectPayload) -> ProjectDetail:
    result = payload.result
    project = Project(
        name=payload.name, original_lyrics=payload.lyrics,
        detected_sections=json.dumps([section.model_dump() for section in result.analysis.sections]),
        edited_sections=json.dumps([section.model_dump() for section in result.analysis.sections]),
        controls=result.controls.model_dump_json(), analysis_features=result.analysis.model_dump_json(),
        raw_model_output=json.dumps(result.raw_model_outputs), validated_output=result.model_dump_json(),
        repair_details=json.dumps([repair.model_dump() for repair in result.repairs]),
        chord_alignment=json.dumps([item.model_dump() for item in result.alignments]),
        manual_chord_edits=json.dumps(payload.manual_chord_edits), random_seed=result.controls.seed,
        decoding_settings=json.dumps({"method": result.controls.decoding_method, "num_beams": result.controls.num_beams,
                                      "temperature": result.controls.temperature, "top_k": result.controls.top_k}),
        generation_source=result.generation_source,
        adapter_version="dev" if "lora" in result.generation_source else "unavailable",
    )
    db.add(project)
    try:
        db.commit(); db.refresh(project)
    except IntegrityError as exc:
        db.rollback(); raise ProjectConflictError("A project with that name already exists") from exc
    return _detail(project)


def list_projects(db: Session) -> list[ProjectSummary]:
    projects = db.scalars(select(Project).order_by(Project.updated_at.desc())).all()
    return [ProjectSummary.model_validate(project) for project in projects]


def get_project(db: Session, project_id: int) -> ProjectDetail | None:
    project = db.get(Project, project_id)
    return _detail(project) if project else None


def update_project(db: Session, project_id: int, payload: ProjectPayload) -> ProjectDetail | None:
    project = db.get(Project, project_id)
    if not project: return None
    project.name = payload.name; project.original_lyrics = payload.lyrics
    project.validated_output = payload.result.model_dump_json()
    project.manual_chord_edits = json.dumps(payload.manual_chord_edits)
    project.controls = payload.result.controls.model_dump_json()
    project.chord_alignment = json.dumps([item.model_dump() for item in payload.result.alignments])
    project.repair_details = json.dumps([repair.model_dump() for repair in payload.result.repairs])
    project.generation_source = payload.result.generation_source
    try:
        db.commit(); db.refresh(project)
    except IntegrityError as exc:
        db.rollback(); raise ProjectConflictError("A project with that name already exists") from exc
    return _detail(project)


def rename_project(db: Session, project_id: int, name: str) -> ProjectDetail | None:
    project = db.get(Project, project_id)
    if not project: return None
    project.name = name
    try:
        db.commit(); db.refresh(project)
    except IntegrityError as exc:
        db.rollback(); raise ProjectConflictError("A project with that name already exists") from exc
    return _detail(project)


def duplicate_project(db: Session, project_id: int) -> ProjectDetail | None:
    project = db.get(Project, project_id)
    if not project: return None
    base, suffix = f"{project.name} Copy", 2
    name = base
    while db.scalar(select(Project).where(Project.name == name)):
        name = f"{base} {suffix}"; suffix += 1
    duplicate = Project(**{column.name: getattr(project, column.name) for column in Project.__table__.columns
                           if column.name not in {"id", "name", "created_at", "updated_at"}})
    duplicate.name = name
    db.add(duplicate); db.commit(); db.refresh(duplicate)
    return _detail(duplicate)


def delete_project(db: Session, project_id: int) -> bool:
    project = db.get(Project, project_id)
    if not project: return False
    db.delete(project); db.commit(); return True
