from __future__ import annotations


def generate(client, song_payload):
    response = client.post("/api/v1/chords/generate", json=song_payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_health_analysis_generation_and_errors(client, song_payload):
    assert client.get("/api/v1/health").status_code == 200
    analysis = client.post("/api/v1/lyrics/analyze", json={"lyrics": song_payload["lyrics"]})
    assert analysis.status_code == 200 and len(analysis.json()["sections"]) == 2
    result = generate(client, song_payload)
    assert result["generation_source"] == "algorithmic_fallback_output"
    assert client.post("/api/v1/chords/generate", json={**song_payload, "lyrics": ""}).status_code == 422
    invalid_tempo = {**song_payload, "controls": {**song_payload["controls"], "tempo": 500}}
    assert client.post("/api/v1/chords/generate", json=invalid_tempo).status_code == 422
    assert client.post("/api/v1/lyrics/analyze", json={"lyrics": "x" * 20001}).status_code == 422


def test_project_crud_and_delete_confirmation(client, song_payload):
    result = generate(client, song_payload)
    payload = {"name": "Test Song", "lyrics": song_payload["lyrics"], "result": result, "manual_chord_edits": []}
    created = client.post("/api/v1/projects", json=payload); assert created.status_code == 201
    project_id = created.json()["id"]
    assert len(client.get("/api/v1/projects").json()) == 1
    assert client.get(f"/api/v1/projects/{project_id}").status_code == 200
    assert client.patch(f"/api/v1/projects/{project_id}/rename", json={"name": "Renamed"}).json()["name"] == "Renamed"
    duplicate = client.post(f"/api/v1/projects/{project_id}/duplicate"); assert duplicate.status_code == 201
    assert client.delete(f"/api/v1/projects/{project_id}").status_code == 400
    assert client.delete(f"/api/v1/projects/{project_id}?confirm=true").status_code == 204


def test_validation_transpose_timeline_and_exports(client, song_payload):
    result = generate(client, song_payload)
    assert client.post("/api/v1/chords/validate", json={"chord": "C", "difficulty": "beginner"}).json()["valid"]
    assert not client.post("/api/v1/chords/validate", json={"chord": "H7", "difficulty": "advanced"}).json()["valid"]
    transposed = client.post("/api/v1/transpose", json={"result": result, "target_key": "Eb"}); assert transposed.status_code == 200
    timeline = client.post("/api/v1/playback/timeline", json={"result": transposed.json()}); assert timeline.status_code == 200 and timeline.json()
    assert timeline.json()[0]["melody_midi_notes"] and timeline.json()[0]["lyric_fragment"]
    for kind, signature in [("txt", b"Test Song"), ("json", b"project_name"), ("pdf", b"%PDF"), ("midi", b"MThd")]:
        response = client.post(f"/api/v1/exports/{kind}", json=transposed.json())
        assert response.status_code == 200 and signature in response.content[:1000]
