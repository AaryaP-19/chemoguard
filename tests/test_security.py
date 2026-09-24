"""Regression tests for the security fixes.

Each of these failed before the fix and would fail again if it regressed.
"""

import io


# --- debug server -----------------------------------------------------------

def test_debug_is_off_by_default(flask_app):
    """FLASK_DEBUG is unset in conftest, so the debugger must be off.

    Guards the non-obvious half of this fix: DevelopmentConfig sets
    DEBUG = True, so app.config has to be overridden explicitly. Checking
    only the app.run() call would not catch a regression here.
    """
    assert flask_app.config["DEBUG"] is False


# --- CSRF -------------------------------------------------------------------

def test_csrf_protection_is_installed(flask_app):
    assert "csrf" in flask_app.extensions


def test_index_renders_a_csrf_token(csrf_token):
    assert len(csrf_token) > 20


def test_mutating_request_without_a_token_is_rejected(client):
    response = client.post(
        "/api/auth/register",
        json={
            "username": "no_token",
            "email": "no_token@example.com",
            "password": "correct-horse",
            "full_name": "No Token",
        },
    )

    assert response.status_code == 400


def test_mutating_request_with_a_token_is_accepted(client, csrf_token):
    response = client.post(
        "/api/auth/register",
        json={
            "username": "with_token",
            "email": "with_token@example.com",
            "password": "correct-horse",
            "full_name": "With Token",
        },
        headers={"X-CSRFToken": csrf_token},
    )

    assert response.status_code in (200, 201), response.data


# --- /upload authentication -------------------------------------------------

def test_anonymous_upload_is_refused(client, csrf_token):
    response = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"%PDF-1.4 not a real report"), "report.pdf")},
        content_type="multipart/form-data",
        headers={"X-CSRFToken": csrf_token},
    )

    assert response.status_code in (401, 302)


def test_anonymous_upload_writes_nothing_to_disk(flask_app, client, csrf_token):
    """The original bug: the file was saved before any identity was checked."""
    import os

    upload_dir = flask_app.config["UPLOAD_FOLDER"]
    before = set(os.listdir(upload_dir))

    client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"%PDF-1.4 not a real report"), "evil.pdf")},
        content_type="multipart/form-data",
        headers={"X-CSRFToken": csrf_token},
    )

    assert set(os.listdir(upload_dir)) == before


# --- /reload_models authentication ------------------------------------------

def test_anonymous_reload_models_is_refused(client, csrf_token):
    """A CSRF token is not access control on this route.

    The token is handed to anyone who fetches ``/``, so before
    ``@login_required`` an anonymous client could reload every artefact from
    disk simply by asking for one first. Sending a valid token here is the
    point of the test, not an oversight.
    """
    response = client.post("/reload_models", headers={"X-CSRFToken": csrf_token})

    assert response.status_code in (401, 302)
