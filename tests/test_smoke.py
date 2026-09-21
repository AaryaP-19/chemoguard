"""Smoke tests: the application imports, starts, and serves.

Kept deliberately free of any dependency on models/ -- the pickles are being
replaced (CLAUDE.md Findings 1 and 5) and a smoke test that needs them would
have to be deleted along with them.
"""

import flask


def test_app_imports_and_is_a_flask_app():
    import app

    assert isinstance(app.app, flask.Flask)


def test_app_starts_and_serves_the_index(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"<html" in response.data.lower()


def test_health_endpoint_responds(client):
    assert client.get("/health").status_code == 200


def test_both_blueprints_are_registered(flask_app):
    assert set(flask_app.blueprints) == {"auth", "routes"}


def test_expected_routes_are_mapped(flask_app):
    paths = {str(rule) for rule in flask_app.url_map.iter_rules()}

    for expected in (
        "/api/auth/login",
        "/api/auth/register",
        "/api/upload",
        "/api/predict",
    ):
        assert expected in paths, f"{expected} is not routed"
