"""Shared pytest fixtures for ChemoGuard.

The application module reads its configuration at import time, so every
environment variable it depends on has to be set *before* `import app`.
That is why the os.environ assignments below sit at module scope rather than
inside a fixture.
"""

import os
import sys
import tempfile

# The application lives at the repository root, one level up from tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DB_FD, _DB_PATH = tempfile.mkstemp(suffix=".db", prefix="chemoguard-test-")
os.close(_DB_FD)

# Never touch a real database or a real secret from the test suite.
os.environ["DATABASE_URL"] = "sqlite:///" + _DB_PATH.replace("\\", "/")
os.environ["SECRET_KEY"] = "test-only-key-not-used-anywhere-else"
# Explicitly off: a test run must never bring up the Werkzeug debug console.
os.environ.pop("FLASK_DEBUG", None)

import pytest  # noqa: E402

import app as app_module  # noqa: E402


@pytest.fixture(scope="session")
def flask_app(tmp_path_factory):
    """The real application object, pointed at a throwaway database."""
    application = app_module.app
    application.config["TESTING"] = True
    # Keep uploads out of the repository's uploads/ directory.
    application.config["UPLOAD_FOLDER"] = str(tmp_path_factory.mktemp("uploads"))

    with application.app_context():
        app_module.db.create_all()

    # Deliberately NOT yielding inside an app context. Flask reuses an
    # already-pushed app context for same-app requests, so a session-scoped
    # context would share one flask.g across every test. Flask-WTF caches the
    # CSRF token on g, so one test's token would leak into the next test's
    # client while that client's session held a different value, and every
    # token check after the first would fail with a 400.
    yield application

    with application.app_context():
        app_module.db.session.remove()
        app_module.db.drop_all()

    try:
        os.unlink(_DB_PATH)
    except OSError:
        pass


@pytest.fixture
def client(flask_app):
    """A fresh test client, so each test starts logged out."""
    return flask_app.test_client()


@pytest.fixture
def csrf_token(client):
    """A valid CSRF token for the given client's session.

    Read from the meta tag the index template renders, which is the same
    route the browser uses.
    """
    import re

    page = client.get("/")
    match = re.search(rb'name="csrf-token" content="([^"]+)"', page.data)
    assert match, "index.html did not render a csrf-token meta tag"
    return match.group(1).decode()
