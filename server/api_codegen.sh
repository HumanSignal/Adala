#!/bin/bash

PYCMD=$(cat <<EOF
import json
from fastapi.openapi.utils import get_openapi
from app import app

with open('openapi.json', 'w') as f:
    json.dump(get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    ), f)
EOF
)

poetry run python -c "$PYCMD"
