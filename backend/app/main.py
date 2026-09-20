from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.app.api import RunCoordinator, router
from backend.app.config import Settings, get_settings
from backend.app.db import session_factory_for
from backend.app.security import security_headers_middleware
from backend.slam.pipeline import SlamPipeline


def create_app(
    *, settings: Settings | None = None, pipeline: SlamPipeline | None = None
) -> FastAPI:
    configured = settings or get_settings()
    production = configured.app_env.lower() == "production"
    app = FastAPI(
        title="O-HIVE Monocular Sparse SLAM",
        debug=False,
        docs_url=None if production else "/docs",
        redoc_url=None,
        openapi_url=None if production else "/openapi.json",
    )
    session_factory, engine = session_factory_for(configured.database_url)
    app.state.settings = configured
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.coordinator = RunCoordinator(
        configured,
        pipeline or SlamPipeline(configured),
        session_factory,
    )
    allowed_hosts = [host.strip() for host in configured.allowed_hosts.split(",") if host.strip()]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    if not production:
        origins = [
            origin.strip()
            for origin in configured.development_cors_origins.split(",")
            if origin.strip()
        ]
        if origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=False,
                allow_methods=["GET", "POST"],
                allow_headers=["Content-Type"],
            )
    app.middleware("http")(security_headers_middleware)
    app.include_router(router)

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger("o_hive_slam").exception(
            "unhandled_request_error path=%s category=%s",
            request.url.path,
            type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": {
                    "code": "INTERNAL_ERROR",
                    "message": "The request could not be completed.",
                }
            },
        )

    return app


app = create_app()
