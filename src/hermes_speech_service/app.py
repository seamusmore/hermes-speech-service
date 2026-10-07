"""One ASGI server, isolated STT/TTS namespaces and a shared lifecycle."""
import asyncio
import hmac
import importlib
import os
from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def create_app(*, token=None, components=None):
    token = os.getenv("HERMES_SPEECH_SERVICE_TOKEN", "") if token is None else token
    if components is None:
        components = {name: importlib.import_module(f".{name}.app", __package__)
                      for name in ("stt", "tts")}

    @asynccontextmanager
    async def lifespan(app):
        async with AsyncExitStack() as stack:
            for component in components.values():
                await stack.enter_async_context(component.lifespan(component.app))
            yield

    app = FastAPI(title="Hermes Speech Service", version="1.0", lifespan=lifespan)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.url.path != "/health":
            supplied = request.headers.get("authorization", "")
            if token:
                if not hmac.compare_digest(supplied, "Bearer " + token):
                    return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            elif request.client and request.client.host not in ("127.0.0.1", "::1", "localhost", "testclient"):
                return JSONResponse({"detail": "Remote clients require a service token"}, status_code=403)
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"service": "hermes-speech-service", "protocol": 1,
                "status": "ok", "capabilities": sorted(components)}

    @app.get("/capabilities")
    def capabilities():
        return {"protocol": 1, "stt": "stt" in components, "tts": "tts" in components,
                "tts_streaming": "tts" in components, "realtime": False,
                "paths": {name: "/" + name for name in components}}

    @app.get("/ready")
    async def ready():
        states = {name: await asyncio.to_thread(component.readiness_state)
                  for name, component in components.items()}
        is_ready = all(state.get("ready", False) for state in states.values())
        return JSONResponse({"ready": is_ready, "components": states}, status_code=200 if is_ready else 503)

    @app.post("/warmup")
    async def warmup():
        values = await asyncio.gather(*(asyncio.to_thread(component.warmup)
                                       for component in components.values()), return_exceptions=True)
        states = {name: ({"ready": False, "error": type(value).__name__}
                         if isinstance(value, Exception) else value)
                  for name, value in zip(components, values)}
        return {"ready": all(state.get("ready", False) for state in states.values()), "components": states}

    for name, component in components.items():
        app.mount("/" + name, component.app)
    return app
