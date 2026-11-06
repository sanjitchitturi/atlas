"""Bound request bodies before parsing and reject untrusted browser writes."""

from starlette.responses import JSONResponse


class RequestSecurityMiddleware:
    def __init__(self, app, *, allowed_origins, max_bytes):
        self.app = app
        self.allowed_origins = set(allowed_origins)
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        origin = headers.get(b"origin")
        host = headers.get(b"host", b"").decode("latin1")
        own = f"{scope.get('scheme', 'http')}://{host}"
        if scope["method"] not in {"GET", "HEAD", "OPTIONS"} and origin is not None:
            value = origin.decode("latin1")
            if value not in self.allowed_origins and value != own:
                return await JSONResponse({"detail": "Untrusted request origin."}, 403)(
                    scope, receive, send
                )
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_bytes:
                return await JSONResponse({"detail": "Request exceeds size limit."}, 413)(
                    scope, receive, send
                )
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        async def protected_send(message):
            if message["type"] == "http.response.start":
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"x-frame-options", b"DENY"),
                ]
            await send(message)

        await self.app(scope, replay, protected_send)
