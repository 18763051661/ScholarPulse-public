"""[002] svc_user 领域服务。"""
class SvcUserService:
    def handle(self, payload: dict) -> dict:
        return {"service": "svc_user", "received": payload}
