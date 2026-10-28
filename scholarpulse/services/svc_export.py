"""[010] svc_export 领域服务。"""
class SvcExportService:
    def handle(self, payload: dict) -> dict:
        return {"service": "svc_export", "received": payload}
