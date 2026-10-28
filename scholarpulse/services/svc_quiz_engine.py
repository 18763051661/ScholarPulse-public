"""[005] svc_quiz_engine 领域服务。"""
class SvcQuizEngineService:
    def handle(self, payload: dict) -> dict:
        return {"service": "svc_quiz_engine", "received": payload}
