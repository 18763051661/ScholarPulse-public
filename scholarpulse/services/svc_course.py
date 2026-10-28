"""[003] svc_course 领域服务。"""
class SvcCourseService:
    def handle(self, payload: dict) -> dict:
        return {"service": "svc_course", "received": payload}
