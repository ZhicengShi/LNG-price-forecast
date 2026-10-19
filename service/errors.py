"""业务错误及对外字段；不返回内部文件路径。"""


def issue(code, message, region=None, month=None, field=None, suggestion=None):
    return dict(code=code, region=region, month=month, field=field,
                message=message, suggestion=suggestion)


class APIError(Exception):
    def __init__(self, status, code, message, details=None):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.details = details or []

