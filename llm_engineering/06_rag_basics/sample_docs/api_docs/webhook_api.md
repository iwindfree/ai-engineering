# CloudStore Webhook API

## 개요

Webhook을 사용하면 CloudStore에서 발생하는 이벤트를 실시간으로 받을 수 있습니다. 파일 업로드, 삭제, 공유 등의 이벤트를 앱에서 처리할 수 있습니다.

## Webhook 설정

### Webhook 생성

**POST** `/api/v1/webhooks`

새 Webhook을 생성합니다.

**Request:**
```json
{
  "url": "https://your-app.com/webhooks/cloudstore",
  "events": ["file.uploaded", "file.deleted", "file.shared"],
  "secret": "your_webhook_secret",
  "active": true
}
```

**Response:**
```json
{
  "webhook_id": "wh_1234567890",
  "url": "https://your-app.com/webhooks/cloudstore",
  "events": ["file.uploaded", "file.deleted", "file.shared"],
  "active": true,
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Webhook 목록 조회

**GET** `/api/v1/webhooks`

생성한 Webhook 목록을 조회합니다.

**Response:**
```json
{
  "webhooks": [
    {
      "webhook_id": "wh_1234567890",
      "url": "https://your-app.com/webhooks/cloudstore",
      "events": ["file.uploaded", "file.deleted"],
      "active": true,
      "created_at": "2024-01-15T10:30:00Z",
      "last_triggered_at": "2024-01-16T08:20:00Z"
    }
  ]
}
```

### Webhook 수정

**PATCH** `/api/v1/webhooks/{webhook_id}`

기존 Webhook을 수정합니다.

**Request:**
```json
{
  "events": ["file.uploaded", "file.shared"],
  "active": false
}
```

### Webhook 삭제

**DELETE** `/api/v1/webhooks/{webhook_id}`

Webhook을 삭제합니다.

**Response:**
```json
{
  "success": true,
  "message": "Webhook deleted successfully"
}
```

## 이벤트 타입

### 파일 이벤트

- `file.uploaded`: 파일이 업로드됨
- `file.updated`: 파일이 업데이트됨
- `file.deleted`: 파일이 삭제됨
- `file.downloaded`: 파일이 다운로드됨
- `file.shared`: 파일이 공유됨

### 폴더 이벤트

- `folder.created`: 폴더가 생성됨
- `folder.deleted`: 폴더가 삭제됨
- `folder.moved`: 폴더가 이동됨

### 공유 이벤트

- `share.created`: 공유 링크가 생성됨
- `share.accessed`: 공유 링크가 액세스됨
- `share.expired`: 공유 링크가 만료됨

### 사용자 이벤트

- `user.login`: 사용자가 로그인함
- `user.logout`: 사용자가 로그아웃함

## Webhook 페이로드

모든 Webhook 이벤트는 다음 구조를 따릅니다:

```json
{
  "event_id": "evt_1234567890",
  "event_type": "file.uploaded",
  "timestamp": "2024-01-16T10:30:00Z",
  "data": {
    "file_id": "f_1234567890",
    "name": "report.pdf",
    "path": "/documents/report.pdf",
    "size": 2048576,
    "uploaded_by": "user_123"
  }
}
```

### file.uploaded 예시

```json
{
  "event_id": "evt_1234567890",
  "event_type": "file.uploaded",
  "timestamp": "2024-01-16T10:30:00Z",
  "data": {
    "file_id": "f_1234567890",
    "name": "report.pdf",
    "path": "/documents/report.pdf",
    "size": 2048576,
    "mime_type": "application/pdf",
    "uploaded_by": "user_123"
  }
}
```

### file.shared 예시

```json
{
  "event_id": "evt_2345678901",
  "event_type": "file.shared",
  "timestamp": "2024-01-16T11:00:00Z",
  "data": {
    "file_id": "f_1234567890",
    "share_link": "https://cloudstore.com/s/abc123",
    "shared_by": "user_123",
    "expires_at": "2024-01-23T11:00:00Z",
    "password_protected": true
  }
}
```

## Webhook 검증

보안을 위해 Webhook 요청을 검증해야 합니다.

### 서명 검증

CloudStore는 모든 Webhook 요청에 `X-CloudStore-Signature` 헤더를 포함합니다.

**Python 예시:**

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected_signature, signature)

# 사용 예
payload = request.body
signature = request.headers['X-CloudStore-Signature']
secret = 'your_webhook_secret'

if verify_webhook(payload, signature, secret):
    # Webhook 처리
    pass
else:
    # 검증 실패
    return 401
```

## Webhook 재시도 정책

- 실패한 Webhook은 자동으로 재시도됩니다
- 재시도 간격: 5분, 15분, 1시간, 6시간, 24시간
- 5번 재시도 후에도 실패하면 Webhook이 비활성화됩니다

## Webhook 로그

**GET** `/api/v1/webhooks/{webhook_id}/logs`

Webhook 전송 로그를 조회합니다.

**Response:**
```json
{
  "logs": [
    {
      "log_id": "log_123",
      "event_type": "file.uploaded",
      "status_code": 200,
      "response_time_ms": 145,
      "triggered_at": "2024-01-16T10:30:00Z",
      "success": true
    },
    {
      "log_id": "log_124",
      "event_type": "file.deleted",
      "status_code": 500,
      "response_time_ms": 2000,
      "triggered_at": "2024-01-16T11:00:00Z",
      "success": false,
      "error": "Internal Server Error"
    }
  ]
}
```

## 모범 사례

1. **멱등성**: Webhook은 중복 전송될 수 있으므로 멱등하게 처리하세요
2. **타임아웃**: Webhook 처리는 5초 이내에 완료되어야 합니다
3. **비동기 처리**: 시간이 오래 걸리는 작업은 큐에 넣고 비동기로 처리하세요
4. **에러 처리**: 200-299 상태 코드를 반환하세요
