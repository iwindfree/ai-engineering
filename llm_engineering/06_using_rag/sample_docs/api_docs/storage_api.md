# CloudStore Storage API

## 개요

Storage API를 사용하면 프로그래밍 방식으로 파일을 업로드, 다운로드, 관리할 수 있습니다.

## 인증

모든 API 요청에는 API 키가 필요합니다. 헤더에 포함시켜 주세요:

```
Authorization: Bearer YOUR_API_KEY
```

## 엔드포인트

### 파일 업로드

**POST** `/api/v1/files/upload`

파일을 CloudStore에 업로드합니다.

**Request:**
```json
{
  "file": "<binary_data>",
  "path": "/documents/report.pdf",
  "metadata": {
    "description": "Monthly report",
    "tags": ["report", "monthly"]
  }
}
```

**Response:**
```json
{
  "file_id": "f_1234567890",
  "name": "report.pdf",
  "path": "/documents/report.pdf",
  "size": 2048576,
  "created_at": "2024-01-15T10:30:00Z",
  "download_url": "https://cloudstore.com/download/f_1234567890"
}
```

### 파일 다운로드

**GET** `/api/v1/files/{file_id}/download`

파일을 다운로드합니다.

**Response:** Binary file data

### 파일 목록 조회

**GET** `/api/v1/files`

파일 목록을 조회합니다.

**Query Parameters:**
- `path` (optional): 특정 경로의 파일만 조회
- `limit` (default: 100): 결과 개수
- `offset` (default: 0): 페이지네이션 오프셋

**Response:**
```json
{
  "files": [
    {
      "file_id": "f_1234567890",
      "name": "report.pdf",
      "path": "/documents/report.pdf",
      "size": 2048576,
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 1,
  "has_more": false
}
```

### 파일 삭제

**DELETE** `/api/v1/files/{file_id}`

파일을 삭제합니다.

**Response:**
```json
{
  "success": true,
  "message": "File deleted successfully"
}
```

### 파일 메타데이터 업데이트

**PATCH** `/api/v1/files/{file_id}/metadata`

파일의 메타데이터를 업데이트합니다.

**Request:**
```json
{
  "metadata": {
    "description": "Updated description",
    "tags": ["updated", "report"]
  }
}
```

## 에러 코드

- `400 Bad Request`: 잘못된 요청
- `401 Unauthorized`: 인증 실패
- `404 Not Found`: 파일을 찾을 수 없음
- `413 Payload Too Large`: 파일 크기 초과
- `429 Too Many Requests`: 요청 제한 초과
- `500 Internal Server Error`: 서버 오류

## 요청 제한

- Basic: 1,000 requests/day
- Pro: 10,000 requests/day
- Enterprise: 무제한
