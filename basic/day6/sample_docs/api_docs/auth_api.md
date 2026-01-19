# CloudStore Authentication API

## 개요

Authentication API를 사용하여 사용자 인증, 토큰 관리, 권한 확인을 수행할 수 있습니다.

## 인증 방식

CloudStore는 다음 인증 방식을 지원합니다:

1. **API 키**: 서버 간 통신에 사용
2. **OAuth 2.0**: 사용자 인증에 사용
3. **JWT 토큰**: 세션 관리에 사용

## 엔드포인트

### API 키 생성

**POST** `/api/v1/auth/api-keys`

새 API 키를 생성합니다.

**Request:**
```json
{
  "name": "My Application",
  "permissions": ["read", "write"],
  "expires_in_days": 90
}
```

**Response:**
```json
{
  "api_key": "sk_live_1234567890abcdef",
  "name": "My Application",
  "created_at": "2024-01-15T10:30:00Z",
  "expires_at": "2024-04-15T10:30:00Z",
  "permissions": ["read", "write"]
}
```

⚠️ **중요**: API 키는 생성 시 한 번만 표시됩니다. 안전한 곳에 보관하세요.

### API 키 목록 조회

**GET** `/api/v1/auth/api-keys`

생성한 API 키 목록을 조회합니다.

**Response:**
```json
{
  "api_keys": [
    {
      "key_id": "key_123",
      "name": "My Application",
      "prefix": "sk_live_1234",
      "created_at": "2024-01-15T10:30:00Z",
      "expires_at": "2024-04-15T10:30:00Z",
      "last_used_at": "2024-01-16T08:20:00Z"
    }
  ]
}
```

### API 키 삭제

**DELETE** `/api/v1/auth/api-keys/{key_id}`

API 키를 삭제합니다.

**Response:**
```json
{
  "success": true,
  "message": "API key revoked successfully"
}
```

### OAuth 2.0 인증

**POST** `/api/v1/auth/oauth/authorize`

OAuth 2.0 인증 플로우를 시작합니다.

**Request:**
```json
{
  "client_id": "your_client_id",
  "redirect_uri": "https://your-app.com/callback",
  "scope": "files:read files:write",
  "state": "random_state_string"
}
```

**Response:**
```json
{
  "authorization_url": "https://cloudstore.com/oauth/authorize?client_id=...",
  "state": "random_state_string"
}
```

### 토큰 교환

**POST** `/api/v1/auth/oauth/token`

인증 코드를 액세스 토큰으로 교환합니다.

**Request:**
```json
{
  "grant_type": "authorization_code",
  "code": "auth_code_from_callback",
  "client_id": "your_client_id",
  "client_secret": "your_client_secret",
  "redirect_uri": "https://your-app.com/callback"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "refresh_token_string",
  "scope": "files:read files:write"
}
```

### 토큰 갱신

**POST** `/api/v1/auth/oauth/refresh`

리프레시 토큰을 사용하여 새 액세스 토큰을 발급받습니다.

**Request:**
```json
{
  "grant_type": "refresh_token",
  "refresh_token": "refresh_token_string",
  "client_id": "your_client_id",
  "client_secret": "your_client_secret"
}
```

**Response:**
```json
{
  "access_token": "new_access_token",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

### 현재 사용자 정보

**GET** `/api/v1/auth/me`

현재 인증된 사용자의 정보를 조회합니다.

**Response:**
```json
{
  "user_id": "user_123",
  "email": "user@example.com",
  "name": "John Doe",
  "plan": "pro",
  "storage_used": 52428800,
  "storage_quota": 1099511627776
}
```

## 보안 모범 사례

1. **API 키 보안**
   - API 키를 코드에 직접 포함하지 마세요
   - 환경 변수나 시크릿 관리 서비스를 사용하세요
   - 정기적으로 API 키를 교체하세요

2. **OAuth 2.0**
   - `state` 매개변수를 사용하여 CSRF 공격을 방지하세요
   - HTTPS를 사용하세요
   - 리프레시 토큰을 안전하게 저장하세요

3. **권한 관리**
   - 최소 권한 원칙을 따르세요
   - 필요한 권한만 요청하세요
