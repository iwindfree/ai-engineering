# CloudStore 빠른 시작 가이드

## 환영합니다!

CloudStore를 선택해 주셔서 감사합니다. 이 가이드는 5분 안에 CloudStore를 시작하는 방법을 안내합니다.

## 1단계: 계정 생성

1. [CloudStore 홈페이지](https://cloudstore.com)에 접속합니다
2. "무료로 시작하기" 버튼을 클릭합니다
3. 이메일 주소와 비밀번호를 입력합니다
4. 이메일로 전송된 인증 링크를 클릭합니다

축하합니다! 이제 10GB의 무료 스토리지를 사용할 수 있습니다.

## 2단계: 첫 파일 업로드

### 웹 인터페이스 사용

1. CloudStore 대시보드에 로그인합니다
2. "파일 업로드" 버튼을 클릭합니다
3. 컴퓨터에서 파일을 선택하거나 드래그 앤 드롭합니다
4. 업로드가 완료되면 파일이 목록에 나타납니다

### 데스크톱 앱 사용

1. [데스크톱 앱 다운로드 페이지](https://cloudstore.com/download)에서 설치 프로그램을 다운로드합니다
2. 설치 프로그램을 실행하고 안내를 따릅니다
3. 로그인 후 동기화할 폴더를 선택합니다
4. 해당 폴더에 파일을 추가하면 자동으로 동기화됩니다

### 모바일 앱 사용

1. App Store (iOS) 또는 Google Play (Android)에서 CloudStore 앱을 다운로드합니다
2. 앱을 실행하고 로그인합니다
3. "+" 버튼을 탭하여 파일을 업로드합니다

## 3단계: 파일 공유

### 공유 링크 생성

1. 공유하려는 파일 옆의 메뉴 아이콘을 클릭합니다
2. "공유 링크 생성"을 선택합니다
3. 링크를 복사하여 원하는 사람과 공유합니다

### 고급 공유 옵션

- **비밀번호 보호**: 링크 접근 시 비밀번호 요구
- **만료일 설정**: 특정 날짜 이후 링크 비활성화
- **다운로드 제한**: 다운로드 횟수 제한
- **링크 추적**: 누가 언제 접근했는지 확인

## 4단계: 팀 협업 (Pro 이상)

### 팀 멤버 초대

1. 설정 > 팀으로 이동합니다
2. "멤버 초대" 버튼을 클릭합니다
3. 이메일 주소를 입력하고 역할을 선택합니다
   - **관리자**: 모든 권한
   - **편집자**: 파일 업로드, 수정, 삭제 가능
   - **뷰어**: 파일 보기 및 다운로드만 가능
4. 초대 이메일이 전송됩니다

### 팀 폴더 생성

1. "새 폴더" 버튼을 클릭합니다
2. 폴더 이름을 입력하고 "팀 폴더로 설정"을 체크합니다
3. 팀 멤버에게 자동으로 접근 권한이 부여됩니다

## 5단계: API 사용 (개발자용)

### API 키 생성

1. 설정 > 개발자로 이동합니다
2. "API 키 생성" 버튼을 클릭합니다
3. 키 이름을 입력하고 권한을 선택합니다
4. API 키를 안전한 곳에 복사합니다

### Python으로 파일 업로드

```python
import requests

API_KEY = "your_api_key_here"
FILE_PATH = "document.pdf"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

with open(FILE_PATH, "rb") as f:
    files = {"file": f}
    data = {"path": "/documents/document.pdf"}

    response = requests.post(
        "https://api.cloudstore.com/v1/files/upload",
        headers=headers,
        files=files,
        data=data
    )

print(response.json())
```

### JavaScript로 파일 업로드

```javascript
const apiKey = 'your_api_key_here';
const fileInput = document.querySelector('input[type="file"]');
const file = fileInput.files[0];

const formData = new FormData();
formData.append('file', file);
formData.append('path', '/documents/' + file.name);

fetch('https://api.cloudstore.com/v1/files/upload', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${apiKey}`
  },
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## 다음 단계

- [보안 가이드](security.md)를 읽고 계정을 보호하세요
- [요금제 가이드](pricing.md)를 확인하고 필요에 맞는 플랜을 선택하세요
- [API 문서](../api_docs/)를 참고하여 자동화를 구현하세요

## 문제 해결

### 파일이 업로드되지 않아요

- 파일 크기를 확인하세요 (Basic: 최대 2GB)
- 네트워크 연결을 확인하세요
- 브라우저 캐시를 삭제하고 다시 시도하세요

### 동기화가 안 돼요

- 데스크톱 앱이 최신 버전인지 확인하세요
- 로그아웃 후 다시 로그인하세요
- 동기화 폴더 경로가 올바른지 확인하세요

### 공유 링크가 작동하지 않아요

- 링크가 만료되지 않았는지 확인하세요
- 비밀번호가 설정되어 있는지 확인하세요
- 링크를 새로 생성해 보세요

## 지원

- 이메일: support@cloudstore.com
- 라이브 채팅: 웹사이트 우측 하단
- 커뮤니티 포럼: https://community.cloudstore.com
- 도움말 센터: https://help.cloudstore.com
