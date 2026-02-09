# 교육 환경 설정

AI Engineering 실습을 위한 Python 개발 환경을 구성합니다.

## 구성 순서

1. **프로젝트 Clone** - GitHub에서 교육 자료 받기
2. **pyenv 설치** - Python 버전 관리 도구
3. **Python 3.13.x 설치** - pyenv를 통한 설치
4. **venv 가상환경 구성** - 프로젝트별 독립 환경
5. **uv를 통한 패키지 설치** - 교육에 필요한 패키지 설치
6. **VS Code + Jupyter 환경 구성** - 개발 환경 설정
7. **Ollama 설치** - 로컬 LLM 실행 환경
8. **OpenAI API 설정** - API Key 발급 및 크레딧 충전
9. **Hugging Face 설정** - Access Token 발급 및 로그인

---

## 1. 프로젝트 Clone

교육에 사용할 프로젝트를 GitHub에서 clone 받습니다.

### Git 설치 확인

```bash
git --version
```

Git이 설치되어 있지 않다면:
- **Mac**: `brew install git`
- **Windows**: https://git-scm.com 에서 다운로드

### 프로젝트 Clone

```bash
git clone https://github.com/iwindfree/ai-engineering.git
```

### 폴더 이동

```bash
cd ai-engineering
```

---

## 2. pyenv 설치

pyenv는 여러 Python 버전을 쉽게 설치하고 전환할 수 있게 해주는 도구입니다.

### Mac (Homebrew)

#### 1) Homebrew로 pyenv 설치

```bash
brew install pyenv
```

#### 2) 쉘 설정 추가

사용하는 쉘에 따라 설정 파일을 수정합니다.

**zsh 사용자** (~/.zshrc에 추가):

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```

**bash 사용자** (~/.bashrc에 추가):

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```

#### 3) 쉘 재시작

```bash
exec "$SHELL"
```

#### 4) 설치 확인

```bash
pyenv --version
```

### Windows (pyenv-win)

#### 1) PowerShell에서 pyenv-win 설치

PowerShell을 **관리자 권한**으로 실행 후:

```powershell
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
```

#### 2) 환경 변수 확인

설치 스크립트가 자동으로 환경 변수를 설정합니다. 수동 설정이 필요한 경우:

```powershell
[System.Environment]::SetEnvironmentVariable('PYENV', $env:USERPROFILE + "\.pyenv\pyenv-win", "User")
[System.Environment]::SetEnvironmentVariable('PYENV_HOME', $env:USERPROFILE + "\.pyenv\pyenv-win", "User")
[System.Environment]::SetEnvironmentVariable('path', $env:USERPROFILE + "\.pyenv\pyenv-win\bin;" + $env:USERPROFILE + "\.pyenv\pyenv-win\shims;" + [System.Environment]::GetEnvironmentVariable('path', "User"), "User")
```

#### 3) PowerShell 재시작 후 확인

```powershell
pyenv --version
```

---

## 3. 설치 가능한 Python 버전 확인

pyenv로 설치할 수 있는 Python 버전 목록을 확인합니다.

### 전체 목록 확인

```bash
pyenv install --list
```

### 3.13.x 버전만 필터링

**Mac/Linux:**

```bash
pyenv install --list | grep "3.13"
```

**Windows PowerShell:**

```powershell
pyenv install --list | Select-String "3.13"
```

출력 예시:

```
  3.13.0
  3.13.1
  3.13.2
```

> 최신 안정 버전을 선택합니다 (예: 3.13.2)

---

## 4. Python 3.13.x 설치

### Python 설치

```bash
pyenv install 3.13.2
```

> 버전 번호는 위에서 확인한 최신 버전으로 변경하세요.

### 설치된 버전 확인

```bash
pyenv versions
```

### 기본 Python 버전 설정

**시스템 전역 설정:**

```bash
pyenv global 3.13.2
```

**특정 폴더에서만 사용 (권장):**

```bash
cd ai-engineering
pyenv local 3.13.2
```

`pyenv local`은 해당 폴더에 `.python-version` 파일을 생성하여 해당 폴더 진입 시 자동으로 버전이 전환됩니다.

### 설정 확인

```bash
python --version
```

출력: `Python 3.13.2`

---

## 5. venv 가상환경 구성

가상환경은 프로젝트별로 독립된 Python 패키지 환경을 제공합니다.

### 가상환경 생성

ai-engineering 폴더에서 실행:

```bash
cd ai-engineering
python -m venv .venv
```

`.venv` 폴더가 생성됩니다.

### 가상환경 활성화

**Mac/Linux:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
.venv\Scripts\activate.bat
```

활성화되면 프롬프트 앞에 `(.venv)`가 표시됩니다:

```
(.venv) user@machine:~/ai-engineering$
```

### 가상환경 비활성화

```bash
deactivate
```

---

## 6. uv를 통한 패키지 설치

uv는 Rust로 작성된 빠른 Python 패키지 관리자입니다. `uv sync` 명령으로 프로젝트에 필요한 패키지를 한 번에 설치합니다.

### uv 설치

**Mac/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**pip으로 설치 (대안):**

```bash
pip install uv
```

### 설치 확인

```bash
uv --version
```

### 패키지 설치

ai-engineering 폴더에서 가상환경이 활성화된 상태로 실행:

```bash
uv sync
```

이 명령은 프로젝트의 `pyproject.toml` 또는 `uv.lock` 파일을 읽어 필요한 패키지를 자동으로 설치합니다.

> uv는 pip보다 빠르게 패키지를 설치합니다.

---

## 7. VS Code에서 Jupyter Notebook 환경 구성

Visual Studio Code에서 프로젝트를 열고 Jupyter Notebook을 실행하는 방법입니다.

### 1) VS Code 설치

https://code.visualstudio.com/ 에서 운영체제에 맞는 버전을 다운로드하여 설치합니다.

### 2) 프로젝트 폴더 열기

**방법 1: 터미널에서 열기**

```bash
cd ai-engineering
code .
```

**방법 2: VS Code에서 열기**

1. VS Code 실행
2. **File** → **Open Folder** (Mac: **File** → **Open...**)
3. `ai-engineering` 폴더 선택

### 3) Python 확장 설치

1. 좌측 사이드바에서 **Extensions** 아이콘 클릭 (또는 `Ctrl+Shift+X` / `Cmd+Shift+X`)
2. 검색창에 `Python` 입력
3. **Python** (Microsoft) 확장 설치
4. **Jupyter** (Microsoft) 확장 설치

### 4) Python 인터프리터 선택

1. `Ctrl+Shift+P` (Mac: `Cmd+Shift+P`)로 Command Palette 열기
2. `Python: Select Interpreter` 입력 후 선택
3. `.venv` 가상환경의 Python 선택
   - Mac/Linux: `./.venv/bin/python`
   - Windows: `.\.venv\Scripts\python.exe`

> 하단 상태바에서 현재 선택된 Python 버전을 확인할 수 있습니다.

### 5) Jupyter Notebook 실행

1. `.ipynb` 파일 클릭하여 열기 (예: `basic/1.llm_token_basic.ipynb`)
2. 우측 상단의 **Select Kernel** 클릭
3. **Python Environments** → `.venv` 가상환경 선택
4. 셀 실행: `Shift+Enter` 또는 셀 좌측의 ▶ 버튼 클릭

### 6) 새 노트북 생성

1. `Ctrl+Shift+P` (Mac: `Cmd+Shift+P`)
2. `Create: New Jupyter Notebook` 입력 후 선택
3. 커널 선택 시 `.venv` 가상환경 선택

### 유용한 단축키

| 단축키 | 기능 |
|--------|------|
| `Shift+Enter` | 현재 셀 실행 후 다음 셀로 이동 |
| `Ctrl+Enter` | 현재 셀 실행 (이동 없음) |
| `A` | 위에 새 셀 추가 (명령 모드) |
| `B` | 아래에 새 셀 추가 (명령 모드) |
| `M` | 마크다운 셀로 변환 |
| `Y` | 코드 셀로 변환 |
| `DD` | 셀 삭제 |

---

## 8. Ollama 설치 (로컬 LLM)

**Ollama**는 로컬 환경에서 대규모 언어 모델(LLM)을 간단하게 실행할 수 있도록 도와주는 도구입니다.

### 주요 특징

| 특징 | 설명 |
|------|------|
| 로컬 실행 | 인터넷 연결 없이도 모델 실행 가능 |
| 간단한 설치 | 별도 CUDA 설정 없이 바로 사용 가능 |
| 모델 관리 | 다운로드·버전 관리가 매우 쉬움 |
| API 제공 | REST API 형태로 외부 프로그램과 연동 가능 |
| 다양한 모델 | LLaMA, Mistral, Gemma, Qwen 등 지원 |

> OpenAI API를 쓰는 구조와 유사하게 로컬 LLM 서버를 띄운다고 생각하면 됩니다.

### 지원 운영체제

- macOS (Apple Silicon / Intel)
- Linux
- Windows

### 설치 방법

1. 공식 사이트 접속: https://ollama.com
2. 운영체제에 맞는 설치 파일 다운로드 후 설치

### 설치 확인

```bash
ollama --version
```

### 기본 사용법

| 명령어 | 설명 |
|--------|------|
| `ollama run llama3` | 모델 실행 (없으면 자동 다운로드) |
| `ollama pull llama3` | 모델 미리 다운로드 |
| `ollama list` | 설치된 모델 확인 |
| `ollama rm llama3` | 모델 삭제 |

### 모델 다운로드 예시

```bash
# Llama 3 (8B) 다운로드
ollama pull llama3

# Qwen 2.5 (7B) 다운로드
ollama pull qwen2.5
```

> 모델 크기에 따라 다운로드 시간이 다릅니다. llama3(8B)는 약 4.7GB입니다.

---

## 9. OpenAI API 설정

OpenAI API를 사용하려면 API 키가 필요합니다. API는 사용량에 따라 과금되며, 최소 $5부터 충전할 수 있습니다.

### 1) 회원가입 및 로그인

1. https://platform.openai.com/ 접속
2. 우측 상단 **Sign up** 클릭하여 회원가입 (또는 **Log in**)
3. 이메일 인증 완료

### 2) 결제 수단 등록 및 크레딧 충전

1. 로그인 후 좌측 메뉴에서 **Settings** → **Billing** 클릭
2. **Add payment details** 클릭하여 결제 수단(신용카드) 등록
3. **Add to credit balance** 클릭
4. 충전 금액 입력 (최소 $5)
5. 결제 완료

> 교육 실습에는 $5~10 정도면 충분합니다. GPT-4o-mini 기준 100만 토큰당 약 $0.15입니다.

### ⚠️ 자동 충전(Auto recharge) 비활성화

**Settings** → **Billing** → **Auto recharge** 옵션이 **OFF**인지 반드시 확인하세요.

자동 충전이 켜져 있으면 잔액이 일정 금액 이하로 떨어질 때 자동으로 결제됩니다. 예상치 못한 과금을 방지하려면 반드시 **비활성화** 상태를 유지하세요.

### 3) API Key 생성

1. 좌측 메뉴에서 **API keys** 클릭
2. **Create new secret key** 클릭
3. 이름 입력 후 **Create secret key** 클릭
4. 생성된 키를 **즉시 복사하여 안전한 곳에 저장**

> ⚠️ API Key는 생성 시 한 번만 표시됩니다. 분실 시 새로 생성해야 합니다.

### 4) 환경 변수 설정

프로젝트 폴더에 `.env` 파일을 생성하고 API Key를 저장합니다:

```bash
# ai-engineering/.env 파일 생성
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

> `.env` 파일은 `.gitignore`에 포함되어 있어 GitHub에 업로드되지 않습니다.

### 5) API Key 테스트

```python
from openai import OpenAI

client = OpenAI()  # 환경변수에서 자동으로 API Key를 읽음
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

**예상 출력:**
```
Hello! How can I assist you today?
```

---

## 10. Hugging Face 설정

**Hugging Face**는 사전 학습된 AI 모델과 데이터셋을 공유하는 플랫폼입니다. Hub에서 비공개 모델을 사용하거나, 모델을 업로드하려면 Access Token이 필요합니다.

### 1) 회원가입 및 로그인

1. https://huggingface.co 접속
2. 우측 상단 **Sign Up** 클릭하여 회원가입
3. 이메일 인증 완료 후 로그인

### 2) Access Token 생성

1. 로그인 후 우측 상단 프로필 아이콘 클릭
2. **Settings** 선택
3. 좌측 메뉴에서 **Access Tokens** 클릭
4. **New token** 버튼 클릭
5. Token 정보 입력:
   - **Name**: 토큰 이름 (예: `ai-engineering-course`)
   - **Type**: **Read** 또는 **Write** 선택
     - **Read**: 모델 다운로드 및 추론만 가능
     - **Write**: 모델 업로드 및 수정 가능 (일반적으로 Read면 충분)
6. **Generate a token** 클릭
7. 생성된 토큰을 **즉시 복사하여 안전한 곳에 저장**

> ⚠️ Access Token은 생성 시 한 번만 표시됩니다. 분실 시 새로 생성해야 합니다.

### 3) 환경 변수 설정

프로젝트 폴더의 `.env` 파일에 Access Token을 추가합니다:

```bash
# ai-engineering/.env 파일에 추가
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

또는 별도로 관리:

```bash
# Hugging Face Token만
HUGGING_FACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> `.env` 파일은 `.gitignore`에 포함되어 있어 GitHub에 업로드되지 않습니다.

### 4) Python에서 토큰 사용

**방법 1: 환경 변수 사용 (권장)**

```python
import os
from huggingface_hub import login

# .env 파일에서 자동으로 로드 (python-dotenv 사용 시)
from dotenv import load_dotenv
load_dotenv()

# 토큰으로 로그인
token = os.getenv("HF_TOKEN")
login(token=token)

print("✅ Hugging Face 로그인 완료!")
```

**방법 2: 직접 입력 (로컬 개발용)**

```python
from huggingface_hub import login

# 대화형으로 토큰 입력
login()
# 또는 토큰 직접 전달
# login(token="hf_xxxxxxxxxxxx")
```

### 언제 Access Token이 필요한가?

| 작업 | Token 필요 여부 |
|------|----------------|
| 공개 모델 다운로드 및 추론 | ❌ 불필요 |
| 비공개(private) 모델 접근 | ✅ 필요 (Read) |
| Gated 모델 접근 (예: LLaMA) | ✅ 필요 (Read + 승인) |
| 모델 업로드 | ✅ 필요 (Write) |
| 데이터셋 업로드 | ✅ 필요 (Write) |
| Spaces 배포 | ✅ 필요 (Write) |

> 대부분의 교육 실습에서는 공개 모델만 사용하므로 토큰이 필수는 아니지만, 사전에 설정해두면 편리합니다.

---

## 요약

| 단계 | 명령어/URL |
|------|--------|
| 프로젝트 Clone | `git clone https://github.com/iwindfree/ai-engineering.git` |
| pyenv 설치 | `brew install pyenv` (Mac) |
| 버전 목록 확인 | `pyenv install --list` |
| Python 설치 | `pyenv install 3.13.2` |
| 버전 설정 | `pyenv local 3.13.2` |
| 가상환경 생성 | `python -m venv .venv` |
| 가상환경 활성화 | `source .venv/bin/activate` (Mac) |
| uv 설치 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| 패키지 설치 | `uv sync` |
| VS Code 설치 | https://code.visualstudio.com |
| VS Code 확장 | Python, Jupyter 확장 설치 |
| Ollama 설치 | https://ollama.com |
| 모델 다운로드 | `ollama pull llama3` |
| OpenAI API | https://platform.openai.com → Billing → $5 충전 |
| Hugging Face | https://huggingface.co → Settings → Access Tokens |
