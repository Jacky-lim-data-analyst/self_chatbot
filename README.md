```
backend/
├── .env
├── .gitignore
├── .python-version
├── README.md
├── chatbot.db
├── pyproject.toml
├── pytest.ini
├── quick_test.py
├── settings.py
├── uv.lock
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   └── __init__.py
│   ├── router/
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   ├── conversation.py
│   │   └── health.py
│   ├── service/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── llm/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── deepseek.py
│   │       ├── factory.py
│   │       ├── ollama.py
│   │       ├── retry.py
│   │       ├── test_deepseek.py
│   │       └── test_ollama.py
│   └── util/
│       ├── __init__.py
│       └── logging.py
└── test/
    ├── conftest.py
    ├── test_chat.py
    ├── test_conversation.py
    └── test_health.py
```