"""Important config manager"""

from pydantic_settings import BaseSettings
from pydantic import SecretStr, field_validator


class Settings(BaseSettings):
    # deepseek api
    deepseek_api_key: SecretStr
    deepseek_base_url: str = "https://api.deepseek.com"

    # ollama default model
    # host
    ollama_host: str = "http://192.168.0.162:11434"

    @field_validator("deepseek_api_key")
    @classmethod
    def validate_deepseek_key(cls, v: SecretStr) -> SecretStr:
        key = v.get_secret_value()
        if not key.startswith("sk-"):
            raise ValueError("DeepSeek API key must start with 'sk-'")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
# if __name__ == "__main__":
#     settings = Settings()
#     print(f"API key: {settings.deepseek_api_key.get_secret_value()}")
#     print(f"ollama host: {settings.ollama_host}")
    # print(f"Ollama model: {settings.ollama_default_model}")
