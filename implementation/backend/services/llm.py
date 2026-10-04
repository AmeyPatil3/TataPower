"""
Local & External LLM Adapter for Tata Power Platform.
Connects to OpenAI-compatible REST server:
1. LM Studio local server (default: http://localhost:1234/v1)
2. Ollama local server (http://localhost:11434/v1)
3. Groq API (if GROQ_API_KEY is present)
4. OpenAI API (if OPENAI_API_KEY is present)
Provides structured status and response handling.
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234/v1")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/v1")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

DEFAULT_MODEL = os.environ.get("LM_STUDIO_MODEL", "google/gemma-4-e2b")


class LMStudioClient:
    def __init__(self, base_url: str = LM_STUDIO_URL, default_model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self._cached_endpoint: Optional[Dict[str, Any]] = None

    def detect_active_endpoint(self, force_refresh: bool = False) -> Optional[Dict[str, str]]:
        """Detects whether LM Studio, Ollama, Groq, or OpenAI is reachable."""
        if self._cached_endpoint and not force_refresh:
            return self._cached_endpoint

        # 1. Check LM Studio
        try:
            req = urllib.request.Request(f"{self.base_url}/models")
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                if resp.status == 200:
                    self._cached_endpoint = {"type": "lm_studio", "url": self.base_url, "auth": None}
                    return self._cached_endpoint
        except Exception:
            pass

        # 2. Check Ollama
        try:
            req = urllib.request.Request(f"{OLLAMA_URL.rstrip('/')}/models")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    self._cached_endpoint = {"type": "ollama", "url": OLLAMA_URL.rstrip('/'), "auth": None}
                    return self._cached_endpoint
        except Exception:
            pass

        # 3. Check Groq API if key is present
        if GROQ_API_KEY:
            self._cached_endpoint = {"type": "groq", "url": "https://api.groq.com/openai/v1", "auth": f"Bearer {GROQ_API_KEY}"}
            return self._cached_endpoint

        # 4. Check OpenAI API if key is present
        if OPENAI_API_KEY:
            self._cached_endpoint = {"type": "openai", "url": "https://api.openai.com/v1", "auth": f"Bearer {OPENAI_API_KEY}"}
            return self._cached_endpoint

        return None

    def is_server_online(self) -> bool:
        return self.detect_active_endpoint() is not None

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500
    ) -> Dict[str, Any]:
        """
        Dispatches a chat completion request to the detected LLM service.
        """
        endpoint = self.detect_active_endpoint()
        if not endpoint:
            # Try once with forced refresh
            endpoint = self.detect_active_endpoint(force_refresh=True)

        if not endpoint:
            return {
                "status": "offline",
                "content": None,
                "model": model or self.default_model,
                "online": False,
                "error": "No local LLM (LM Studio / Ollama) or API key active."
            }

        url = f"{endpoint['url']}/chat/completions"
        target_model = model or self.default_model

        if endpoint["type"] == "groq" and "gemma" in target_model:
            target_model = "llama-3.1-8b-instant"

        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        headers = {"Content-Type": "application/json"}
        if endpoint["auth"]:
            headers["Authorization"] = endpoint["auth"]

        json_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=json_data, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=180.0) as resp:
                result = json.loads(resp.read().decode())
                choice = result["choices"][0]
                msg = choice["message"]
                content = msg.get("content") or ""
                # If content is empty because reasoning consumed tokens, check reasoning_content
                if not content.strip() and msg.get("reasoning_content"):
                    content = msg.get("reasoning_content")

                print(f"[LLM] Received natural language response from {target_model} via {endpoint['type']} ({len(content)} chars)")
                return {
                    "status": "success",
                    "content": content.strip(),
                    "model": target_model,
                    "provider": endpoint["type"],
                    "online": True
                }
        except Exception as e:
            # Invalidate cached endpoint on communication failure so next call retries
            self._cached_endpoint = None
            print(f"[LLM] Error calling {target_model}: {e}")
            return {
                "status": "error",
                "content": None,
                "model": target_model,
                "online": False,
                "error": str(e)
            }
