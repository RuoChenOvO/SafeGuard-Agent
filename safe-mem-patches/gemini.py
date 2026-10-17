import base64
import json
import mimetypes
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_BASE_URL = "https://c-z0-api-01.hash070.com/v1"
DEFAULT_MODEL = "gemini-2.5-flash-lite"
load_dotenv(PROJECT_ROOT / ".env")


def _get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set; add it to scene_graph/.env or the environment"
        )
    return api_key


def _get_base_url():
    return os.getenv("GEMINI_BASE_URL", DEFAULT_BASE_URL).strip()


def _get_model():
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip()


def _read_prompt(objs_prompt_path=None):
    base_prompt = (PROMPT_DIR / "generate_frame_scene_graph.txt").read_text(
        encoding="utf-8"
    )
    if objs_prompt_path:
        obj_prompt_path = Path(objs_prompt_path)
        if obj_prompt_path.is_file():
            obj_prompt = obj_prompt_path.read_text(encoding="utf-8")
            return f"{obj_prompt}\n\n{base_prompt}"
    return base_prompt


def _parse_json_response(text):
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def _generate_via_openai_compatible(image_path, prompt):
    image_path = Path(image_path)
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    timeout = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "90"))
    client = OpenAI(
        base_url=_get_base_url(),
        api_key=_get_api_key(),
        timeout=timeout,
        max_retries=1,
    )
    started = time.perf_counter()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded}",
                    },
                },
            ],
        }
    ]
    response = None
    last_error = None
    for attempt in range(5):
        try:
            response = client.chat.completions.create(model=_get_model(), messages=messages)
            break
        except Exception as e:
            last_error = e
            print(f"Gemini call attempt {attempt+1}/5 failed: {type(e).__name__}: {e}")
            if attempt < 4:
                time.sleep(30)
    if response is None:
        raise last_error
    usage = response.usage
    print(f"Gemini latency: {time.perf_counter() - started:.2f}s")
    if usage is not None:
        print(
            "Gemini tokens:",
            f"input={usage.prompt_tokens}",
            f"output={usage.completion_tokens}",
            f"total={usage.total_tokens}",
        )
    return response.choices[0].message.content or ""


def _generate_via_google(image_path, prompt):
    client = genai.Client(api_key=_get_api_key())
    response = client.models.generate_content(
        model=_get_model(),
        contents=[Image.open(image_path), prompt],
    )
    return response.text or ""


def generate_frame_scene_graph(image_path, objs_prompt_path=None):
    prompt = _read_prompt(objs_prompt_path)
    if _get_base_url():
        text = _generate_via_openai_compatible(image_path, prompt)
    else:
        text = _generate_via_google(image_path, prompt)
    return _parse_json_response(text)
