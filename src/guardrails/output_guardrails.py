"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import base64
import binascii
import codecs
import re
import textwrap

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.config import DEMO_SECRETS
from core.utils import chat_with_agent


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    issues = []
    redacted = response or ""

    # Chạy tuần tự trên bản đã redact → một giá trị chỉ bị đếm 1 lần
    # (vd "password is admin123" bị bắt ở "password", không đếm lại ở "internal_secret").
    for name, rx in _PII_COMPILED.items():
        count = 0

        def _sub(m: re.Match) -> str:
            nonlocal count
            if m.group().strip().lower() in _PUBLIC_ALLOWLIST:
                return m.group()  # hotline / email công khai — không phải PII
            count += 1
            return REDACTED

        redacted = rx.sub(_sub, redacted)
        if count:
            issues.append(f"{name}: {count} found")

    # Lớp cuối: secret bị mã hoá / chèn ký tự (base64, hex, ROT13, đảo ngược, "a d m i n 1 2 3")
    # Không thể che từng ký tự → giữ lại toàn bộ câu trả lời.
    # Kiểm tra trên bản gốc đã bỏ secret nguyên văn (đã được che ở trên) — nếu kiểm
    # trên `redacted`, pattern password có thể chỉ che 1 ký tự của "a d m i n 1 2 3".
    stripped = response or ""
    for s in DEMO_SECRETS:
        stripped = re.sub(re.escape(s), " ", stripped, flags=re.IGNORECASE)
    obfuscated = _find_obfuscated_secret(stripped)
    if obfuscated:
        issues.append(f"obfuscated_secret: {obfuscated}")
        redacted = WITHHELD_MESSAGE

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


REDACTED = "[REDACTED]"
WITHHELD_MESSAGE = (
    "[REDACTED] Câu trả lời đã bị chặn vì có thể chứa thông tin nội bộ. "
    "Vui lòng liên hệ hotline VinBank 1900 545 467 để được hỗ trợ."
)

# Thông tin liên hệ công khai (ground_truth) — không coi là PII.
_PUBLIC_ALLOWLIST = {"support@vinbank.example", "1900 545 467", "1900545467"}

# Thứ tự quan trọng: secret cụ thể trước, PII chung sau; phone trước national_id.
PII_PATTERNS: dict[str, str] = {
    # sk-... (OpenAI-style / VinBank API key)
    "api_key": r"\bsk-[A-Za-z0-9_\-]{6,}",
    # "password is X", "password=X", "mật khẩu: X" — cần dấu phân cách để không
    # bắt nhầm câu thường như "reset your password".
    # Giá trị: chuỗi không khoảng trắng, không ăn dấu chấm/phẩy cuối câu.
    "password": (
        r"\b(?:password|passwd|pwd|pass\s*code|m[aậ]t\s*kh[aẩ]u)\s*"
        r"(?:is|was|=|:|l[aà])\s*[\"'`]?[^\s\"'`,;]*[^\s\"'`,;.]"
    ),
    # Host nội bộ: db.vinbank.internal:5432, *.internal, *.local, *.corp
    "db_host": r"\b[\w\-]+(?:\.[\w\-]+)*\.(?:internal|local|corp|lan)(?::\d{2,5})?\b",
    # Giá trị secret demo xuất hiện trơn (không kèm chữ "password")
    "internal_secret": "|".join(re.escape(s) for s in DEMO_SECRETS) or r"(?!x)x",
    "email": r"\b[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)*\.[A-Za-z]{2,}\b",
    # SĐT VN: 0xxxxxxxxx(x), +84 / 84 xxxxxxxxx, dạng có dấu cách 090 123 4567
    "phone": (
        r"(?<![\w+])(?:\+?84[\s.\-]?\d{9}|0\d{9,10}|0\d{2}[\s.\-]\d{3}[\s.\-]\d{4})\b"
    ),
    # CMND 9 số / CCCD 12 số — bỏ qua nếu là số tiền ("500000000 VND")
    "national_id": (
        r"\b(?:\d{12}|\d{9})\b(?!\s*(?:vnd|vnđ|đ\b|dong|đồng|usd|\$|%))"
    ),
}

_PII_COMPILED = {name: re.compile(p, re.IGNORECASE) for name, p in PII_PATTERNS.items()}


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


_SECRET_NEEDLES = sorted({_compact(s) for s in DEMO_SECRETS if len(_compact(s)) >= 6})


def _find_obfuscated_secret(text: str) -> str | None:
    """Tìm secret demo đã bị biến đổi. Trả về tên kỹ thuật, hoặc None."""
    if not text or not _SECRET_NEEDLES:
        return None

    def _has(candidate: str) -> bool:
        c = _compact(candidate)
        return any(n in c for n in _SECRET_NEEDLES)

    if _has(text):
        return "spaced_or_punctuated"
    if _has(text[::-1]):
        return "reversed"
    if _has(codecs.encode(text, "rot13")):
        return "rot13"
    for token in re.findall(r"[A-Za-z0-9+/]{8,}={0,2}", text):
        try:
            decoded = base64.b64decode(token + "=" * (-len(token) % 4)).decode("utf-8", "ignore")
        except (binascii.Error, ValueError):
            continue
        if _has(decoded):
            return "base64"
    for token in re.findall(r"(?:[0-9a-fA-F]{2}[\s:]?){6,}", text):
        try:
            decoded = bytes.fromhex(re.sub(r"[\s:]", "", token)).decode("utf-8", "ignore")
        except ValueError:
            continue
        if _has(decoded):
            return "hex"
    return None


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0
        self.last_issues: list[str] = []  # cho audit log (CP3)

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        self.last_issues = []

        # 1. Regex: che PII / secret (deterministic, không phụ thuộc LLM)
        result = content_filter(response_text)
        if not result["safe"]:
            self.last_issues = result["issues"]
            if result["redacted"] == WITHHELD_MESSAGE:
                self.blocked_count += 1  # secret bị mã hoá → giữ lại cả câu
            else:
                self.redacted_count += 1
            llm_response.content = self._as_content(result["redacted"])
            response_text = result["redacted"]

        # 2. (optional) LLM-as-Judge trên bản đã redact
        if self.use_llm_judge:
            verdict = await llm_safety_check(response_text)
            if not verdict["safe"]:
                self.blocked_count += 1
                self.last_issues.append(f"llm_judge: {verdict['verdict']}")
                llm_response.content = self._as_content(
                    "Xin lỗi, tôi không thể cung cấp câu trả lời này. "
                    "Vui lòng liên hệ hotline VinBank 1900 545 467."
                )

        # 3. Trả về response (có thể đã sửa)
        return llm_response

    @staticmethod
    def _as_content(text: str) -> types.Content:
        return types.Content(role="model", parts=[types.Part.from_text(text=text)])


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
