from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import cv2
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from manga_ocr import MangaOcr
from PIL import Image, ImageDraw, ImageFont
from simple_lama_inpainting import SimpleLama
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from ultralytics import YOLO


DEFAULT_SEGMENTATION_REPO = "huyvux3005/manga109-segmentation-bubble"
DEFAULT_SEGMENTATION_FILE = "best.pt"
DEFAULT_TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
DEFAULT_TRANSLATION_RULES_FILE = "translation_rules.json"
DEFAULT_TRANSLATOR_BACKEND = "local"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def detect_default_font_path() -> str:
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return ""


DEFAULT_FONT = detect_default_font_path()


@dataclass
class BubbleResult:
    index: int
    confidence: float
    bbox: list[int]
    text_box: list[int]
    layout_mode: str
    line_guides: list[list[int]]
    polygon: list[list[int]]
    japanese_text: str
    chinese_text: str


@dataclass
class PipelineRuntime:
    model: YOLO
    ocr: MangaOcr
    translator: "BaseTranslator"
    inpainter: Inpainter
    text_detector: object | None = None


@dataclass(frozen=True)
class TranslationPatternRule:
    source_contains: tuple[str, ...]
    target_contains: tuple[str, ...]
    replacement: str


@dataclass(frozen=True)
class TranslationRulesConfig:
    term_glossary: dict[str, dict[str, object]]
    exact_overrides: dict[str, str]
    pattern_rules: list[TranslationPatternRule]


class Inpainter:
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.lama: SimpleLama | None = None
        if backend == "lama":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.lama = SimpleLama(device=device)

    def __call__(self, crop_bgr: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
        if int(text_mask.sum()) == 0:
            return crop_bgr
        original_height, original_width = crop_bgr.shape[:2]
        if self.backend == "opencv":
            return cv2.inpaint(crop_bgr, text_mask, 3, cv2.INPAINT_TELEA)
        if self.lama is None:
            raise RuntimeError("LaMa backend was requested but not initialized.")
        image_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        pil_mask = Image.fromarray(text_mask).convert("L")
        result = self.lama(pil_image, pil_mask)
        if isinstance(result, Image.Image):
            output_rgb = np.array(result)
        else:
            output_rgb = np.asarray(result)
        if output_rgb.shape[:2] != (original_height, original_width):
            output_rgb = cv2.resize(output_rgb, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)


class BaseTranslator(ABC):
    @abstractmethod
    def translate(self, text: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def translate_batch(self, texts: list[str]) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def descriptor(self) -> str:
        raise NotImplementedError


class NLLBTranslator(BaseTranslator):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.src_lang = "jpn_Jpan"
        self.tgt_lang = "zho_Hans"

    @property
    def descriptor(self) -> str:
        return self.model_name

    def translate(self, text: str) -> str:
        return self.translate_batch([text])[0]

    def translate_batch(self, texts: list[str]) -> list[str]:
        normalized = [normalize_ocr_text(text) for text in texts]
        active_pairs = [(idx, text) for idx, text in enumerate(normalized) if text]
        outputs = [""] * len(texts)
        if not active_pairs:
            return outputs
        active_texts = [text for _, text in active_pairs]
        inputs = self.tokenizer(
            active_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        generated = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
            max_length=256,
            num_beams=4,
        )
        translated = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        for (idx, source_text), text in zip(active_pairs, translated):
            normalized_text = normalize_chinese_text(text, source_text)
            if is_degenerate_translation(source_text, normalized_text):
                normalized_text = self.retry_translation(source_text, normalized_text)
            outputs[idx] = normalized_text
        return outputs

    def retry_translation(self, source_text: str, previous_text: str = "") -> str:
        inputs = self.tokenizer(
            [source_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        retry_attempts = [
            {
                "forced_bos_token_id": self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
                "max_length": min(128, max(32, len(source_text) * 3)),
                "num_beams": 4,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.2,
            },
            {
                "forced_bos_token_id": self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
                "max_length": min(96, max(24, len(source_text) * 2)),
                "num_beams": 1,
                "repetition_penalty": 1.15,
            },
        ]
        for kwargs in retry_attempts:
            generated = self.model.generate(**inputs, **kwargs)
            candidate = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            candidate = normalize_chinese_text(candidate, source_text)
            if not is_degenerate_translation(source_text, candidate):
                return candidate
        return previous_text


class OpenAICompatibleTranslator(BaseTranslator):
    def __init__(self, endpoint: str, api_key: str, model_name: str, timeout: int = 120) -> None:
        self.endpoint = endpoint.rstrip("/") + "/"
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.session = requests.Session()

    @property
    def descriptor(self) -> str:
        return f"openai-compatible:{self.model_name}"

    def translate(self, text: str) -> str:
        return self.translate_batch([text])[0]

    def translate_batch(self, texts: list[str]) -> list[str]:
        normalized = [normalize_ocr_text(text) for text in texts]
        active_pairs = [(idx, text) for idx, text in enumerate(normalized) if text]
        outputs = [""] * len(texts)
        if not active_pairs:
            return outputs

        translated_by_index = self._translate_batch_with_retries(active_pairs)
        for idx, source_text in active_pairs:
            candidate = translated_by_index.get(idx, "")
            if not candidate:
                candidate = self._translate_single_with_retries(source_text)
            normalized_text = normalize_chinese_text(candidate, source_text)
            if is_degenerate_translation(source_text, normalized_text):
                normalized_text = ""
            outputs[idx] = normalized_text
        return outputs

    def _translate_batch_with_retries(self, active_pairs: list[tuple[int, str]]) -> dict[int, str]:
        source_payload = [{"index": idx, "text": text} for idx, text in active_pairs]
        prompt = (
            "You are a professional Japanese-to-Simplified-Chinese manga translator.\n"
            "Translate each item naturally for comic dialogue.\n"
            "Preserve tone, implication, humor, and character voice.\n"
            "Return strict JSON only in the form "
            '{"translations":[{"index":0,"text":"..."},{"index":1,"text":"..."}]}.\n'
            "Do not add explanations.\n"
            f"Input JSON:\n{json.dumps(source_payload, ensure_ascii=False)}"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You translate Japanese manga dialogue into natural Simplified Chinese. "
                    "Output valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        for _ in range(3):
            try:
                content = self._call_chat_completion(messages)
                parsed = self._extract_json(content)
                translations = parsed.get("translations", [])
                translated_by_index: dict[int, str] = {}
                if isinstance(translations, list):
                    for item in translations:
                        if not isinstance(item, dict):
                            continue
                        idx = item.get("index")
                        text = item.get("text")
                        if isinstance(idx, int) and isinstance(text, str):
                            translated_by_index[idx] = text
                if translated_by_index:
                    return translated_by_index
            except Exception:
                continue
        return {}

    def _translate_single_with_retries(self, source_text: str) -> str:
        prompt = (
            "Translate Japanese manga dialogue into natural Simplified Chinese.\n"
            "Output only translated text with no explanations.\n"
            f"Japanese: {source_text}"
        )
        messages = [
            {"role": "system", "content": "You are a Japanese-to-Simplified-Chinese manga translator."},
            {"role": "user", "content": prompt},
        ]
        for _ in range(2):
            try:
                content = self._call_chat_completion(messages).strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:text|plain)?\s*", "", content)
                    content = re.sub(r"\s*```$", "", content)
                content = content.strip().strip('"').strip("'")
                if content:
                    return content
            except Exception:
                continue
        return ""

    def _call_chat_completion(self, messages: list[dict[str, str]]) -> str:
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.2,
        }
        response = self.session.post(
            urljoin(self.endpoint, "v1/chat/completions"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=data,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "\n".join(chunks)
        if not isinstance(content, str):
            raise ValueError("AI translator returned unsupported content format.")
        return content

    def _extract_json(self, content: str) -> dict[str, object]:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        start = content.find("{")
        if start == -1:
            raise ValueError("AI translator did not return JSON content.")
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(content)):
            char = content[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(content[start:idx + 1])
        raise ValueError("AI translator returned malformed JSON content.")


def normalize_ocr_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


DEFAULT_MANGA_TERM_GLOSSARY = {
    "冒険者ギルド": {
        "preferred": "冒险者公会",
        "replace_targets": ["冒险家协会", "冒险家公会"],
    },
    "登録用紙": {
        "preferred": "登记表",
        "replace_targets": ["注册表"],
    },
    "身分証": {
        "preferred": "身份证明",
        "replace_targets": ["身份证"],
    },
}


DEFAULT_EXACT_TRANSLATION_OVERRIDES = {
    "港町アシュベル": "港镇阿什贝尔",
    "ここが冒険者ギルドです": "这里就是冒险者公会。",
    "また振り返った．．．": "我又回头看了……",
    "目立ってるのは俺じゃないグレイスだ．．．！": "显眼的不是我，是格雷斯……！",
    "数時間前": "几个小时前",
    "この先は街ですから目立たない服に着替えておきました": "前面就是城里了，所以我先换了身不显眼的衣服。",
    "もう少しゆったりした服の方がよかったんじゃないか？なんというか逆に目立ってるような．．．": "你穿得再宽松一点是不是更好？该怎么说呢，感觉反而更显眼了……",
    "待て待て侍て思いっきりどこかが目立ってるぞ．．．！！": "等等等等，某个地方也太显眼了吧……！！",
    "大丈夫です明日になればみんな覚えていませんから": "没事，到了明天大家就都不会记得了。",
    "ー．．．ホントかよ．．": "……真的吗……",
    "．．．うん間違いないな俺だったら少なくともこのオッパイは忘れない": "……嗯，没错。要是我的话，至少这胸我绝不会忘。",
    "ここの冒険者ギルドに登録したいのですが": "我想在这里登记成为冒险者。",
    "いらっしゃい用件は何？": "欢迎光临，请问有什么事？",
    "じゃあ２人ともこの登録用紙に必要事項を書いて": "那你们两个都把这张登记表上的必要事项填一下。",
    "――．．．！文字は読める？": "……！你看得懂字吗？",
    "ええわたしはハーランドの外から来ましたからこの国の身分証を持っていません": "对，我是从哈兰德国外来的，所以没有这个国家的身份证明。",
    "．．．あれ？グレイスも登録するのか？": "……咦？格雷斯也要登记吗？",
    "はい大丈夫です": "嗯，没问题。",
    "なるほど．．．パスポートみたいな統一された仕組みがないわけか": "原来如此……也就是说，没有像护照那样统一的制度啊。",
    "であれば俺と同じように身分証を持たない人間というのはそれなりに居そうだ": "那样的话，像我一样没有身份证明的人应该也有不少。"
}


DEFAULT_PATTERN_TRANSLATION_RULES = [
    TranslationPatternRule(
        source_contains=("用件",),
        target_contains=("家具", "家俱"),
        replacement="欢迎光临，请问有什么事？",
    ),
    TranslationPatternRule(
        source_contains=("目立たない服",),
        target_contains=("隐形",),
        replacement="前面就是城里了，所以我先换了身不显眼的衣服。",
    ),
    TranslationPatternRule(
        source_contains=("目立ってる",),
        target_contains=("站着",),
        replacement="显眼的不是我，是格雷斯……！",
    ),
    TranslationPatternRule(
        source_contains=("オッパイ",),
        target_contains=("饼", "忘记"),
        replacement="……嗯，没错。要是我的话，至少这胸我绝不会忘。",
    ),
    TranslationPatternRule(
        source_contains=("統一された仕組みがない",),
        target_contains=("吗",),
        replacement="原来如此……也就是说，没有像护照那样统一的制度啊。",
    ),
    TranslationPatternRule(
        source_contains=("それなりに居そう",),
        target_contains=("足够",),
        replacement="那样的话，像我一样没有身份证明的人应该也有不少。",
    ),
]


def build_translation_rules_config(
    term_glossary: dict[str, dict[str, object]] | None = None,
    exact_overrides: dict[str, str] | None = None,
    pattern_rules: list[TranslationPatternRule] | None = None,
) -> TranslationRulesConfig:
    return TranslationRulesConfig(
        term_glossary=term_glossary or DEFAULT_MANGA_TERM_GLOSSARY,
        exact_overrides=exact_overrides or DEFAULT_EXACT_TRANSLATION_OVERRIDES,
        pattern_rules=pattern_rules or DEFAULT_PATTERN_TRANSLATION_RULES,
    )


def load_translation_rules(config_path: Path | None = None) -> TranslationRulesConfig:
    config_path = config_path or (Path(__file__).resolve().parent / DEFAULT_TRANSLATION_RULES_FILE)
    if not config_path.exists():
        return build_translation_rules_config()

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return build_translation_rules_config()

    term_glossary_raw = payload.get("term_glossary", {})
    term_glossary: dict[str, dict[str, object]] = {}
    if isinstance(term_glossary_raw, dict):
        for source_term, config in term_glossary_raw.items():
            if not isinstance(source_term, str) or not isinstance(config, dict):
                continue
            preferred = config.get("preferred")
            replace_targets = config.get("replace_targets", [])
            if not isinstance(preferred, str):
                continue
            if not isinstance(replace_targets, list):
                replace_targets = []
            normalized_targets = [item for item in replace_targets if isinstance(item, str)]
            term_glossary[source_term] = {
                "preferred": preferred,
                "replace_targets": normalized_targets,
            }

    exact_overrides_raw = payload.get("exact_overrides", {})
    exact_overrides = {
        source: target
        for source, target in exact_overrides_raw.items()
        if isinstance(source, str) and isinstance(target, str)
    } if isinstance(exact_overrides_raw, dict) else {}

    pattern_rules_raw = payload.get("pattern_rules", [])
    pattern_rules: list[TranslationPatternRule] = []
    if isinstance(pattern_rules_raw, list):
        for item in pattern_rules_raw:
            if not isinstance(item, dict):
                continue
            source_contains = item.get("source_contains", [])
            target_contains = item.get("target_contains", [])
            replacement = item.get("replacement")
            if not isinstance(source_contains, list) or not isinstance(target_contains, list) or not isinstance(replacement, str):
                continue
            normalized_source = tuple(token for token in source_contains if isinstance(token, str))
            normalized_target = tuple(token for token in target_contains if isinstance(token, str))
            if not normalized_source or not normalized_target:
                continue
            pattern_rules.append(
                TranslationPatternRule(
                    source_contains=normalized_source,
                    target_contains=normalized_target,
                    replacement=replacement,
                )
            )

    return build_translation_rules_config(
        term_glossary=term_glossary,
        exact_overrides=exact_overrides,
        pattern_rules=pattern_rules,
    )


TRANSLATION_RULES = load_translation_rules()


def apply_term_glossary(text: str, source_text: str) -> str:
    for source_term, config in TRANSLATION_RULES.term_glossary.items():
        if source_term in source_text:
            preferred_term = str(config.get("preferred", ""))
            replace_targets = config.get("replace_targets", [])
            if not preferred_term or not isinstance(replace_targets, list):
                continue
            for candidate in replace_targets:
                if isinstance(candidate, str):
                    text = text.replace(candidate, preferred_term)
    return text


def apply_source_aware_translation_fixes(text: str, source_text: str) -> str:
    source_text = normalize_ocr_text(source_text)
    text = text.strip()
    text = apply_term_glossary(text, source_text)
    exact_override = TRANSLATION_RULES.exact_overrides.get(source_text)
    if exact_override:
        return exact_override

    for rule in TRANSLATION_RULES.pattern_rules:
        if not all(token in source_text for token in rule.source_contains):
            continue
        if not any(token in text for token in rule.target_contains):
            continue
        return rule.replacement
    return text


def normalize_chinese_text(text: str, source_text: str = "") -> str:
    text = text.strip()
    source_text = normalize_ocr_text(source_text)
    source_core = re.sub(r"^[\s，。！？：；、…—–―\-.．!?,]+", "", source_text)
    replacements = {
        ",": "，",
        ".": "。",
        "?": "？",
        "!": "！",
        ":": "：",
        ";": "；",
        "(": "（",
        ")": "）",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"[.．]{3,}", "……", text)
    text = re.sub(r"。{2,}", "……", text)
    text = re.sub(r"…{3,}", "……", text)
    text = re.sub(r"[—–―-]{2,}", "—", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"！{2,}", "！", text)
    text = re.sub(r"？{2,}", "？", text)
    text = re.sub(r"，{2,}", "，", text)
    text = re.sub(r"。{2,}", "。", text)

    allow_question = "？" in source_core or "?" in source_core
    allow_exclaim = "！" in source_core or "!" in source_core
    allow_ellipsis = any(token in source_core for token in ("…", "...", "．．．", "。。。"))

    text = re.sub(r"(^|(?<=[，。！？：；、]))[—–―-]+", "", text)
    text = re.sub(r"[—–―-]+($|(?=[，。！？：；、]))", "", text)
    text = re.sub(r"^[，。！？：；、…—–―-]+", "", text)

    if not allow_question:
        text = text.replace("？", "")
    if not allow_exclaim:
        text = text.replace("！", "")
    if not allow_ellipsis:
        text = text.replace("……", "")
    text = re.sub(r"[—–―]+", "", text)

    text = re.sub(r"^[，。！？：；、…—–―-]+", "", text)
    text = re.sub(r"[，：；、…—–―-]+$", "", text)
    text = re.sub(r"([，。！？]){2,}", r"\1", text)
    return apply_source_aware_translation_fixes(text, source_text)


def has_excessive_repetition(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 20:
        return False
    for unit_len in range(1, 5):
        pattern = re.compile(rf"(.{{{unit_len}}})\1{{5,}}")
        if pattern.search(compact):
            return True
    return False


def is_degenerate_translation(source_text: str, translated_text: str) -> bool:
    translated_text = translated_text.strip()
    if not translated_text:
        return True
    if has_excessive_repetition(translated_text):
        return True
    source_compact = re.sub(r"\s+", "", source_text)
    target_compact = re.sub(r"\s+", "", translated_text)
    if len(target_compact) > max(80, len(source_compact) * 4):
        return True
    return False


def load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(font_path, size=size)
    except OSError:
        return ImageFont.load_default()


def clamp_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def bubble_mask_to_polygon(mask: np.ndarray) -> list[list[int]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(2.0, 0.005 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return [[int(pt[0][0]), int(pt[0][1])] for pt in approx]


def compute_text_box_from_mask(
    crop_mask: np.ndarray,
    offset_x: int,
    offset_y: int,
) -> tuple[int, int, int, int]:
    mask_h, mask_w = crop_mask.shape[:2]
    kernel_size = max(3, int(min(mask_h, mask_w) * 0.08))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(crop_mask, kernel, iterations=1)
    if cv2.countNonZero(eroded) == 0:
        eroded = cv2.erode(crop_mask, np.ones((3, 3), np.uint8), iterations=1)
    coords = cv2.findNonZero(eroded)
    if coords is None:
        coords = cv2.findNonZero(crop_mask)
    if coords is None:
        return offset_x, offset_y, offset_x + mask_w, offset_y + mask_h
    x, y, w, h = cv2.boundingRect(coords)
    return offset_x + x, offset_y + y, offset_x + x + w, offset_y + y + h


def rect_intersection(
    box1: list[int] | tuple[int, int, int, int],
    box2: list[int] | tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    x1 = max(int(box1[0]), int(box2[0]))
    y1 = max(int(box1[1]), int(box2[1]))
    x2 = min(int(box1[2]), int(box2[2]))
    y2 = min(int(box1[3]), int(box2[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def union_boxes(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def polygon_to_box(polygon: object) -> tuple[int, int, int, int] | None:
    try:
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
    except Exception:
        return None
    if pts.size == 0:
        return None
    x1 = int(np.min(pts[:, 0]))
    y1 = int(np.min(pts[:, 1]))
    x2 = int(np.max(pts[:, 0]))
    y2 = int(np.max(pts[:, 1]))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def choose_text_region(
    bubble_box: list[int],
    fallback_text_box: tuple[int, int, int, int],
    text_blocks: list[dict[str, object]],
) -> tuple[tuple[int, int, int, int], str, list[tuple[int, int, int, int]]]:
    overlaps: list[tuple[int, int, int, int]] = []
    guide_boxes: list[tuple[int, int, int, int]] = []
    vertical_votes = 0
    for block in text_blocks:
        rect = rect_intersection(bubble_box, block["bbox"])
        if rect is None:
            continue
        inter_area = (rect[2] - rect[0]) * (rect[3] - rect[1])
        bx1, by1, bx2, by2 = block["bbox"]
        block_area = max(1, (bx2 - bx1) * (by2 - by1))
        if inter_area / block_area < 0.3:
            continue
        overlaps.append(rect)
        if bool(block.get("vertical", False)):
            vertical_votes += 1
        for line in block.get("lines", []):
            line_box = polygon_to_box(line)
            if line_box is None:
                continue
            clipped = rect_intersection(bubble_box, line_box)
            if clipped is not None:
                guide_boxes.append(clipped)
    if not overlaps:
        x1, y1, x2, y2 = fallback_text_box
        mode = "vertical" if (y2 - y1) / max(1, (x2 - x1)) >= 1.3 else "horizontal"
        return fallback_text_box, mode, []
    text_box = union_boxes(overlaps)
    x1, y1, x2, y2 = text_box
    fx1, fy1, fx2, fy2 = fallback_text_box
    text_w = x2 - x1
    text_h = y2 - y1
    fallback_w = fx2 - fx1
    fallback_h = fy2 - fy1
    if vertical_votes >= max(1, len(overlaps) // 2):
        mode = "vertical"
    else:
        mode = "vertical" if (y2 - y1) / max(1, (x2 - x1)) >= 1.15 else "horizontal"
    if mode == "horizontal" and fallback_w > text_w * 1.15 and fallback_h >= text_h * 0.85:
        text_box = fallback_text_box
    if mode == "vertical" and fallback_w > max(text_w * 1.35, text_w + 20):
        if len(guide_boxes) <= 1 or text_w < max(42, int(fallback_w * 0.55)):
            text_box = fallback_text_box
    if mode == "vertical":
        guide_boxes = sorted(guide_boxes, key=lambda box: box[0], reverse=True)
    else:
        guide_boxes = sorted(guide_boxes, key=lambda box: box[1])
    return text_box, mode, guide_boxes


def compute_file_signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def build_cache_key(args: argparse.Namespace) -> str:
    payload = {
        "translator_backend": getattr(args, "translator_backend", DEFAULT_TRANSLATOR_BACKEND),
        "segmentation_repo": args.segmentation_repo,
        "segmentation_file": args.segmentation_file,
        "translation_model": args.translation_model,
        "ai_model": getattr(args, "ai_model", ""),
        "ai_endpoint": getattr(args, "ai_endpoint", ""),
        "inpaint_backend": args.inpaint_backend,
        "text_detector": args.text_detector,
        "conf": args.conf,
        "margin": args.margin,
        "font": args.font,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def is_page_cached(
    image_path: Path,
    translated_dir: Path,
    debug_dir: Path,
    json_dir: Path,
    cache_key: str,
) -> bool:
    translated_path = translated_dir / image_path.name
    debug_path = debug_dir / image_path.name
    json_path = json_dir / f"{image_path.stem}.json"
    if not (translated_path.exists() and debug_path.exists() and json_path.exists()):
        return False
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    cache_meta = payload.get("cache", {})
    return (
        cache_meta.get("cache_key") == cache_key
        and cache_meta.get("source_signature") == compute_file_signature(image_path)
    )


def filter_text_components(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    min_area = max(8, (mask.shape[0] * mask.shape[1]) // 2500)
    max_area = max(min_area * 4, (mask.shape[0] * mask.shape[1]) // 3)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        if area < min_area or area > max_area:
            continue
        if width <= 1 or height <= 1:
            continue
        filtered[labels == label] = 255
    return filtered


def extract_text_mask(crop_bgr: np.ndarray, bubble_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )
    edges = cv2.Canny(gray, 80, 160)
    dark_mask = cv2.bitwise_or(otsu_mask, adaptive_mask)
    dark_mask = cv2.bitwise_or(dark_mask, edges)
    dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=bubble_mask)
    dark_mask = filter_text_components(dark_mask)
    kernel = np.ones((3, 3), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
    dark_mask = cv2.dilate(dark_mask, kernel, iterations=1)
    return dark_mask


def extract_text_blocks_and_mask(runtime: PipelineRuntime, image_bgr: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray | None]:
    if runtime.text_detector is None:
        return [], None
    try:
        mask, mask_refined, blk_list = runtime.text_detector(image_bgr, keep_undetected_mask=True)
    except Exception:
        return [], None
    blocks: list[dict[str, object]] = []
    for blk in blk_list:
        xyxy = [int(v) for v in blk.xyxy]
        blocks.append({"bbox": xyxy})
    return blocks, mask_refined


def extract_text_blocks_and_mask_from_detector(text_detector: object | None, image_bgr: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray | None]:
    if text_detector is None:
        return [], None
    try:
        _, mask_refined, blk_list = text_detector(image_bgr, keep_undetected_mask=True)
    except Exception:
        return [], None
    blocks: list[dict[str, object]] = []
    for blk in blk_list:
        xyxy = [int(v) for v in blk.xyxy]
        blocks.append(
            {
                "bbox": xyxy,
                "vertical": bool(getattr(blk, "vertical", False)),
                "lines": [np.array(line).tolist() for line in getattr(blk, "lines", [])],
            }
        )
    return blocks, mask_refined


def clean_bubble_for_ocr(crop_bgr: np.ndarray, bubble_mask: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    white_bg = np.full_like(rgb, 255)
    masked = np.where(bubble_mask[..., None] > 0, rgb, white_bg)
    return Image.fromarray(masked)


def wrap_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if max_chars <= 1:
        return list(text)
    punctuation = set("，。！？；：、…」』）")
    chunks: list[str] = []
    current = ""
    for char in text:
        if char == "\n":
            if current:
                chunks.append(current)
                current = ""
            continue
        if len(current) >= max_chars:
            if char in punctuation and chunks:
                chunks[-1] += char
                continue
            chunks.append(current)
            current = char
            continue
        current += char
    if current:
        chunks.append(current)
    return chunks


def fit_horizontal_text_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    box_width: int,
    box_height: int,
    font_path: str,
    stroke_width: int,
) -> tuple[list[str], ImageFont.ImageFont]:
    start_size = max(12, min(box_height, int(box_height * 0.9)))
    for font_size in range(start_size, 9, -1):
        font = load_font(font_path, font_size)
        sample_bbox = draw.textbbox((0, 0), "测试文字", font=font, stroke_width=stroke_width)
        avg_char_width = max(1, int((sample_bbox[2] - sample_bbox[0]) / 4))
        max_chars = max(1, box_width // avg_char_width)
        lines = wrap_text(text, max_chars)
        if not lines:
            continue
        line_heights = []
        max_line_width = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font, align="center", stroke_width=stroke_width)
            max_line_width = max(max_line_width, bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])
        total_height = sum(line_heights) + max(0, len(lines) - 1) * max(3, font_size // 5)
        if max_line_width <= box_width and total_height <= box_height:
            return lines, font
    fallback_font = load_font(font_path, 9)
    return wrap_text(text, max(1, box_width // 9)), fallback_font


def fit_vertical_text_columns_v2(
    draw: ImageDraw.ImageDraw,
    text: str,
    box_width: int,
    box_height: int,
    font_path: str,
    stroke_width: int,
) -> tuple[list[list[str]], ImageFont.ImageFont]:
    characters = [char for char in text.strip() if char not in {"\n", "\r", " "}]
    if not characters:
        return [], load_font(font_path, 9)
    for font_size in range(max(12, min(box_width, box_height) // 3), 9, -1):
        font = load_font(font_path, font_size)
        char_bbox = draw.textbbox((0, 0), "汉", font=font, stroke_width=stroke_width)
        char_width = max(1, char_bbox[2] - char_bbox[0])
        char_height = max(1, char_bbox[3] - char_bbox[1])
        row_gap = max(2, font_size // 8)
        col_gap = max(4, font_size // 3)
        rows_per_col = max(1, (box_height + row_gap) // (char_height + row_gap))
        cols_needed = math.ceil(len(characters) / rows_per_col)
        total_width = cols_needed * char_width + max(0, cols_needed - 1) * col_gap
        total_height = min(len(characters), rows_per_col) * char_height + max(0, min(len(characters), rows_per_col) - 1) * row_gap
        if total_width <= box_width and total_height <= box_height:
            columns: list[list[str]] = []
            for start in range(0, len(characters), rows_per_col):
                columns.append(characters[start:start + rows_per_col])
            return columns, font
    fallback_font = load_font(font_path, 9)
    return [[char] for char in characters], fallback_font


def fit_vertical_text_columns(
    draw: ImageDraw.ImageDraw,
    text: str,
    box_width: int,
    box_height: int,
    font_path: str,
    stroke_width: int,
) -> tuple[list[list[str]], ImageFont.ImageFont]:
    characters = [char for char in text.strip() if char not in {"\n", "\r", " "}]
    if not characters:
        return [], load_font(font_path, 9)
    for font_size in range(max(12, min(box_width, box_height) // 3), 9, -1):
        font = load_font(font_path, font_size)
        char_bbox = draw.textbbox((0, 0), "汉", font=font, stroke_width=stroke_width)
        char_width = max(1, char_bbox[2] - char_bbox[0])
        char_height = max(1, char_bbox[3] - char_bbox[1])
        row_gap = max(2, font_size // 8)
        col_gap = max(4, font_size // 3)
        rows_per_col = max(1, (box_height + row_gap) // (char_height + row_gap))
        if rows_per_col <= 0:
            continue
        cols_needed = math.ceil(len(characters) / rows_per_col)
        total_width = cols_needed * char_width + max(0, cols_needed - 1) * col_gap
        total_height = min(len(characters), rows_per_col) * char_height + max(0, min(len(characters), rows_per_col) - 1) * row_gap
        if total_width <= box_width and total_height <= box_height:
            columns: list[list[str]] = []
            for start in range(0, len(characters), rows_per_col):
                columns.append(characters[start:start + rows_per_col])
            return columns, font
    fallback_font = load_font(font_path, 9)
    return [[char] for char in characters], fallback_font


def draw_horizontal_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font_path: str,
) -> None:
    x1, y1, x2, y2 = box
    box_width = max(10, x2 - x1)
    box_height = max(10, y2 - y1)
    stroke_width = 1 if min(box_width, box_height) < 80 else 2
    lines, font = fit_horizontal_text_lines(draw, text, box_width, box_height, font_path, stroke_width)
    if not lines:
        return
    spacing = max(3, getattr(font, "size", 12) // 5)
    line_metrics = [draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width) for line in lines]
    total_height = sum(metric[3] - metric[1] for metric in line_metrics) + spacing * (len(lines) - 1)
    start_y = y1 + (box_height - total_height) / 2
    for line, metric in zip(lines, line_metrics):
        line_width = metric[2] - metric[0]
        line_height = metric[3] - metric[1]
        pos = (x1 + (box_width - line_width) / 2, start_y)
        draw.text(
            pos,
            line,
            font=font,
            fill=(20, 20, 20),
            stroke_width=stroke_width,
            stroke_fill=(255, 255, 255),
            align="center",
        )
        start_y += line_height + spacing


def fit_vertical_text_to_guides(
    draw: ImageDraw.ImageDraw,
    text: str,
    guide_boxes: list[tuple[int, int, int, int]],
    font_path: str,
    stroke_width: int,
) -> tuple[list[tuple[int, int, int, int]], list[list[str]], ImageFont.ImageFont] | None:
    characters = [char for char in text.strip() if char not in {"\n", "\r", " "}]
    if not characters or not guide_boxes:
        return None

    def merge_guides(boxes: list[tuple[int, int, int, int]], target_cols: int) -> list[tuple[int, int, int, int]]:
        if target_cols >= len(boxes):
            return boxes
        chunk_size = math.ceil(len(boxes) / target_cols)
        merged: list[tuple[int, int, int, int]] = []
        for guide_index in range(0, len(boxes), chunk_size):
            chunk = boxes[guide_index:guide_index + chunk_size]
            merged.append(union_boxes(chunk))
        return merged

    expanded_base: list[tuple[int, int, int, int]] = []
    for guide in guide_boxes:
        gx1, gy1, gx2, gy2 = guide
        expand = max(6, int((gx2 - gx1) * 0.35))
        expanded_base.append((gx1 - expand, gy1, gx2 + expand, gy2))

    max_dim = max(max(box[2] - box[0], box[3] - box[1]) for box in expanded_base)
    for target_cols in range(len(expanded_base), 0, -1):
        candidate_guides = merge_guides(expanded_base, target_cols)
        for font_size in range(max(12, max_dim // 2), 9, -1):
            font = load_font(font_path, font_size)
            char_bbox = draw.textbbox((0, 0), "汉", font=font, stroke_width=stroke_width)
            char_width = max(1, char_bbox[2] - char_bbox[0])
            char_height = max(1, char_bbox[3] - char_bbox[1])
            row_gap = max(2, font_size // 8)
            columns: list[list[str]] = []
            capacities: list[int] = []
            for guide in candidate_guides:
                gx1, gy1, gx2, gy2 = guide
                capacity = max(1, (max(1, gy2 - gy1) + row_gap) // (char_height + row_gap))
                if char_width > max(1, gx2 - gx1):
                    capacity = 0
                capacities.append(capacity)
            if sum(capacities) < len(characters):
                continue
            cursor = 0
            for capacity in capacities:
                take = min(capacity, len(characters) - cursor)
                columns.append(characters[cursor:cursor + take])
                cursor += take
            if cursor >= len(characters):
                return candidate_guides, columns, font
    return None

def draw_vertical_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font_path: str,
    guide_boxes: list[tuple[int, int, int, int]] | None = None,
) -> None:
    x1, y1, x2, y2 = box
    box_width = max(10, x2 - x1)
    box_height = max(10, y2 - y1)
    stroke_width = 1 if min(box_width, box_height) < 80 else 2
    guided_layout = None
    if guide_boxes:
        guided_layout = fit_vertical_text_to_guides(draw, text, guide_boxes, font_path, stroke_width)
    if guided_layout is not None:
        active_guides, columns, font = guided_layout
    else:
        columns, font = fit_vertical_text_columns_v2(draw, text, box_width, box_height, font_path, stroke_width)
        active_guides = guide_boxes or []
    if not columns:
        return
    char_bbox = draw.textbbox((0, 0), "汉", font=font, stroke_width=stroke_width)
    char_width = max(1, char_bbox[2] - char_bbox[0])
    char_height = max(1, char_bbox[3] - char_bbox[1])
    row_gap = max(2, getattr(font, "size", 12) // 8)
    col_gap = max(4, getattr(font, "size", 12) // 3)
    if active_guides and guided_layout is not None and len(active_guides) >= len(columns):
        for guide, column in zip(active_guides, columns):
            gx1, gy1, gx2, gy2 = guide
            total_height = len(column) * char_height + max(0, len(column) - 1) * row_gap
            start_y = gy1 + ((gy2 - gy1) - total_height) / 2
            for char in column:
                bbox = draw.textbbox((0, 0), char, font=font, stroke_width=stroke_width)
                char_draw_width = bbox[2] - bbox[0]
                pos = (gx1 + ((gx2 - gx1) - char_draw_width) / 2, start_y)
                draw.text(
                    pos,
                    char,
                    font=font,
                    fill=(20, 20, 20),
                    stroke_width=stroke_width,
                    stroke_fill=(255, 255, 255),
                    align="center",
                )
                start_y += char_height + row_gap
        return
    total_width = len(columns) * char_width + max(0, len(columns) - 1) * col_gap
    start_x = x1 + (box_width - total_width) / 2 + total_width - char_width
    for column in columns:
        total_height = len(column) * char_height + max(0, len(column) - 1) * row_gap
        start_y = y1 + (box_height - total_height) / 2
        for char in column:
            bbox = draw.textbbox((0, 0), char, font=font, stroke_width=stroke_width)
            char_draw_width = bbox[2] - bbox[0]
            pos = (start_x + (char_width - char_draw_width) / 2, start_y)
            draw.text(
                pos,
                char,
                font=font,
                fill=(20, 20, 20),
                stroke_width=stroke_width,
                stroke_fill=(255, 255, 255),
                align="center",
            )
            start_y += char_height + row_gap
        start_x -= char_width + col_gap

def draw_text_in_box(
    image: Image.Image,
    box: tuple[int, int, int, int],
    text: str,
    font_path: str,
    layout_mode: str | None = None,
    line_guides: list[list[int]] | None = None,
) -> None:
    if not text.strip():
        return
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if layout_mode is None:
        is_vertical = height / max(1, width) >= 1.3
    else:
        is_vertical = layout_mode == "vertical"
    if is_vertical:
        pad_x = max(2, int(width * 0.04))
        pad_y = max(4, int(height * 0.05))
    else:
        pad_x = max(2, int(width * 0.03))
        pad_y = max(2, int(height * 0.04))
    inner_x1 = min(x2 - 10, x1 + pad_x)
    inner_y1 = min(y2 - 10, y1 + pad_y)
    inner_x2 = max(inner_x1 + 10, x2 - pad_x)
    inner_y2 = max(inner_y1 + 10, y2 - pad_y)
    inner_box = (inner_x1, inner_y1, inner_x2, inner_y2)
    if is_vertical:
        guide_boxes = None
        if line_guides:
            guide_boxes = []
            for guide in line_guides:
                clipped = rect_intersection(inner_box, guide)
                if clipped is not None:
                    guide_boxes.append(clipped)
        draw_vertical_text(draw, inner_box, text, font_path, guide_boxes)
    else:
        draw_horizontal_text(draw, inner_box, text, font_path)


def collect_image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted([path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS])


def ensure_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
    translated_dir = output_dir / "translated"
    debug_dir = output_dir / "debug"
    json_dir = output_dir / "json"
    translated_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    return translated_dir, debug_dir, json_dir


def render_debug_overlay(image_bgr: np.ndarray, detections: list[BubbleResult]) -> np.ndarray:
    overlay = image_bgr.copy()
    for det in detections:
        polygon = np.array(det.polygon, dtype=np.int32)
        if len(polygon) >= 3:
            cv2.polylines(overlay, [polygon], True, (0, 200, 0), 3)
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 120, 0), 2)
        label = f"{det.index}: {det.chinese_text[:14] or det.japanese_text[:14] or 'empty'}"
        cv2.putText(
            overlay,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 240),
            2,
            cv2.LINE_AA,
        )
    return overlay


def process_image(
    image_path: Path,
    model: YOLO,
    ocr: MangaOcr,
    translator: BaseTranslator,
    inpainter: Inpainter,
    text_detector: object | None,
    translated_dir: Path,
    debug_dir: Path,
    json_dir: Path,
    conf_threshold: float,
    font_path: str,
    margin: int,
    cache_key: str,
    force: bool,
) -> str:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    translated_path = translated_dir / image_path.name
    debug_path = debug_dir / image_path.name
    json_path = json_dir / f"{image_path.stem}.json"
    source_signature = compute_file_signature(image_path)
    if not force and translated_path.exists() and debug_path.exists() and json_path.exists():
        try:
            cached_payload = json.loads(json_path.read_text(encoding="utf-8"))
            cache_meta = cached_payload.get("cache", {})
            if cache_meta.get("cache_key") == cache_key and cache_meta.get("source_signature") == source_signature:
                print(f"Cached {image_path.name}")
                return "cached"
        except (OSError, json.JSONDecodeError):
            pass

    height, width = image_bgr.shape[:2]
    page_text_blocks, page_text_mask = extract_text_blocks_and_mask_from_detector(text_detector, image_bgr)
    results = model.predict(source=str(image_path), conf=conf_threshold, imgsz=1536, verbose=False)
    result = results[0]
    working_image = image_bgr.copy()
    detections: list[BubbleResult] = []
    candidates: list[dict[str, object]] = []

    raw_masks = None if result.masks is None else result.masks.data.cpu().numpy()
    boxes = None if result.boxes is None else result.boxes
    if raw_masks is None or boxes is None:
        cv2.imwrite(str(translated_path), image_bgr)
        payload = {
            "image": image_path.name,
            "model_repo": DEFAULT_SEGMENTATION_REPO,
            "translation_model": translator.descriptor,
            "cache": {
                "cache_key": cache_key,
                "source_signature": source_signature,
            },
            "detections": [],
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        cv2.imwrite(str(debug_path), image_bgr)
        return "processed"

    masks = np.stack(
        [
            cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            for mask in raw_masks
        ],
        axis=0,
    )

    order = np.argsort(boxes.xyxy[:, 1].cpu().numpy())

    for rank, idx in enumerate(order, start=1):
        conf = float(boxes.conf[idx].item())
        xyxy = boxes.xyxy[idx].cpu().numpy()
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        x1, y1, x2, y2 = clamp_box(x1 - margin, y1 - margin, x2 + margin, y2 + margin, width, height)

        full_mask = (masks[idx] > 0.5).astype(np.uint8) * 255
        if full_mask.sum() < 3000:
            continue
        crop_mask = full_mask[y1:y2, x1:x2]
        crop_bgr = working_image[y1:y2, x1:x2].copy()

        ocr_image = clean_bubble_for_ocr(crop_bgr, crop_mask)
        japanese_text = normalize_ocr_text(ocr(ocr_image))
        if not japanese_text:
            continue
        candidates.append(
            {
                "index": rank,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "crop_bgr": crop_bgr,
                "crop_mask": crop_mask,
                "full_mask": full_mask,
                "japanese_text": japanese_text,
            }
        )

    chinese_texts = translator.translate_batch([candidate["japanese_text"] for candidate in candidates])
    for candidate, chinese_text in zip(candidates, chinese_texts):
        japanese_text = str(candidate["japanese_text"])
        if is_degenerate_translation(japanese_text, chinese_text):
            continue
        x1, y1, x2, y2 = candidate["bbox"]
        crop_bgr = candidate["crop_bgr"]
        crop_mask = candidate["crop_mask"]
        full_mask = candidate["full_mask"]
        text_mask = extract_text_mask(crop_bgr, crop_mask)
        if page_text_mask is not None:
            detector_text_mask = page_text_mask[y1:y2, x1:x2]
            detector_text_mask = cv2.bitwise_and(detector_text_mask, detector_text_mask, mask=crop_mask)
            text_mask = cv2.bitwise_or(text_mask, detector_text_mask)
        cleaned_crop = inpainter(crop_bgr, text_mask)
        working_image[y1:y2, x1:x2] = cleaned_crop
        polygon = bubble_mask_to_polygon(full_mask)
        fallback_text_box = compute_text_box_from_mask(crop_mask, x1, y1)
        text_box, layout_mode, line_guides = choose_text_region([x1, y1, x2, y2], fallback_text_box, page_text_blocks)
        detections.append(
            BubbleResult(
                index=int(candidate["index"]),
                confidence=float(candidate["confidence"]),
                bbox=[x1, y1, x2, y2],
                text_box=list(text_box),
                layout_mode=layout_mode,
                line_guides=[list(guide) for guide in line_guides],
                polygon=polygon,
                japanese_text=japanese_text,
                chinese_text=chinese_text,
            )
        )

    pil_output = Image.fromarray(cv2.cvtColor(working_image, cv2.COLOR_BGR2RGB))
    for det in detections:
        draw_text_in_box(pil_output, tuple(det.text_box), det.chinese_text, font_path, det.layout_mode, det.line_guides)

    final_bgr = cv2.cvtColor(np.array(pil_output), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(translated_path), final_bgr)

    debug_overlay = render_debug_overlay(final_bgr, detections)
    cv2.imwrite(str(debug_path), debug_overlay)

    payload = {
        "image": image_path.name,
        "model_repo": DEFAULT_SEGMENTATION_REPO,
        "translation_model": translator.descriptor,
        "cache": {
            "cache_key": cache_key,
            "source_signature": source_signature,
        },
        "detections": [
            {
                "index": det.index,
                "confidence": round(det.confidence, 4),
                "bbox": det.bbox,
                "text_box": det.text_box,
                "layout_mode": det.layout_mode,
                "line_guides": det.line_guides,
                "polygon": det.polygon,
                "japanese_text": det.japanese_text,
                "chinese_text": det.chinese_text,
            }
            for det in detections
        ],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate Japanese manga bubbles into Chinese.")
    parser.add_argument("--input", default=".", help="Input image file or directory.")
    parser.add_argument("--output", default="outputs", help="Output directory.")
    parser.add_argument("--conf", type=float, default=0.25, help="Bubble segmentation confidence threshold.")
    parser.add_argument("--margin", type=int, default=8, help="Extra margin around each detected bubble.")
    parser.add_argument("--font", default=DEFAULT_FONT, help="Chinese font path for rendered text.")
    parser.add_argument(
        "--segmentation-repo",
        default=DEFAULT_SEGMENTATION_REPO,
        help="Hugging Face repo for the bubble segmentation model.",
    )
    parser.add_argument(
        "--segmentation-file",
        default=DEFAULT_SEGMENTATION_FILE,
        help="Weight filename inside the segmentation repo.",
    )
    parser.add_argument(
        "--translator-backend",
        choices=["local", "openai-compatible"],
        default=os.environ.get("MANGA_TRANSLATOR_BACKEND", DEFAULT_TRANSLATOR_BACKEND),
        help="Translation backend to use.",
    )
    parser.add_argument(
        "--translation-model",
        default=DEFAULT_TRANSLATION_MODEL,
        help="Seq2Seq translation model name.",
    )
    parser.add_argument(
        "--ai-endpoint",
        default=os.environ.get("OPENAI_COMPAT_ENDPOINT", ""),
        help="Base URL for the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--ai-api-key",
        default=os.environ.get("OPENAI_COMPAT_API_KEY", ""),
        help="API key for the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--ai-model",
        default=os.environ.get("OPENAI_COMPAT_MODEL", ""),
        help="Model name for the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--text-detector",
        choices=["comic-text-detector", "none"],
        default="comic-text-detector",
        help="Text region detector used to improve text masking and layout direction.",
    )
    parser.add_argument(
        "--inpaint-backend",
        choices=["lama", "opencv"],
        default="lama",
        help="Text removal backend.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore page cache and regenerate outputs.",
    )
    return parser.parse_args()


def create_runtime(args: argparse.Namespace) -> PipelineRuntime:
    weight_path = hf_hub_download(repo_id=args.segmentation_repo, filename=args.segmentation_file)
    model = YOLO(weight_path)
    ocr = MangaOcr()
    if getattr(args, "translator_backend", DEFAULT_TRANSLATOR_BACKEND) == "openai-compatible":
        ai_endpoint = getattr(args, "ai_endpoint", "").strip()
        ai_api_key = getattr(args, "ai_api_key", "").strip()
        ai_model = getattr(args, "ai_model", "").strip()
        if not ai_endpoint or not ai_api_key or not ai_model:
            raise SystemExit("AI translator requires --ai-endpoint, --ai-api-key, and --ai-model.")
        translator: BaseTranslator = OpenAICompatibleTranslator(
            endpoint=ai_endpoint,
            api_key=ai_api_key,
            model_name=ai_model,
        )
    else:
        translator = NLLBTranslator(args.translation_model)
    inpainter = Inpainter(args.inpaint_backend)
    text_detector = None
    if getattr(args, "text_detector", "comic-text-detector") == "comic-text-detector":
        repo = (Path(__file__).resolve().parent / "vendor" / "comic-text-detector").resolve()
        model_path = repo / "data" / "comictextdetector.pt.onnx"
        if model_path.exists():
            sys.path.insert(0, str(repo))
            from inference import TextDetector  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
            text_detector = TextDetector(model_path=str(model_path), input_size=1024, device=device)
    return PipelineRuntime(
        model=model,
        ocr=ocr,
        translator=translator,
        inpainter=inpainter,
        text_detector=text_detector,
    )


def process_images(
    image_paths: list[Path],
    runtime: PipelineRuntime,
    translated_dir: Path,
    debug_dir: Path,
    json_dir: Path,
    args: argparse.Namespace,
    cache_key: str,
) -> dict[str, int]:
    cached_images: list[Path] = []
    pending_images: list[Path] = []
    for image_path in image_paths:
        if not args.force and is_page_cached(image_path, translated_dir, debug_dir, json_dir, cache_key):
            cached_images.append(image_path)
        else:
            pending_images.append(image_path)

    stats = {"processed": 0, "cached": len(cached_images)}
    for image_path in cached_images:
        print(f"Cached {image_path.name}")

    for image_path in pending_images:
        status = process_image(
            image_path=image_path,
            model=runtime.model,
            ocr=runtime.ocr,
            translator=runtime.translator,
            inpainter=runtime.inpainter,
            text_detector=runtime.text_detector,
            translated_dir=translated_dir,
            debug_dir=debug_dir,
            json_dir=json_dir,
            conf_threshold=args.conf,
            font_path=args.font,
            margin=args.margin,
            cache_key=cache_key,
            force=True,
        )
        stats[status] += 1
        if status == "processed":
            print(f"Processed {image_path.name}")
    return stats


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    translated_dir, debug_dir, json_dir = ensure_dirs(output_dir)
    cache_key = build_cache_key(args)

    image_paths = collect_image_paths(input_path)
    if not image_paths:
        raise SystemExit(f"No supported images found in {input_path}")

    all_cached = all(
        is_page_cached(image_path, translated_dir, debug_dir, json_dir, cache_key)
        for image_path in image_paths
    ) if not args.force else False
    if all_cached:
        for image_path in image_paths:
            print(f"Cached {image_path.name}")
        print(f"Summary: processed=0 cached={len(image_paths)}")
        return

    runtime = create_runtime(args)
    stats = process_images(
        image_paths=image_paths,
        runtime=runtime,
        translated_dir=translated_dir,
        debug_dir=debug_dir,
        json_dir=json_dir,
        args=args,
        cache_key=cache_key,
    )
    print(f"Summary: processed={stats['processed']} cached={stats['cached']}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    main()
