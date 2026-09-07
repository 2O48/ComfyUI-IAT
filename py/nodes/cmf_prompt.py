from __future__ import annotations

"""Deterministic CMF request normalization, planning, and prompt rendering.

The language model may suggest a component/material assignment, but it never
owns the final wording or any user supplied color value.  This module keeps
those hard constraints in one small, offline-capable compiler.
"""

import hashlib
import itertools
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class CMFRequestError(ValueError):
    """Raised when a structured CMF request cannot be normalized."""


class CMFValidationError(ValueError):
    """Raised when a rendered CMF prompt does not satisfy its plan."""


COLOR_FAMILY_LABELS = {
    "black": "黑色系",
    "brown": "棕色系",
    "gray": "灰色系",
    "white": "白色系",
    "blue": "蓝色系",
    "green": "绿色系",
    "red": "红色系",
    "orange": "橙色系",
    "purple": "紫色系",
    "yellow": "黄色系",
    "cyan": "青色系",
    "beige": "米色系",
    "custom": "综合色系",
}
COLOR_FAMILIES = tuple(COLOR_FAMILY_LABELS)
AUXILIARY_COLOR_STRATEGIES = ("reuse_secondary", "dataset", "style", "none")

_COLOR_ALIASES = {
    "black": ("black", "黑色系", "黑色"),
    "brown": ("brown", "棕色系", "棕色", "褐色系", "褐色"),
    "gray": ("gray", "grey", "灰色系", "灰色", "灰蓝色系", "灰蓝色"),
    "white": ("white", "白色系", "白色", "米白色系", "米白色"),
    "blue": ("blue", "蓝色系", "蓝色"),
    "green": ("green", "绿色系", "绿色", "橄榄绿", "鼠尾草绿"),
    "red": ("red", "红色系", "红色", "酒红", "酒红色"),
    "orange": ("orange", "橙色系", "橙色", "赤陶橙", "陶土橙"),
    "purple": ("purple", "紫色系", "紫色", "紫灰色系", "紫灰色"),
    "yellow": ("yellow", "黄色系", "黄色", "金色系", "金色"),
    "cyan": ("cyan", "teal", "青色系", "青色", "蓝绿色系", "蓝绿色"),
    "beige": ("beige", "米色系", "米色", "米灰色系", "米灰色", "驼色系", "驼色"),
}

_FAMILY_NAME_ONLY_ALIASES = {
    "赤陶橙",
    "陶土橙",
    "橄榄绿",
    "鼠尾草绿",
    "酒红",
    "酒红色",
    "金色",
}
_FAMILY_MATCH_TERMS = sorted(
    (
        (alias, family)
        for family, aliases in _COLOR_ALIASES.items()
        for alias in aliases
        if alias not in _FAMILY_NAME_ONLY_ALIASES
    ),
    key=lambda item: len(item[0]),
    reverse=True,
)
_FAMILY_PATTERN = re.compile(
    "|".join(re.escape(alias) for alias, _family in _FAMILY_MATCH_TERMS),
    flags=re.IGNORECASE,
)

_FAMILY_BASE_HEX = {
    "black": "#1f1f1d",
    "brown": "#8b4f2f",
    "gray": "#8a8178",
    "white": "#d9d4c9",
    "blue": "#405a73",
    "green": "#526b52",
    "red": "#7b3730",
    "orange": "#a96735",
    "purple": "#665a78",
    "yellow": "#b28a3b",
    "cyan": "#477b7b",
    "beige": "#b6a182",
    "custom": "#808080",
}
_STYLE_FAMILY_HEXES = {
    "offroad": {
        "black": "#111820",
        "brown": "#7a4b2f",
        "gray": "#2f3f49",
        "orange": "#c96f3a",
        "green": "#4f604e",
        "purple": "#6b5b73",
    },
    "home": {
        "black": "#24272a",
        "brown": "#7a4b2f",
        "gray": "#8f9290",
        "beige": "#d8cbb8",
        "green": "#77856b",
        "purple": "#9b879f",
    },
}
_STYLE_PALETTE_FAMILIES = {
    "offroad": (
        ("primary", "orange"),
        ("secondary", "gray"),
        ("accent", "brown"),
        ("neutral", "black"),
    ),
    "home": (
        ("primary", "beige"),
        ("secondary", "gray"),
        ("accent", "brown"),
        ("neutral", "black"),
    ),
    "custom": (
        ("primary", "gray"),
        ("secondary", "black"),
        ("accent", "brown"),
        ("neutral", "white"),
    ),
}

_MATERIAL_LABELS = {
    "leather": "皮质",
    "suede": "麂皮",
    "fabric": "织物",
    "wood": "木纹",
    "metal": "金属",
    "rubber": "橡胶",
    "plastic": "塑料",
    "glass": "玻璃",
    "alcantara": "翻毛皮",
}
_MATERIAL_ALIASES = {
    "leather": ("leather", "皮革", "皮质", "真皮", "合成皮"),
    "suede": ("suede", "麂皮", "绒面", "绒面革"),
    "fabric": ("fabric", "织物", "布艺", "编织", "针织", "亚麻"),
    "wood": ("wood", "木纹", "木饰面", "木材", "原木"),
    "metal": ("metal", "金属", "铝饰面", "镀铬"),
    "rubber": ("rubber", "橡胶"),
    "plastic": ("plastic", "塑料"),
    "glass": ("glass", "玻璃"),
    "alcantara": ("alcantara", "翻毛皮", "超纤绒面"),
}
_MATERIAL_MATCH_TERMS = sorted(
    ((alias, material) for material, aliases in _MATERIAL_ALIASES.items() for alias in aliases),
    key=lambda item: len(item[0]),
    reverse=True,
)

STYLE_PHRASES = {
    "offroad": "越野风格CMF设计",
    "home": "居家风格CMF设计",
    "custom": "CMF风格设计",
}
_STYLE_ALIASES = {
    "offroad": ("offroad", "越野", "越野风格", "越野cmf", "越野风格cmf设计"),
    "home": ("home", "居家", "居家风格", "居家cmf", "居家风格cmf设计"),
}

SAMPLE_TYPES = ("full_cabin", "color_material", "detail")
SAMPLE_CLASSIFICATION_VERSION = 2
_FULL_CABIN_HINTS = (
    "完整座舱", "整车座舱", "座舱全景", "内饰全景", "完整内饰", "驾驶舱全景", "full cabin", "cockpit view",
)
_DETAIL_HINTS = (
    "座椅主面", "座椅侧翼", "门板", "中控台", "仪表台", "方向盘", "扶手", "顶棚", "a柱", "dashboard",
    "center console", "door panel", "steering wheel", "seat",
)
_COLOR_HINT_PATTERN = re.compile(
    r"(?:色系|色调|颜色|色彩|黑色|棕色|灰色|白色|蓝色|绿色|红色|橙色|紫色|黄色|青色|米色|#[0-9a-f]{6})",
    re.IGNORECASE,
)

_COMPONENTS = (
    "座椅主面料",
    "座椅侧翼",
    "门板内衬肌理",
    "中控台面上层",
    "中控台前饰板",
    "马鞍区面板",
    "方向盘",
)
_COLOR_ROLES = ("primary", "secondary", "accent", "neutral")
_MATERIAL_COMPATIBILITY = {
    "座椅主面料": ("leather", "fabric", "suede", "alcantara"),
    "座椅侧翼": ("leather", "suede", "alcantara", "fabric"),
    "门板内衬肌理": ("suede", "fabric", "leather", "alcantara"),
    "中控台面上层": ("leather", "suede", "fabric", "metal", "plastic", "glass"),
    "中控台前饰板": ("leather", "fabric", "metal", "plastic", "wood", "glass"),
    "马鞍区面板": ("wood", "leather", "metal", "fabric", "plastic"),
    "方向盘": ("leather", "suede", "alcantara", "rubber"),
}
_COMPONENT_ALIASES = {
    "座椅主面料": ("座椅主面料", "座椅主面", "座椅中心", "seat center", "seat main"),
    "座椅侧翼": ("座椅侧翼", "座椅侧面", "座椅翼部", "seat bolster", "seat side"),
    "门板内衬肌理": ("门板内衬肌理", "门板内衬", "门板", "door panel", "door trim"),
    "中控台面上层": ("中控台面上层", "中控台面", "仪表台面上层", "仪表台", "dashboard", "center console"),
    "中控台前饰板": ("中控台前饰板", "中控台前部", "中控前饰板", "center console front"),
    "马鞍区面板": ("马鞍区面板", "马鞍区", "换挡区域", "鞍座区", "saddle panel"),
    "方向盘": ("方向盘", "steering wheel"),
}
_COMPONENT_MATCH_TERMS = sorted(
    ((alias, component) for component, aliases in _COMPONENT_ALIASES.items() for alias in aliases),
    key=lambda item: len(item[0]),
    reverse=True,
)

# These names are intentionally deterministic.  Exact examples from the
# product brief are pinned so later previews cannot rename the same swatch.
_KNOWN_COLOR_NAMES = {
    "#c96f3a": "荒漠赤陶橙",
    "#2f3f49": "岩岭蓝灰",
    "#7a4b2f": "胡桃木棕",
    "#111820": "深曜黑",
}
_STYLE_NAME_HINTS = {
    "offroad": {
        "black": ("深曜黑", "岩炭黑", "曜石黑"),
        "brown": ("胡桃木棕", "复古鞍棕", "荒原棕"),
        "gray": ("岩岭蓝灰", "岩层灰", "雾岩灰"),
        "orange": ("荒漠赤陶橙", "沙砾橙", "陶土橙"),
        "purple": ("暮岩紫", "烟黛紫", "岩紫灰"),
        "green": ("深苔原绿", "荒野苔绿", "岩松绿"),
    },
    "home": {
        "black": ("柔和炭黑", "深灰黑", "静谧黑"),
        "brown": ("胡桃木棕", "暖木棕", "陶土棕"),
        "gray": ("雾灰", "柔岩灰", "暖灰"),
        "beige": ("暖米灰", "亚麻米白", "柔沙米色"),
        "green": ("鼠尾草绿", "橄榄灰绿", "苔原绿"),
        "purple": ("灰调浅紫", "柔雾紫", "暮光紫"),
    },
}
_COLOR_NAME_CACHE: Dict[str, str] = {}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_family(value: Any) -> Optional[str]:
    text = _clean_text(value).casefold()
    if not text:
        return None
    for family, aliases in _COLOR_ALIASES.items():
        if text in {alias.casefold() for alias in aliases}:
            return family
    if text in {"custom", "其他", "综合色系"}:
        return "custom"
    return None


def family_label(family: str) -> str:
    return COLOR_FAMILY_LABELS.get(family, COLOR_FAMILY_LABELS["custom"])


def normalize_material(value: Any) -> Optional[str]:
    text = _clean_text(value).casefold()
    if not text:
        return None
    for material, aliases in _MATERIAL_ALIASES.items():
        if text in {alias.casefold() for alias in aliases}:
            return material
    return None


def material_label(material: str) -> str:
    return _MATERIAL_LABELS.get(material, material)


def normalize_hex(value: Any) -> Optional[str]:
    if value is None or not _clean_text(value):
        return None
    if not isinstance(value, str):
        raise CMFRequestError("颜色 HEX 必须是字符串，例如 #A077AB。")
    text = value.strip().lower()
    if text.startswith("0x"):
        text = "#" + text[2:]
    if not text.startswith("#"):
        text = "#" + text
    digits = text[1:]
    if len(digits) == 3 and re.fullmatch(r"[0-9a-f]{3}", digits):
        digits = "".join(char * 2 for char in digits)
    if not re.fullmatch(r"[0-9a-f]{6}", digits):
        raise CMFRequestError(f"无效的颜色 HEX：{value!r}。必须是 #RRGGBB 或 #RGB。")
    return "#" + digits


def _normalize_rgb(value: Any) -> Optional[Tuple[int, int, int]]:
    if value is None or not _clean_text(value):
        return None
    raw = value
    if isinstance(value, str):
        match = re.fullmatch(r"\s*\(?\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*\)?\s*", value)
        if not match:
            raise CMFRequestError(f"无效的 RGB：{value!r}。格式应为 [R, G, B]。")
        raw = [int(item) for item in match.groups()]
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise CMFRequestError("RGB 必须包含三个 0-255 的整数。")
    channels: List[int] = []
    for channel in raw:
        if isinstance(channel, bool) or not isinstance(channel, (int, float)) or not math.isfinite(float(channel)):
            raise CMFRequestError("RGB 必须包含三个 0-255 的整数。")
        if int(channel) != float(channel) or not 0 <= int(channel) <= 255:
            raise CMFRequestError("RGB 必须包含三个 0-255 的整数。")
        channels.append(int(channel))
    return tuple(channels)  # type: ignore[return-value]


def _rgb_to_hex(rgb: Sequence[int]) -> str:
    return "#" + "".join(f"{int(channel):02x}" for channel in rgb)


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))  # type: ignore[return-value]


def _rgb_to_lab(rgb: Sequence[float]) -> Tuple[float, float, float]:
    linear = []
    for channel in rgb:
        value = float(channel) / 255.0
        linear.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
    r, g, b = linear
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / 1.00000
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    def pivot(value: float) -> float:
        return value ** (1.0 / 3.0) if value > 0.0088564517 else (7.787037 * value) + (16.0 / 116.0)

    fx, fy, fz = pivot(x), pivot(y), pivot(z)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def delta_e_2000(left_rgb: Sequence[float], right_rgb: Sequence[float]) -> float:
    """Return CIEDE2000 for two sRGB colors without an external color package."""
    l1, a1, b1 = _rgb_to_lab(left_rgb)
    l2, a2, b2 = _rgb_to_lab(right_rgb)
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0
    g = 0.5 * (1.0 - math.sqrt((c_bar ** 7) / (c_bar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)

    def hue(a_value: float, b_value: float) -> float:
        if abs(a_value) < 1e-12 and abs(b_value) < 1e-12:
            return 0.0
        value = math.degrees(math.atan2(b_value, a_value))
        return value + 360.0 if value < 0.0 else value

    h1p, h2p = hue(a1p, b1), hue(a2p, b2)
    delta_l = l2 - l1
    delta_c = c2p - c1p
    if c1p * c2p == 0.0:
        delta_h = 0.0
    elif abs(h2p - h1p) <= 180.0:
        delta_h = h2p - h1p
    elif h2p <= h1p:
        delta_h = h2p - h1p + 360.0
    else:
        delta_h = h2p - h1p - 360.0
    delta_big_h = 2.0 * math.sqrt(c1p * c2p) * math.sin(math.radians(delta_h / 2.0))
    l_bar = (l1 + l2) / 2.0
    c_bar_p = (c1p + c2p) / 2.0
    if c1p * c2p == 0.0:
        h_bar_p = h1p + h2p
    elif abs(h1p - h2p) <= 180.0:
        h_bar_p = (h1p + h2p) / 2.0
    elif h1p + h2p < 360.0:
        h_bar_p = (h1p + h2p + 360.0) / 2.0
    else:
        h_bar_p = (h1p + h2p - 360.0) / 2.0
    t = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar_p - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar_p))
        + 0.32 * math.cos(math.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar_p - 63.0))
    )
    delta_theta = 30.0 * math.exp(-((h_bar_p - 275.0) / 25.0) ** 2)
    r_c = 2.0 * math.sqrt((c_bar_p ** 7) / (c_bar_p ** 7 + 25.0 ** 7))
    s_l = 1.0 + (0.015 * (l_bar - 50.0) ** 2) / math.sqrt(20.0 + (l_bar - 50.0) ** 2)
    s_c = 1.0 + 0.045 * c_bar_p
    s_h = 1.0 + 0.015 * c_bar_p * t
    r_t = -math.sin(math.radians(2.0 * delta_theta)) * r_c
    return math.sqrt(
        (delta_l / s_l) ** 2
        + (delta_c / s_c) ** 2
        + (delta_big_h / s_h) ** 2
        + r_t * (delta_c / s_c) * (delta_big_h / s_h)
    )


def _cbrt(value: float) -> float:
    return math.copysign(abs(value) ** (1.0 / 3.0), value)


def _rgb_to_oklch(rgb: Sequence[int]) -> Tuple[float, float, float]:
    channels = []
    for value in rgb:
        normalized = float(value) / 255.0
        channels.append(normalized / 12.92 if normalized <= 0.04045 else ((normalized + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_root, m_root, s_root = _cbrt(l), _cbrt(m), _cbrt(s)
    lightness = 0.2104542553 * l_root + 0.7936177850 * m_root - 0.0040720468 * s_root
    a = 1.9779984951 * l_root - 2.4285922050 * m_root + 0.4505937099 * s_root
    b_axis = 0.0259040371 * l_root + 0.7827717662 * m_root - 0.8086757660 * s_root
    chroma = math.hypot(a, b_axis)
    hue = (math.degrees(math.atan2(b_axis, a)) + 360.0) % 360.0 if chroma > 1e-8 else 0.0
    return lightness, chroma, hue


def family_from_hex(value: str) -> str:
    lightness, chroma, hue = _rgb_to_oklch(_hex_to_rgb(value))
    if lightness < 0.27:
        return "black"
    if lightness > 0.9 and chroma < 0.07:
        return "white"
    if lightness > 0.68 and 45.0 <= hue < 105.0 and chroma >= 0.02:
        return "beige"
    if chroma < 0.045:
        return "gray"
    if lightness < 0.55 and 20.0 <= hue < 85.0:
        return "brown"
    if 330.0 <= hue or hue < 20.0:
        return "red"
    if hue < 58.0:
        return "orange"
    if hue < 105.0:
        return "yellow"
    if hue < 172.0:
        return "green"
    if hue < 215.0:
        return "cyan"
    if hue < 272.0:
        return "blue"
    return "purple"


def _conditioning_color_label(family: str, rgb: Sequence[int]) -> str:
    """Create a factual color descriptor; marketing names stay UI-only."""
    lightness, chroma, _hue = _rgb_to_oklch(rgb)
    if family in {"black", "gray"}:
        if lightness < 0.36:
            return "深中性灰黑" if family == "black" else "深灰"
        if lightness < 0.62:
            return "中性灰黑" if family == "black" else "中性灰"
        return "浅灰黑" if family == "black" else "浅灰"
    if family == "white":
        return "高明度中性白"
    if family == "beige":
        return "浅暖米色" if chroma < 0.12 else "浅暖沙色"
    family_word = {
        "brown": "棕",
        "blue": "蓝",
        "green": "绿",
        "red": "红",
        "orange": "橙",
        "purple": "紫",
        "yellow": "黄",
        "cyan": "青",
        "custom": "综合色",
    }.get(family, "综合色")
    if family == "orange" and lightness > 0.72 and chroma < 0.14:
        return "浅暖杏橙"
    if lightness < 0.36:
        return f"深{family_word}"
    if lightness > 0.72:
        return f"浅{family_word}色"
    if chroma < 0.10:
        return f"低饱和{family_word}色"
    return f"{family_word}色"


def _style_id(value: Any, text: str = "") -> str:
    candidate = _clean_text(value).casefold()
    for style, aliases in _STYLE_ALIASES.items():
        if candidate in {alias.casefold() for alias in aliases}:
            return style
    normalized = text.casefold()
    for style, aliases in _STYLE_ALIASES.items():
        if any(alias.casefold() in normalized for alias in aliases):
            return style
    return "custom"


def _style_hex(style_id: str, family: str) -> str:
    return _STYLE_FAMILY_HEXES.get(style_id, {}).get(family, _FAMILY_BASE_HEX.get(family, _FAMILY_BASE_HEX["custom"]))


def _family_from_name(value: Any) -> Optional[str]:
    text = _clean_text(value)
    for alias, family in _FAMILY_MATCH_TERMS:
        if alias.casefold() in text.casefold():
            return family
    return None


def _name_from_color(style_id: str, family: str, hex_value: str) -> str:
    if hex_value in _KNOWN_COLOR_NAMES:
        return _KNOWN_COLOR_NAMES[hex_value]
    hints = _STYLE_NAME_HINTS.get(style_id, {}).get(family) or _STYLE_NAME_HINTS.get("offroad", {}).get(family)
    if not hints:
        hints = (f"{family_label(family)}色",)
    digest = hashlib.sha256(f"{style_id}|{family}|{hex_value}".encode("utf-8")).digest()
    return hints[digest[0] % len(hints)]


def _load_name_cache(cache_path: Optional[Path]) -> Dict[str, str]:
    if cache_path is None:
        return {}
    try:
        payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_name_cache(cache_path: Optional[Path], payload: Dict[str, str]) -> None:
    if cache_path is None:
        return
    path = Path(cache_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(descriptor)
        temp_path = Path(temp_name)
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(temp_path, path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)  # type: ignore[possibly-undefined]
        except Exception:
            pass


def name_colors(colors: Sequence[Dict[str, Any]], style_id: str, cache_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    disk_cache = _load_name_cache(cache_path)
    changed = False
    result: List[Dict[str, Any]] = []
    for color in colors:
        current = dict(color)
        if not current.get("name"):
            key = f"{style_id}|{current['family']}|{current['hex']}".casefold()
            cached_name = _COLOR_NAME_CACHE.get(key) or disk_cache.get(key)
            name = cached_name
            if not name:
                name = _name_from_color(style_id, current["family"], current["hex"])
            _COLOR_NAME_CACHE[key] = name
            disk_cache[key] = name
            current["name"] = name
            current["name_source"] = "cache" if cached_name else "deterministic"
            changed = True
        else:
            current["name_source"] = "user"
        result.append(current)
    if changed:
        _save_name_cache(cache_path, disk_cache)
    return result


def _assign_color_ids(colors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for index, color in enumerate(colors, start=1):
        color["color_id"] = f"C{index}"
    return colors


def _extract_family_mentions(text: str) -> List[Tuple[str, int, int, str]]:
    mentions = []
    for match in _FAMILY_PATTERN.finditer(text or ""):
        alias = match.group(0)
        family = normalize_family(alias)
        if family:
            last_open = max((text.rfind(token, 0, match.start()) for token in ("（", "(")), default=-1)
            last_close = max((text.rfind(token, 0, match.start()) for token in ("）", ")")), default=-1)
            if last_open > last_close and any(previous[2] <= last_open for previous in mentions):
                # A generic word such as "黑色" inside "黑色系（炭黑色，...）"
                # belongs to the name, not to a second palette entry.
                continue
            mentions.append((family, match.start(), match.end(), alias))
    return mentions


def _extract_color_mentions(text: str) -> List[Dict[str, Any]]:
    source = text or ""
    family_mentions = _extract_family_mentions(source)
    hex_matches = list(re.finditer(r"#[0-9a-fA-F]{3,6}", source))
    used_hex = set()
    colors: List[Dict[str, Any]] = []
    for index, (family, _start, end, _alias) in enumerate(family_mentions):
        segment_end = family_mentions[index + 1][1] if index + 1 < len(family_mentions) else len(source)
        segment = source[end:segment_end]
        candidate_hex = next((match for match in hex_matches if end <= match.start() < segment_end), None)
        hex_value = normalize_hex(candidate_hex.group(0)) if candidate_hex else None
        if candidate_hex:
            used_hex.add(candidate_hex.start())
        name = ""
        if candidate_hex:
            before_hex = segment[: candidate_hex.start() - end]
            parenthetical = re.search(r"[（(]\s*([^（）()#]{1,48}?)\s*[,，;；]?\s*$", before_hex)
            candidate_name = parenthetical.group(1) if parenthetical else before_hex
            candidate_name = re.sub(r"^[\s,，:：/、（(]+|[\s,，:：/、）)]+$", "", candidate_name)
            if candidate_name and not normalize_family(candidate_name):
                name = candidate_name
        colors.append({"family": family, "hex": hex_value, "name": name, "locked": bool(hex_value)})
    for match in hex_matches:
        if match.start() in used_hex:
            continue
        value = normalize_hex(match.group(0))
        colors.append({"family": family_from_hex(value), "hex": value, "name": "", "locked": True})
    colors.sort(key=lambda item: source.find(item.get("hex") or item.get("family") or ""))
    return colors


def extract_materials(text: str) -> List[str]:
    matches: List[Tuple[int, str]] = []
    source = text or ""
    for alias, material in _MATERIAL_MATCH_TERMS:
        for match in re.finditer(re.escape(alias), source, flags=re.IGNORECASE):
            matches.append((match.start(), material))
    return list(dict.fromkeys(material for _position, material in sorted(matches)))


def extract_components(text: str) -> List[str]:
    """Extract stable CMF regions for material-aware retrieval diagnostics."""
    return _extract_components(text)


def _normalize_color_entry(raw: Any, style_id: str, index: int) -> Dict[str, Any]:
    if isinstance(raw, str):
        raw = {"family": raw}
    if not isinstance(raw, dict):
        raise CMFRequestError(f"colors[{index}] 必须是对象。")
    raw_hex = raw.get("hex")
    raw_rgb = _normalize_rgb(raw.get("rgb"))
    hex_value = normalize_hex(raw_hex) if raw_hex is not None and _clean_text(raw_hex) else None
    if raw_rgb is not None:
        rgb_hex = _rgb_to_hex(raw_rgb)
        if hex_value is not None and hex_value != rgb_hex:
            raise CMFRequestError(f"colors[{index}] 的 RGB 与 HEX 不一致。")
        hex_value = rgb_hex
    family = normalize_family(raw.get("family") or raw.get("color_family"))
    name = _clean_text(raw.get("name") or raw.get("color_name"))
    if family is None:
        family = _family_from_name(name)
    detected_family = family_from_hex(hex_value) if hex_value else None
    if family is None:
        family = detected_family or "custom"
    if family not in COLOR_FAMILIES:
        raise CMFRequestError(f"colors[{index}] 的色系不可识别。")
    if hex_value is None:
        hex_value = _style_hex(style_id, family)
    rgb = list(_hex_to_rgb(hex_value))
    role = _clean_text(raw.get("role")).casefold() or ("primary" if index == 0 else "secondary" if index == 1 else "accent")
    if role not in {"primary", "secondary", "accent", "neutral"}:
        raise CMFRequestError(f"colors[{index}].role 必须是 primary、secondary、accent 或 neutral。")
    explicit_value = raw_hex is not None and _clean_text(raw_hex) or raw_rgb is not None
    return {
        "family": family,
        "family_label": family_label(family),
        "hex": hex_value,
        "rgb": rgb,
        "conditioning_label": _conditioning_color_label(family, rgb),
        "name": name,
        "role": role,
        "locked": bool(explicit_value) or bool(raw.get("locked", False)),
        "detected_family": detected_family,
        "family_mismatch": bool(detected_family and family != detected_family),
    }


def _normalize_material_list(value: Any, fallback_text: str) -> List[str]:
    raw_values = value if isinstance(value, list) else ([value] if isinstance(value, str) and value.strip() else [])
    materials: List[str] = []
    for raw in raw_values:
        material = normalize_material(raw)
        if material is None:
            raise CMFRequestError(f"不可识别的材质：{raw!r}。")
        if material not in materials:
            materials.append(material)
    if not materials:
        materials = extract_materials(fallback_text)
    return materials


def normalize_cmf_request(
    value: Any,
    *,
    fallback_text: str = "",
    dataset_metadata: Optional[Dict[str, Any]] = None,
    cache_path: Optional[Path] = None,
    require_colors: bool = False,
    default_auxiliary_color_strategy: str = "none",
) -> Dict[str, Any]:
    """Normalize JSON or the current short Chinese label text into one request."""
    dataset_metadata = dataset_metadata or {}
    raw_value = value
    if isinstance(value, str):
        text_value = value.strip()
        if text_value.startswith("{"):
            try:
                raw_value = json.loads(text_value)
            except json.JSONDecodeError as exc:
                raise CMFRequestError(f"cmf_request_json 不是有效 JSON：{exc.msg}。") from exc
        else:
            raw_value = {}
    elif value is None:
        raw_value = {}
    if raw_value and not isinstance(raw_value, dict):
        raise CMFRequestError("CMF 请求必须是 JSON 对象或中文标签文本。")
    raw = dict(raw_value or {})
    source_text = _clean_text(fallback_text or (value if isinstance(value, str) else ""))
    style_candidate = raw.get("style_id") or raw.get("style")
    style_id = _style_id(style_candidate, source_text) if style_candidate or source_text else str(dataset_metadata.get("style_id") or "custom")
    if style_id == "custom" and dataset_metadata.get("style_id"):
        style_id = _style_id(dataset_metadata.get("style_id"), "")
    style_phrase = _clean_text(raw.get("style_phrase"))
    if not style_phrase:
        style_phrase = STYLE_PHRASES.get(style_id) or _clean_text(dataset_metadata.get("style_phrase")) or STYLE_PHRASES["custom"]

    raw_colors = raw.get("colors")
    if isinstance(raw_colors, str):
        raw_colors = _extract_color_mentions(raw_colors)
    if raw_colors is None:
        raw_colors = _extract_color_mentions(source_text)
    if not isinstance(raw_colors, list):
        raise CMFRequestError("colors 必须是数组。")
    colors = [_normalize_color_entry(item, style_id, index) for index, item in enumerate(raw_colors)]
    if len(colors) > 8:
        raise CMFRequestError("最多支持 8 个色彩输入。")
    colors = _assign_color_ids(name_colors(colors, style_id, cache_path))
    if require_colors and not colors:
        raise CMFRequestError("至少需要一个颜色。")

    materials = _normalize_material_list(raw.get("materials"), source_text)
    scope = _clean_text(raw.get("scope")).casefold() or "full_cabin"
    if scope not in set(SAMPLE_TYPES):
        raise CMFRequestError("scope 必须是 full_cabin、color_material 或 detail。")
    palette_policy = _clean_text(raw.get("palette_policy")).casefold() or "balanced"
    if palette_policy not in {"balanced", "input_only", "strict"}:
        raise CMFRequestError("palette_policy 必须是 balanced、input_only 或 strict。")
    default_strategy = _clean_text(default_auxiliary_color_strategy).casefold() or "reuse_secondary"
    if default_strategy not in AUXILIARY_COLOR_STRATEGIES:
        raise CMFRequestError(
            "default_auxiliary_color_strategy 必须是 reuse_secondary、dataset、style 或 none。"
        )
    auxiliary_color_strategy = _clean_text(raw.get("auxiliary_color_strategy")).casefold()
    if not auxiliary_color_strategy:
        auxiliary_color_strategy = "none" if palette_policy in {"input_only", "strict"} else default_strategy
    if auxiliary_color_strategy not in AUXILIARY_COLOR_STRATEGIES:
        raise CMFRequestError(
            "auxiliary_color_strategy 必须是 reuse_secondary、dataset、style 或 none。"
        )
    if auxiliary_color_strategy == "none" and not colors:
        raise CMFRequestError("auxiliary_color_strategy=none 时至少需要一个用户颜色。")
    material_policy = _clean_text(raw.get("material_policy")).casefold() or "strict"
    if material_policy not in {"strict", "prefer", "dataset"}:
        raise CMFRequestError("material_policy 必须是 strict、prefer 或 dataset。")
    trigger_mode = _clean_text(raw.get("trigger_mode")).casefold() or "inline"
    if trigger_mode not in {"inline", "prefix", "none"}:
        raise CMFRequestError("trigger_mode 必须是 inline、prefix 或 none。")
    return {
        "schema_version": 1,
        "style_id": style_id,
        "style_phrase": style_phrase,
        "scope": scope,
        "colors": colors,
        "materials": materials,
        "preserve_geometry": bool(raw.get("preserve_geometry", True)),
        "preserve_reference_color": bool(raw.get("preserve_reference_color", False)),
        "palette_policy": palette_policy,
        "auxiliary_color_strategy": auxiliary_color_strategy,
        "material_policy": material_policy,
        "trigger_mode": trigger_mode,
        "deterministic": bool(raw.get("deterministic", True)),
        "source_text": source_text,
        "warnings": [
            f"{color['family_label']} 的 HEX 与 OKLCH 推断色系为 {family_label(color['detected_family'])}，保留了用户显式色系。"
            for color in colors
            if color.get("family_mismatch") and color.get("detected_family")
        ],
    }


def request_to_retrieval_text(request: Dict[str, Any]) -> str:
    colors = "、".join(
        f"{item['family_label']} {item['name']} {item['hex']}" for item in request.get("colors", [])
    )
    materials = "、".join(material_label(item) for item in request.get("materials", []))
    return "，".join(
        part for part in (
            request.get("style_phrase", ""),
            request.get("scope", ""),
            colors,
            materials,
            request.get("source_text", ""),
        ) if part
    )


def build_cmf_image_conditioning_contract(
    request: Dict[str, Any],
    *,
    reference_image_count: int = 0,
) -> Dict[str, Any]:
    """Describe the image-conditioning handoff without pretending to run ControlNet."""
    has_reference = int(reference_image_count) > 0
    return {
        "reference_image_count": max(0, int(reference_image_count)),
        "input_image_required": has_reference,
        "preserve_geometry": bool(request.get("preserve_geometry", True)),
        "preserve_reference_color": bool(request.get("preserve_reference_color", False)),
        "color_condition": "exact_hex_rgb_text_and_color_reference" if has_reference else "exact_hex_rgb_text",
        "material_condition": "text_locked_plus_material_reference_samples",
        "recommended_controls": ["depth", "lineart"] if has_reference else [],
        "recommended_control_strength": {"depth": 0.55, "lineart": 0.35} if has_reference else {},
        "recommended_denoise": 0.32 if has_reference and request.get("preserve_geometry", True) else 0.45,
        "post_generation_check": {
            "required": True,
            "color_measurement": "masked_lab_delta_e_2000",
            "material_check": "visual_or_vlm_check_per_requested_material",
            "geometry_check": "reference_structure_and_view_consistency",
        },
    }


def _classify_caption(caption: str) -> str:
    text = _clean_text(caption).casefold()
    if any(term.casefold() in text for term in _FULL_CABIN_HINTS):
        return "full_cabin"
    component_names = {
        component
        for term, component in _COMPONENT_MATCH_TERMS
        if term.casefold() in text
    }
    component_hit = bool(component_names)
    material_hit = any(term.casefold() in text for term, _material in _MATERIAL_MATCH_TERMS)
    color_hit = bool(_COLOR_HINT_PATTERN.search(text))
    if len(component_names) >= 4 or (
        len(component_names) >= 2 and any(token in text for token in ("内饰", "座舱", "整体", "interior", "cabin"))
    ):
        return "full_cabin"
    if component_hit and material_hit:
        return "detail"
    if color_hit or material_hit:
        return "color_material"
    return "detail"


def classify_caption(caption: str) -> List[str]:
    """Return one stable sample type used by retrieval and bundle metadata."""
    return [_classify_caption(caption)]


def _extract_components(text: str) -> List[str]:
    hits: List[Tuple[int, str]] = []
    for alias, component in _COMPONENT_MATCH_TERMS:
        for match in re.finditer(re.escape(alias), text or "", flags=re.IGNORECASE):
            hits.append((match.start(), component))
    return list(dict.fromkeys(component for _position, component in sorted(hits)))


def _extract_plan_materials(text: str) -> List[str]:
    return extract_materials(text)


def _palette_from_retrieved(
    retrieved: Sequence[Dict[str, Any]],
    style_id: str,
    *,
    fallback: bool = True,
) -> List[Dict[str, Any]]:
    found = _extract_color_mentions("\n".join(str(item.get("caption") or "") for item in retrieved))
    colors: List[Dict[str, Any]] = []
    seen = set()
    for index, item in enumerate(found):
        normalized = _normalize_color_entry(item, style_id, index)
        key = normalized["family"]
        if key in seen:
            continue
        seen.add(key)
        normalized["role"] = _COLOR_ROLES[len(colors)]
        normalized["source"] = "dataset"
        colors.append(normalized)
        if len(colors) == 4:
            break
    if colors or not fallback:
        return _assign_color_ids(name_colors(colors, style_id))
    return _assign_color_ids([
        {
            "family": family,
            "family_label": family_label(family),
            "hex": _style_hex(style_id, family),
            "rgb": list(_hex_to_rgb(_style_hex(style_id, family))),
            "conditioning_label": _conditioning_color_label(
                family, _hex_to_rgb(_style_hex(style_id, family))
            ),
            "name": _name_from_color(style_id, family, _style_hex(style_id, family)),
            "role": role,
            "locked": False,
            "detected_family": family,
            "family_mismatch": False,
            "source": "fallback",
        }
        for role, family in (("primary", "black"), ("secondary", "gray"))
    ])


def _style_palette(style_id: str) -> List[Dict[str, Any]]:
    palette = _STYLE_PALETTE_FAMILIES.get(style_id) or _STYLE_PALETTE_FAMILIES["custom"]
    colors = [
        _normalize_color_entry({"family": family, "role": role}, style_id, index)
        for index, (role, family) in enumerate(palette)
    ]
    for color in colors:
        color["source"] = "style"
    return _assign_color_ids(name_colors(colors, style_id))


def _fill_missing_color_roles(
    colors: List[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    used_roles = {str(item.get("role") or "") for item in colors}
    used_families = {str(item.get("family") or "") for item in colors}
    added: List[Dict[str, Any]] = []
    for candidate in candidates:
        family = str(candidate.get("family") or "")
        if not family or family in used_families:
            continue
        missing_roles = [role for role in _COLOR_ROLES if role not in used_roles]
        if not missing_roles:
            break
        preferred_role = str(candidate.get("role") or "")
        role = preferred_role if preferred_role in missing_roles else missing_roles[0]
        current = dict(candidate)
        current["role"] = role
        colors.append(current)
        added.append(current)
        used_roles.add(role)
        used_families.add(family)
    return added


def _resolve_plan_colors(
    request_colors: Sequence[Dict[str, Any]],
    strategy: str,
    retrieved: Sequence[Dict[str, Any]],
    style_id: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    colors = [dict(item) for item in request_colors]
    for color in colors:
        color.setdefault("source", "user")

    auxiliary_colors: List[Dict[str, Any]] = []
    if strategy == "none":
        if not colors:
            raise CMFRequestError("auxiliary_color_strategy=none 时至少需要一个用户颜色。")
        return _assign_color_ids(colors), auxiliary_colors

    if strategy == "reuse_secondary":
        if not colors:
            fallback = _palette_from_retrieved(retrieved, style_id)
            primary = dict(fallback[0])
            primary["role"] = "primary"
            colors.append(primary)
            auxiliary_colors.append(primary)
        return _assign_color_ids(colors), auxiliary_colors

    candidates = (
        _palette_from_retrieved(retrieved, style_id, fallback=False)
        if strategy == "dataset"
        else _style_palette(style_id)
    )
    auxiliary_colors.extend(_fill_missing_color_roles(colors, candidates))
    if not colors:
        raise CMFRequestError(
            "auxiliary_color_strategy=dataset 时检索结果没有可用颜色，请输入至少一个颜色或改用 style。"
        )
    return _assign_color_ids(colors), auxiliary_colors


def _component_from_value(value: Any) -> Optional[str]:
    text = _clean_text(value).casefold()
    if not text:
        return None
    for component, aliases in _COMPONENT_ALIASES.items():
        if text in {alias.casefold() for alias in aliases}:
            return component
    return None


def _choose_material(component: str, materials: Sequence[str], index: int) -> str:
    if not materials:
        return "皮质"
    for candidate in _MATERIAL_COMPATIBILITY.get(component, ()):
        if candidate in materials:
            return candidate
    raise CMFRequestError(f"材质 {materials[index % len(materials)]} 无法应用到部件 {component}。")


def _build_material_assignments(materials: Sequence[str]) -> List[str]:
    """Choose compatible assignments while covering every requested material."""
    normalized = list(dict.fromkeys(materials))
    if len(normalized) > len(_COMPONENTS):
        raise CMFRequestError(
            f"当前 CMF 模板只有 {len(_COMPONENTS)} 个可分配区域，无法覆盖 {len(normalized)} 种材质。"
        )
    options = []
    for component in _COMPONENTS:
        candidates = tuple(material for material in _MATERIAL_COMPATIBILITY[component] if material in normalized)
        if not candidates:
            # A single selected material is a deliberate monochrome material
            # treatment; use it consistently instead of silently inventing a
            # material for an uncovered component.
            candidates = (normalized[0],)
        options.append(candidates)
    for material in normalized:
        if not any(material in candidates for candidates in options):
            raise CMFRequestError(f"材质 {material_label(material)} 没有可兼容的座舱部件。")

    required = set(normalized)
    best_assignment: Optional[Tuple[str, ...]] = None
    best_score: Optional[Tuple[int, int]] = None
    for candidate in itertools.product(*options):
        if not required.issubset(set(candidate)):
            continue
        preference_score = sum(
            len(_MATERIAL_COMPATIBILITY[component])
            - (
                _MATERIAL_COMPATIBILITY[component].index(material)
                if material in _MATERIAL_COMPATIBILITY[component]
                else len(_MATERIAL_COMPATIBILITY[component]) + 5
            )
            for component, material in zip(_COMPONENTS, candidate)
        )
        coverage_score = len(set(candidate))
        score = (coverage_score, preference_score)
        if best_score is None or score > best_score:
            best_assignment = candidate
            best_score = score
    if best_assignment is None:
        raise CMFRequestError("当前部件模板无法同时覆盖所有用户材质。")
    return list(best_assignment)


def _repair_material_coverage(
    assignments: List[Dict[str, Any]],
    materials: Sequence[str],
) -> None:
    """Repair model suggestions that silently dropped a requested material."""
    required = set(materials)
    counts = {material: 0 for material in required}
    for assignment in assignments:
        material = str(assignment.get("material") or "")
        counts[material] = counts.get(material, 0) + 1

    for missing in (material for material in materials if counts.get(material, 0) == 0):
        candidates = []
        for index, assignment in enumerate(assignments):
            component = str(assignment.get("component") or "")
            if missing not in _MATERIAL_COMPATIBILITY.get(component, ()):
                continue
            current = str(assignment.get("material") or "")
            preserves_required = int(current not in required or counts.get(current, 0) > 1)
            preference = _MATERIAL_COMPATIBILITY[component].index(missing)
            candidates.append((preserves_required, -preference, -index, index))
        if not candidates:
            raise CMFRequestError(f"模型方案无法覆盖材质 {material_label(missing)}。")
        _preserves, _preference, _index, selected_index = max(candidates)
        old_material = str(assignments[selected_index].get("material") or "")
        assignments[selected_index]["material"] = missing
        counts[old_material] = counts.get(old_material, 0) - 1
        counts[missing] = counts.get(missing, 0) + 1


def _choose_color(
    component: str,
    colors: Sequence[Dict[str, Any]],
    index: int,
    auxiliary_color_strategy: str,
) -> Dict[str, Any]:
    if not colors:
        raise CMFRequestError("CMF 方案没有可用颜色。")
    by_role = {item.get("role"): item for item in colors}
    component_role = {
        "座椅主面料": "primary",
        "门板内衬肌理": "secondary",
        "中控台面上层": "secondary",
        "马鞍区面板": "accent",
        "方向盘": "neutral",
    }.get(component)
    if component_role and by_role.get(component_role):
        return by_role[component_role]
    if component == "方向盘":
        for item in colors:
            if item["family"] == "black":
                return item
    if auxiliary_color_strategy == "reuse_secondary":
        return by_role.get("secondary") or by_role.get("primary") or colors[0]
    if component_role in {"accent", "neutral"}:
        return by_role.get("primary") or by_role.get("secondary") or colors[0]
    if component_role == "secondary":
        return by_role.get("primary") or colors[0]
    return colors[index % len(colors)]


def _apply_model_assignments(
    assignments: List[Dict[str, Any]],
    model_assignments: Optional[Sequence[Dict[str, Any]]],
    colors: Sequence[Dict[str, Any]],
    materials: Sequence[str],
) -> None:
    allowed_colors_by_id = {str(item.get("color_id")): item for item in colors if item.get("color_id")}
    allowed_colors_by_family = {item["family"]: item for item in colors}
    allowed_materials = set(materials)
    for raw in model_assignments or ():
        if not isinstance(raw, dict):
            continue
        component = _component_from_value(raw.get("component") or raw.get("component_id"))
        color_id = str(raw.get("color_id") or "").strip()
        family = normalize_family(raw.get("family") or raw.get("color_family"))
        material = normalize_material(raw.get("material"))
        color = allowed_colors_by_id.get(color_id) if color_id else allowed_colors_by_family.get(family)
        if (
            not component
            or color is None
            or not material
            or material not in allowed_materials
            or material not in _MATERIAL_COMPATIBILITY.get(component, ())
        ):
            continue
        for assignment in assignments:
            if assignment["component"] == component:
                assignment["color"] = color
                assignment["material"] = material
                break


def build_cmf_plan(
    request: Dict[str, Any],
    dataset_metadata: Optional[Dict[str, Any]] = None,
    retrieved: Sequence[Dict[str, Any]] = (),
    *,
    seed: int = 0,
    model_assignments: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a constrained CMF structure plan; model suggestions are optional."""
    dataset_metadata = dataset_metadata or {}
    style_id = str(request.get("style_id") or "custom")
    auxiliary_color_strategy = str(request.get("auxiliary_color_strategy") or "reuse_secondary").casefold()
    if auxiliary_color_strategy not in AUXILIARY_COLOR_STRATEGIES:
        raise CMFRequestError(
            "auxiliary_color_strategy 必须是 reuse_secondary、dataset、style 或 none。"
        )
    colors, auxiliary_colors = _resolve_plan_colors(
        request.get("colors") or [],
        auxiliary_color_strategy,
        retrieved,
        style_id,
    )
    materials = list(request.get("materials") or [])
    if not materials:
        captions = "\n".join(str(item.get("caption") or "") for item in retrieved)
        materials = _extract_plan_materials(captions)
    if not materials:
        materials = ["leather", "fabric", "suede"]
    if request.get("material_policy") == "dataset":
        supported = [normalize_material(item) for item in dataset_metadata.get("supported_materials") or []]
        materials = list(dict.fromkeys(item for item in supported if item)) or materials
    materials = list(dict.fromkeys(materials))
    material_assignments = _build_material_assignments(materials)

    assignments: List[Dict[str, Any]] = []
    for index, component in enumerate(_COMPONENTS):
        color = _choose_color(component, colors, index, auxiliary_color_strategy)
        material = material_assignments[index]
        assignments.append({"component": component, "color": color, "material": material})
    _apply_model_assignments(assignments, model_assignments, colors, materials)
    _repair_material_coverage(assignments, materials)

    triggers = [str(item).strip() for item in dataset_metadata.get("trigger_words") or [] if str(item).strip()]
    return {
        "schema_version": 1,
        "style_id": style_id,
        "style_phrase": str(request.get("style_phrase") or STYLE_PHRASES.get(style_id) or STYLE_PHRASES["custom"]),
        "scope": request.get("scope") or "full_cabin",
        "colors": colors,
        "auxiliary_colors": auxiliary_colors,
        "auxiliary_color_strategy": auxiliary_color_strategy,
        "materials": materials,
        "material_coverage": {
            "requested": list(materials),
            "used": sorted({item["material"] for item in assignments}),
            "missing": sorted(set(materials) - {item["material"] for item in assignments}),
        },
        "component_assignments": assignments,
        "trigger_words": triggers,
        "trigger_mode": request.get("trigger_mode") or "inline",
        "preserve_geometry": bool(request.get("preserve_geometry", True)),
        "deterministic": bool(request.get("deterministic", True)),
        "seed": int(seed),
        "source_sample_types": sorted(
            {
                sample_type
                for item in retrieved
                for sample_type in item.get("sample_types") or []
                if sample_type in SAMPLE_TYPES
            }
        ),
    }


def parse_cmf_structure_output(text: str) -> List[Dict[str, Any]]:
    """Extract only assignment objects from a model JSON response."""
    candidate = str(text or "").strip()
    candidate = re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", candidate, flags=re.IGNORECASE).strip()
    candidate = re.sub(r"```(?:json|text)?\s*|```", "", candidate, flags=re.IGNORECASE).strip()
    payload: Any = None
    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(candidate[index:])
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(payload, dict):
        return []
    assignments = payload.get("assignments") or payload.get("component_assignments")
    return list(assignments) if isinstance(assignments, list) else []


def _display_color(color: Dict[str, Any]) -> str:
    rgb = tuple(color.get("rgb") or _hex_to_rgb(str(color.get("hex") or "#808080")))
    label = str(color.get("conditioning_label") or _conditioning_color_label(str(color.get("family") or "custom"), rgb))
    rgb_text = ",".join(str(int(channel)) for channel in rgb)
    return f"{color['family_label']}（{label}，{color['hex']}，RGB({rgb_text})）"


def _color_key(color: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(color.get("family") or ""),
        str(color.get("hex") or "").casefold(),
        str(color.get("role") or ""),
    )


def render_cmf_prompt(plan: Dict[str, Any]) -> str:
    colors = list(plan.get("colors") or [])
    if not colors:
        raise CMFValidationError("无法渲染没有颜色的 CMF 方案。")
    primary = [item for item in colors if item.get("role") == "primary"] or colors[:1]
    secondary = [item for item in colors if item.get("role") == "secondary"]
    header_colors = primary + secondary
    assignments = list(plan.get("component_assignments") or [])
    assigned_keys = {_color_key(item.get("color") or {}) for item in assignments}
    header_keys = {_color_key(item) for item in header_colors}
    for item in colors:
        key = _color_key(item)
        if key not in assigned_keys and key not in header_keys:
            header_colors.append(item)
            header_keys.add(key)
    color_text = "与".join(_display_color(item) for item in header_colors)
    clauses = "；".join(
        f"{item['component']}使用{_display_color(item['color'])}的{material_label(item['material'])}"
        for item in assignments
    )
    prompt = (
        f"由原状态转变为{plan.get('style_phrase') or STYLE_PHRASES['custom']}，"
        f"以{color_text}为主色调，其中，{clauses}。"
    )
    trigger_words = [str(item).strip() for item in plan.get("trigger_words") or [] if str(item).strip()]
    if plan.get("trigger_mode") in {"inline", "prefix"}:
        missing = [word for word in trigger_words if word.casefold() not in prompt.casefold()]
        if missing:
            prompt = "、".join(missing) + "，" + prompt
    return re.sub(r"\s+", " ", prompt).strip()


def validate_cmf_prompt(prompt: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    value = str(prompt or "").strip()
    if not value:
        errors.append("最终提示词为空")
    if "\n" in value or "\r" in value:
        errors.append("最终提示词必须是一行")
    expected_hexes = {str(item.get("hex")).casefold() for item in plan.get("colors") or []}
    actual_hexes = {item.casefold() for item in re.findall(r"#[0-9a-fA-F]{6}", value)}
    if not expected_hexes.issubset(actual_hexes):
        errors.append("存在颜色没有渲染到最终提示词")
    if not actual_hexes.issubset(expected_hexes):
        errors.append("最终提示词包含未授权的 HEX")
    for item in plan.get("colors") or []:
        display = _display_color(item)
        if display not in value:
            errors.append(f"缺少颜色约束：{display}")
    for assignment in plan.get("component_assignments") or []:
        if assignment.get("component") not in value:
            errors.append(f"缺少部件：{assignment.get('component')}")
        material = material_label(str(assignment.get("material") or ""))
        if material not in value:
            errors.append(f"缺少材质：{material}")
    requested_materials = set(plan.get("materials") or [])
    used_materials = {str(item.get("material") or "") for item in plan.get("component_assignments") or []}
    missing_materials = requested_materials - used_materials
    if missing_materials:
        errors.append(
            "最终方案遗漏用户材质：" + "、".join(material_label(item) for item in sorted(missing_materials))
        )
    if plan.get("trigger_mode") != "none":
        for trigger in plan.get("trigger_words") or []:
            if str(trigger).casefold() not in value.casefold():
                errors.append(f"缺少触发词：{trigger}")
    if not value.startswith("由原状态转变为") and not any(
        value.startswith(str(trigger)) for trigger in plan.get("trigger_words") or []
    ):
        errors.append("不符合固定 CMF 模板开头")
    result = {"valid": not errors, "errors": errors, "hexes": sorted(actual_hexes)}
    if errors:
        raise CMFValidationError("；".join(errors))
    return result


def compile_cmf_prompt(
    request: Dict[str, Any],
    dataset_metadata: Optional[Dict[str, Any]] = None,
    retrieved: Sequence[Dict[str, Any]] = (),
    *,
    seed: int = 0,
    model_assignments: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    plan = build_cmf_plan(
        request,
        dataset_metadata,
        retrieved,
        seed=seed,
        model_assignments=model_assignments,
    )
    prompt = render_cmf_prompt(plan)
    validation = validate_cmf_prompt(prompt, plan)
    return prompt, plan, validation


__all__ = [
    "AUXILIARY_COLOR_STRATEGIES",
    "CMFRequestError",
    "CMFValidationError",
    "COLOR_FAMILY_LABELS",
    "COLOR_FAMILIES",
    "SAMPLE_TYPES",
    "SAMPLE_CLASSIFICATION_VERSION",
    "build_cmf_plan",
    "build_cmf_image_conditioning_contract",
    "classify_caption",
    "compile_cmf_prompt",
    "delta_e_2000",
    "extract_materials",
    "extract_components",
    "family_from_hex",
    "family_label",
    "material_label",
    "name_colors",
    "normalize_cmf_request",
    "normalize_family",
    "normalize_hex",
    "normalize_material",
    "parse_cmf_structure_output",
    "request_to_retrieval_text",
    "render_cmf_prompt",
    "validate_cmf_prompt",
]
