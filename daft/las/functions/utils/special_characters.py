# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import string

import emoji

MAIN_SPECIAL_CHARACTERS = string.punctuation + string.digits + string.whitespace
OTHER_SPECIAL_CHARACTERS = (
    "''"
    "–— ''　   ' ￼''"
    "–ー一▬…✦­£​•€«»°·═"
    "×士＾˘⇓↓↑←→（）§″′´¿−±∈¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‐⠀‰‑≤≥‖"
    "◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†上ン：∼⁄・♡✓⊕․．⋅÷１；،、¨ााी्े◦˚"
    "゜ʼ≖ʼ¤ッツシ℃√！【】‿∞➤～πه۩☛₨➩☻๑٪♥ı《'©٬？▷Г♫∟™ª₪®「—❖"
    "」》"
)
EMOJI = list(emoji.EMOJI_DATA.keys())
SPECIAL_CHARACTERS = set(MAIN_SPECIAL_CHARACTERS + OTHER_SPECIAL_CHARACTERS)
SPECIAL_CHARACTERS.update(EMOJI)

VARIOUS_WHITESPACES = {
    " ",
    "	",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    "　",
    "​",
    "‌",
    "‍",
    "⁠",
    "￼",
    "",
}

BULLET_POINTS = {
    "•",
    "-",
    "·",
    "●",
    "▪",
    "—",
    "*",
    "‣",
    "◦",
    "‧",
    "➤",
    "➔",
    "→",
    "»",
    "›",
    "–",
    "⁃",
    "⁌",
    "⁍",
    "∙",
    "○",
    "◘",
    "⦾",
    "⦿",
    "¤",
    "☑",
    "☐",
    "☛",
    "☞",
    "✔",
    "✦",
    "✧",
    "★",
    "☆",
    "✪",
    "➢",
    "➣",
    "❯",
    "▸",
}
