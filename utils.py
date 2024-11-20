GROUP2LANG: dict[int, list[str]] = {
    1: ["da", "nl", "de", "is", "no", "sv", "af"],
    2: ["ca", "ro", "gl", "it", "pt", "es"],
    3: ["bg", "mk", "sr", "uk", "ru"],
    4: ["id", "ms", "th", "vi", "mg", "fr"],
    5: ["hu", "el", "cs", "pl", "lt", "lv"],
    6: ["ka", "zh", "ja", "ko", "fi", "et"],
    7: ["gu", "hi", "mr", "ne", "ur"],
    8: ["az", "kk", "ky", "tr", "uz", "ar", "he", "fa"],
}

LANG2GROUP = {lang: str(group) for group, langs in GROUP2LANG.items() for lang in langs}

LANG_TABLE = {
    "en": "English",
    "da": "Danish",
    "nl": "Dutch",
    "de": "German",
    "is": "Icelandic",
    "no": "Norwegian",
    "sv": "Swedish",
    "af": "Afrikaans",
    "ca": "Catalan",
    "ro": "Romanian",
    "gl": "Galician",
    "it": "Italian",
    "pt": "Portuguese",
    "es": "Spanish",
    "bg": "Bulgarian",
    "mk": "Macedonian",
    "sr": "Serbian",
    "uk": "Ukrainian",
    "ru": "Russian",
    "id": "Indonesian",
    "ms": "Malay",
    "th": "Thai",
    "vi": "Vietnamese",
    "mg": "Malagasy",
    "fr": "French",
    "hu": "Hungarian",
    "el": "Greek",
    "cs": "Czech",
    "pl": "Polish",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "ka": "Georgian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fi": "Finnish",
    "et": "Estonian",
    "gu": "Gujarati",
    "hi": "Hindi",
    "mr": "Marathi",
    "ne": "Nepali",
    "ur": "Urdu",
    "az": "Azerbaijani",
    "kk": "Kazakh",
    "ky": "Kyrgyz",
    "tr": "Turkish",
    "uz": "Uzbek",
    "ar": "Arabic",
    "he": "Hebrew",
    "fa": "Persian",
}


def get_language_name(target_lang: str) -> str:
    if target_lang not in LANG_TABLE:
        raise ValueError(f"Unsupported target language: {target_lang}")

    return LANG_TABLE[target_lang]


def get_xalma_model_name_by_group(group_id: int) -> str:
    if group_id not in GROUP2LANG:
        raise ValueError(f"Invalid group id: {group_id}")

    return f"haoranxu/X-ALMA-13B-Group{group_id}"


def get_xalma_model_name_by_lang(target_lang: str) -> str:
    if target_lang not in LANG2GROUP:
        raise ValueError(f"Unsupported target language: {target_lang}")

    group_id = LANG2GROUP[target_lang]
    return f"haoranxu/X-ALMA-13B-Group{group_id}"
