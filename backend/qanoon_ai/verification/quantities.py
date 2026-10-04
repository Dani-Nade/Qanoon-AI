"""Extract numeric quantities from English, Roman Urdu and Urdu legal text.

Used to check that an answer point states the same numbers as its evidence, and the
same kind of number: "within 14 days" is a duration, "by the fourteenth of each month"
is a day of the month.
"""

from __future__ import annotations

import re
import unicodedata

URDU_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

# Values of number words. Roman Urdu and Urdu words double as ordinary words
# ("do" = give, "ek" = a), so they only count when a unit follows.
ENGLISH_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
ENGLISH_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
ENGLISH_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50, "sixtieth": 60,
    "seventieth": 70, "eightieth": 80, "ninetieth": 90,
}
ROMAN_URDU = {
    "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6, "chay": 6,
    "saat": 7, "aath": 8, "nau": 9, "das": 10, "gyarah": 11, "barah": 12, "terah": 13, "chaudah": 14,
    "pandrah": 15, "sola": 16, "solah": 16, "satrah": 17, "atharah": 18, "athara": 18, "bees": 20,
    "bais": 22, "chaubees": 24, "pachees": 25, "tees": 30, "chalees": 40, "pachas": 50, "saath": 60,
    "nabbe": 90, "sau": 100,
}
URDU = {
    "ایک": 1, "دو": 2, "تین": 3, "چار": 4, "پانچ": 5, "چھ": 6, "سات": 7, "آٹھ": 8, "نو": 9, "دس": 10,
    "گیارہ": 11, "بارہ": 12, "تیرہ": 13, "چودہ": 14, "پندرہ": 15, "سولہ": 16, "سترہ": 17, "اٹھارہ": 18,
    "بیس": 20, "بائیس": 22, "چوبیس": 24, "پچیس": 25, "تیس": 30, "چالیس": 40, "پچاس": 50, "ساٹھ": 60,
    "نوے": 90, "سو": 100,
}
MULTIPLIERS = {
    "hundred": 100, "thousand": 1000, "lakh": 100000, "lac": 100000, "million": 1000000,
    "hazar": 1000, "hazaar": 1000, "ہزار": 1000, "لاکھ": 100000,
}

UNIT_KINDS = {
    "year": "year", "years": "year", "saal": "year", "sal": "year", "baras": "year", "سال": "year", "برس": "year",
    "month": "month", "months": "month", "mahine": "month", "mahina": "month", "maah": "month", "ماہ": "month", "مہینے": "month", "مہینہ": "month",
    "week": "week", "weeks": "week", "hafte": "week", "hafta": "week", "ہفتے": "week", "ہفتہ": "week",
    "day": "day", "days": "day", "din": "day", "dinon": "day", "دن": "day", "دنوں": "day", "یوم": "day", "ایام": "day",
    "سالوں": "year", "مہینوں": "month", "ہفتوں": "week", "گھنٹوں": "hour", "saalon": "year", "mahinon": "month",
    "hour": "hour", "hours": "hour", "ghante": "hour", "ghanta": "hour", "گھنٹے": "hour", "گھنٹہ": "hour",
    "rupee": "money", "rupees": "money", "rs": "money", "rupay": "money", "rupaye": "money", "روپے": "money", "روپیہ": "money",
    "percent": "percent", "fisad": "percent", "فیصد": "percent",
    "man": "person", "men": "person", "woman": "person", "women": "person", "members": "person", "member": "person",
    "witnesses": "person", "mard": "person", "aurat": "person", "auratein": "person", "gawah": "person",
    "مرد": "person", "عورت": "person", "عورتیں": "person", "گواہ": "person", "ارکان": "person",
    "tareekh": "date", "tarikh": "date", "تاریخ": "date",
}
TOKEN = re.compile(r"\d+(?:st|nd|rd|th)?|[^\W\d_]+", re.UNICODE)

# Provision references are not quantities: "Section 497(1)", "Article 10A", "sub-section (3)", "(ii)".
REFERENCES = re.compile(
    r"\b(?:sections?|articles?|clauses?|sub-?sections?|rules?|chapters?|parts?|schedules?|dafa|شق|دفعہ|آرٹیکل)"
    r"\s*\d+[a-z]?(?:\s*\(\s*\w+\s*\))*|\(\s*(?:\d+[a-z]?|[a-z]|[ivx]+)\s*\)|\b\d+\s*\[",
    re.IGNORECASE,
)


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold().translate(URDU_DIGITS)
    text = REFERENCES.sub(" ", text)
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    return TOKEN.findall(text.replace("-", " "))


def _number_at(tokens: list[str], i: int) -> tuple[int, int, bool, bool] | None:
    """Parse a number starting at tokens[i]: (value, tokens used, is_ordinal, is_word_needing_unit)."""
    token = tokens[i]
    if re.fullmatch(r"\d+(?:st|nd|rd|th)?", token):
        digits = re.match(r"\d+", token).group()
        value, used, ordinal, needs_unit = int(digits), 1, token != digits, False
    elif token in ENGLISH_ORDINALS:
        # "first appearance", "second marriage": low ordinals are ordinary words.
        return ENGLISH_ORDINALS[token], 1, True, ENGLISH_ORDINALS[token] <= 10
    elif token in ENGLISH_TENS:
        value, used, ordinal, needs_unit = ENGLISH_TENS[token], 1, False, False
        if i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt in ENGLISH_UNITS and ENGLISH_UNITS[nxt] < 10:
                value, used = value + ENGLISH_UNITS[nxt], 2
            elif nxt in ENGLISH_ORDINALS and ENGLISH_ORDINALS[nxt] < 10:
                return value + ENGLISH_ORDINALS[nxt], 2, True, False
    elif token in ENGLISH_UNITS:
        value, used, ordinal, needs_unit = ENGLISH_UNITS[token], 1, False, token == "one"
    elif token in ROMAN_URDU:
        value, used, ordinal, needs_unit = ROMAN_URDU[token], 1, False, True
    elif token in URDU:
        value, used, ordinal, needs_unit = URDU[token], 1, False, True
    else:
        return None
    while i + used < len(tokens) and tokens[i + used] in MULTIPLIERS:
        value *= MULTIPLIERS[tokens[i + used]]
        used += 1
        needs_unit = False
    return value, used, ordinal, needs_unit


def extract_quantities(text: str) -> set[tuple[int, str | None]]:
    """Return {(value, kind)}; kind is a unit class, "date", or None for a bare number."""
    tokens = _tokens(text)
    found: set[tuple[int, str | None]] = set()
    i = 0
    while i < len(tokens):
        parsed = _number_at(tokens, i)
        if parsed is None:
            i += 1
            continue
        value, used, ordinal, needs_unit = parsed
        following = tokens[i + used] if i + used < len(tokens) else ""
        kind = UNIT_KINDS.get(following)
        if kind == "day" and tokens[i + used + 1:i + used + 4] in (["of", "each", "month"], ["of", "every", "month"]):
            kind = "date"  # "by fourteen day of each month" is a date, not a duration
        if kind is None and ordinal and (following == "of" or not needs_unit):
            kind = "date"  # "fourteenth of each month", "14th"
        elif kind is None and needs_unit:
            i += used
            continue
        found.add((value, kind))
        i += used
    return found


def unsupported_quantities(claim: str, evidence: str) -> list[str]:
    """Quantities in the claim that the evidence does not state with a compatible kind."""
    available = extract_quantities(evidence)
    problems = []
    for value, kind in sorted(extract_quantities(claim), key=lambda item: (item[0], item[1] or "")):
        compatible = any(
            value == other and (kind == other_kind or kind is None or other_kind is None)
            for other, other_kind in available
        )
        if not compatible:
            problems.append(f"{value} {kind}" if kind else str(value))
    return problems
