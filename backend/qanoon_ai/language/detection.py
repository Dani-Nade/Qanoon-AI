"""Lightweight language detection for English, Urdu, and Roman Urdu."""

from __future__ import annotations

import re
from dataclasses import dataclass


URDU_RE = re.compile(r"[\u0600-\u06ff]")
LATIN_RE = re.compile(r"[A-Za-z]")

ROMAN_URDU_HINTS = {
    "kya",
    "hai",
    "hain",
    "mein",
    "main",
    "ka",
    "ki",
    "ke",
    "saza",
    "qanoon",
    "qanun",
    "chori",
    "zamanat",
    "talaq",
    "nikah",
    "virasat",
    "gawah",
    "kaun",
    "kon",
}

# Informal Roman Urdu spellings mapped to one canonical form before detection and search.
ROMAN_VARIANTS = {
    "agr": "agar", "agar": "agar", "mene": "maine", "maine": "maine", "mainay": "maine", "mein": "mein", "mai": "mein",
    "kese": "kaise", "kaisay": "kaise", "kesy": "kaise", "kaise": "kaise", "kse": "kaise",
    "tou": "to", "toh": "to", "kia": "kya", "kya": "kya", "kiya": "kya", "hy": "hai", "hay": "hai", "hain": "hain", "hen": "hain",
    "kr": "kar", "krna": "karna", "krni": "karni", "krny": "karne", "kren": "karen", "kro": "karo",
    "skta": "sakta", "skti": "sakti", "sakta": "sakta", "sakti": "sakti", "hu": "hoon", "hun": "hoon", "hoon": "hoon",
    "dounga": "dunga", "dunga": "dunga", "doon": "dun", "deni": "dena", "dena": "dena", "dy": "de", "dain": "dein",
    "mjhe": "mujhe", "mujhy": "mujhe", "mujhe": "mujhe", "mera": "mera", "meri": "meri", "mery": "mere",
    "bhai": "bhai", "bina": "bina", "baghair": "baghair", "wala": "wala", "kisi": "kisi", "koi": "koi",
    "chahiye": "chahiye", "chahye": "chahiye", "chaheye": "chahiye", "hoga": "hoga", "hogi": "hogi",
    "nahi": "nahi", "nai": "nahi", "nhi": "nahi", "kab": "kab", "kahan": "kahan", "kyun": "kyun",
    "talaaq": "talaq", "tlaq": "talaq", "zamant": "zamanat", "zmanat": "zamanat", "saza": "saza", "sza": "saza",
    "qanoon": "qanoon", "kanoon": "qanoon", "qanun": "qanoon", "adalat": "adalat", "court": "court",
}
ROMAN_URDU_HINTS.update({"agar", "maine", "kaise", "kya", "hai", "kar", "karna", "sakta", "sakti", "hoon", "dena", "dunga",
                         "mujhe", "mera", "meri", "kisi", "koi", "chahiye", "hoga", "nahi", "kab", "kahan", "kyun", "bhai", "bina"})

LEGAL_URDU_TO_ENGLISH = {
    "چوری": "theft",
    "سزا": "punishment",
    "قانون": "law",
    "عدالت": "court",
    "ضمانت": "bail",
    "طلاق": "divorce",
    "نکاح": "marriage",
    "وراثت": "inheritance",
}

ROMAN_URDU_TO_ENGLISH = {
    "chori": "theft",
    "saza": "punishment",
    "qanoon": "law",
    "qanun": "law",
    "zamanat": "bail",
    "talaq": "divorce",
    "nikah": "marriage",
    "virasat": "inheritance",
    "jaidad": "property",
    "adalt": "court",
    "adalat": "court",
    "gawah": "witness",
    "banne": "become",
    "kaun": "who",
    "kon": "who",
    "gawahi": "testimony", "nafaqa": "maintenance",
    "nafaqah": "maintenance", "kharcha": "maintenance", "kharch": "maintenance",
    "kharchay": "maintenance", "bachon": "children", "bache": "children",
    "biwi": "wife", "darkhwast": "application", "dawa": "claim",
    "tareeqa": "procedure", "tarika": "procedure", "zamanath": "bail",
    "zamanaat": "bail", "haqooq": "rights", "miras": "inheritance",
}

# Statutes use "talaq" and "nikah" themselves, so keep the word alongside its translation.
ROMAN_URDU_TO_ENGLISH.update({
    "talaq": "talaq divorce", "nikah": "nikah marriage",
    "dhamki": "threaten", "jaan": "death", "maarne": "kill", "qatl": "qatl",
    "hirasat": "custody", "ghante": "hours", "ghanta": "hours", "bina": "without", "baghair": "without",
    "rakh": "keep", "giraftar": "arrest", "giraftari": "arrest", "darj": "register",
    "das": "ten", "saal": "years", "din": "days", "umar": "age", "bacha": "child", "bachay": "child",
    "jurm": "offence", "muft": "free", "lazmi": "compulsory", "taleem": "education", "hukumat": "state",
    "qarz": "loan financial", "likhit": "writing", "muahide": "agreement instrument", "muahida": "agreement instrument",
    "mard": "men", "aurat": "woman", "auratein": "women", "shohar": "husband", "pehli": "existing",
    "doosri": "another", "shadi": "marriage", "ijazat": "permission", "muassar": "effective", "baad": "after",
    "sarkari": "civil", "mulazim": "servant", "gaari": "motor vehicle", "gari": "motor vehicle",
    "chalane": "drive", "motorcycle": "motorcycle", "sabit": "proof", "kam": "under",
})

# Roman Urdu phrases replaced before splitting into words.
ROMAN_URDU_PHRASES = {
    "na-qabil-e-zamanat": "non bailable", "naqabil-e-zamanat": "non bailable", "na qabil e zamanat": "non bailable",
    "qabil-e-zamanat": "bailable", "kam az kam": "minimum",
}

LEGAL_URDU_TO_ENGLISH.update({
    "ایف آئی آر": "fir", "درج": "register", "پولیس": "police", "مجسٹریٹ": "magistrate", "حراست": "custody",
    "حکم": "order", "بغیر": "without", "گھنٹے": "hours", "چیک": "cheque", "باؤنس": "bounce",
    "منصفانہ": "fair", "ٹرائل": "trial", "بنیادی": "fundamental", "مؤثر": "effective", "موثر": "effective",
    "بعد": "after", "دن": "days", "سال": "years", "عمر": "age", "حق": "right",
    "طلاق": "talaq divorce", "نکاح": "nikah marriage",
})

LEGAL_URDU_TO_ENGLISH.update({
    "گواہی": "testimony", "گواہ": "witness", "اہل": "competent",
    "نان و نفقہ": "maintenance", "نان نفقہ": "maintenance", "نفقہ": "maintenance",
    "خرچہ": "maintenance", "خرچ": "maintenance", "خاندانی": "family",
    "فیملی": "family", "دعویٰ": "claim", "دعوی": "claim", "درخواست": "application",
    "بچوں": "children", "بیوی": "wife", "طریقہ": "procedure",
    "حقوق": "rights", "ملازمین": "employees", "سرکاری": "civil servants",
    "چوری": "theft", "چور": "thief", "قید": "imprisonment",
    "جرمانہ": "fine", "ضمانت": "bail",
})

SEARCH_STOPWORDS = set("""
a an and are as at be been being by can could do does did explain for from get give
how i if in into is it its me my of on or please tell that the their them these they
this those to under us was were what when where which who whom why will with would
you your pakistan pakistani law laws legal process procedure provisions provision
quote according regarding about simple english urdu roman claim apply applying
mein main ka ki ke kya hai hain ho hota hoti hotay kaise kese kaisay ko se ne tak
liye liyeh aur ya mujhe mera meri mere hum ap aap batao batayein bataein sakta sakti
sakte ban banne kon kaun doon kar karna karnay hota mil milta milti hoga hogi
kitne kitni kitna kab chahiye honi hona karwate karwana dene dena karne pa par hote hue wala wali
kya kese kaisay jata jati jaati kis paband
agar maine dunga dun dena de dein mujhe mera meri mere bhai kisi koi chahiye hoon karni karen karo kahan kyun wala
someone anyone somebody happen happens happened give gives giving make makes must
punjab sindh balochistan khyber pakhtunkhwa kp پنجاب سندھ بلوچستان
کتنے کتنی کتنا کب چاہیے جاتی جاتا کروائی
پاکستان میں کی کا کے کیا ہے ہیں کیسے کون کو سے اور یا یہ وہ ایک لیے لئے مجھے ہوں
ہو سکتا سکتی سکتے دے دیں بتائیں ہوتا ہوتی کرنا کس پر بھی
""".split())


def normalize_legal_query(text: str) -> str:
    """Normalize legal vocabulary independently of the selected output language."""
    normalized = text.casefold().replace("ي", "ی").replace("ك", "ک")
    for phrase, replacement in ROMAN_URDU_PHRASES.items():
        normalized = normalized.replace(phrase, f" {replacement} ")
    for term, replacement in sorted(LEGAL_URDU_TO_ENGLISH.items(), key=lambda item: -len(item[0])):
        # Whole words only: "دن" (day) must not rewrite part of "بدن" or "دنیا".
        normalized = re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", f" {replacement} ", normalized)
    # Arabic-script punctuation (\u060c \u061b \u061f \u06d4) sits inside the Urdu block; strip it so
    # "\u06c1\u06d2\u061f" still matches the "\u06c1\u06d2" stopword.
    normalized = re.sub(r"[\u060c\u061b\u061f\u06d4]", " ", normalized)
    words = [ROMAN_VARIANTS.get(word, word) for word in re.findall(r"[a-z0-9]+|[\u0600-\u06ff]+", normalized)]
    normalized_words = [part for word in words for part in ROMAN_URDU_TO_ENGLISH.get(word, word).split()]
    return " ".join(word for word in normalized_words if word not in SEARCH_STOPWORDS)


@dataclass(frozen=True)
class DetectedLanguage:
    language: str
    script: str
    confidence: float


def detect_language(text: str, requested: str = "auto") -> DetectedLanguage:
    requested = (requested or "auto").strip().lower()
    if requested != "auto":
        script = "arabic" if requested == "urdu" else "latin"
        return DetectedLanguage(language=requested, script=script, confidence=1.0)

    if URDU_RE.search(text):
        return DetectedLanguage(language="urdu", script="arabic", confidence=0.98)

    tokens = {ROMAN_VARIANTS.get(t.lower(), t.lower()) for t in re.findall(r"[A-Za-z]+", text)}
    roman_hits = len(tokens & (ROMAN_URDU_HINTS | set(ROMAN_URDU_TO_ENGLISH)))
    if LATIN_RE.search(text) and roman_hits >= 2:
        return DetectedLanguage(language="roman_urdu", script="latin", confidence=0.82)

    return DetectedLanguage(language="english", script="latin", confidence=0.75)


def build_retrieval_queries(text: str, detected: DetectedLanguage) -> list[str]:
    """Normalize search independently of the requested response language."""

    normalized = normalize_legal_query(text)
    return [normalized] if normalized else []


def not_enough_sources_message(language: str) -> str:
    if language == "urdu":
        return "اس سوال کا معتبر جواب دینے کے لیے دستیاب قانونی ذرائع میں کافی معلومات نہیں ملیں۔ براہِ کرم متعلقہ قانون، صوبہ یا معاملے کی مزید تفصیل بتائیں۔"
    if language == "roman_urdu":
        return (
            "Mere paas is sawal ka reliable jawab dene ke liye kaafi "
            "verified legal source material nahi hai."
        )
    return (
        "I do not have enough verified legal source material indexed to answer "
        "this question safely."
    )
