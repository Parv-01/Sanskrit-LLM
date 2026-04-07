
import re
import json
import unicodedata
from typing import List, Dict, Set, Tuple, Optional

# ── RapidFuzz (optional) with stdlib difflib fallback ────────────────────────
try:
    from rapidfuzz import process, fuzz
    _RAPIDFUZZ = True
except ImportError:
    import difflib
    _RAPIDFUZZ = False


# =============================================================================
# DICTIONARIES
# =============================================================================

HINGLISH_MAP: Dict[str, str] = {
    "tez bukhar":            "high fever",
    "bukhar":                "fever",
    "bukhaar":               "fever",
    "sir me dard":           "headache",
    "sar me dard":           "headache",
    "sir dard":              "headache",
    "sar dard":              "headache",
    "pet me dard":           "abdominal pain",
    "pait dard":             "abdominal pain",
    "pet dard":              "abdominal pain",
    "seene me dard":         "chest pain",
    "seene ka dard":         "chest pain",
    "kamar me dard":         "back pain",
    "kamar dard":            "back pain",
    "gale me dard":          "sore throat",
    "jodon me dard":         "joint pain",
    "jod dard":              "joint pain",
    "muscles me dard":       "muscle pain",
    "badan me dard":         "body ache",
    "body me dard":          "body ache",
    "badan dard":            "body ache",
    "pair dard":             "leg pain",
    "haath dard":            "arm pain",
    "ulti ho rahi":          "vomiting",
    "ulti hona":             "vomiting",
    "ulti":                  "vomiting",
    "ji machalna":           "nausea",
    "matli":                 "nausea",
    "loose motion":          "diarrhea",
    "dast":                  "diarrhea",
    "pet me gas":            "bloating",
    "qabz":                  "constipation",
    "kabz":                  "constipation",
    "bhooke nahi":           "loss of appetite",
    "bhook nahi":            "loss of appetite",
    "chakkrana":             "dizziness",
    "chakkara":              "dizziness",
    "sar ghoomna":           "dizziness",
    "chakkar":               "dizziness",
    "sans lene me takleef":  "difficulty breathing",
    "sans phoolna":          "shortness of breath",
    "naak se paani":         "runny nose",
    "naak jam gaya":         "nasal congestion",
    "naak bund":             "nasal congestion",
    "sukhhi khansi":         "dry cough",
    "khaansi":               "cough",
    "khansi":                "cough",
    "zukam":                 "cold",
    "jukam":                 "cold",
    "chheenk":               "sneezing",
    "awaaz band":            "hoarseness",
    "sar bhaari":            "heaviness in head",
    "aankhon me dard":       "eye pain",
    "aankhon se paani":      "watery eyes",
    "kaan me dard":          "ear pain",
    "kaan bhaari":           "ear fullness",
    "kamzoori":              "weakness",
    "kamzori":               "weakness",
    "thakaan":               "fatigue",
    "thakan":                "fatigue",
    "neend na aana":         "insomnia",
    "neend nahi":            "insomnia",
    "bahut neend":           "excessive sleepiness",
    "wazn kamm":             "weight loss",
    "bhaar badhna":          "weight gain",
    "bahut paseena":         "excessive sweating",
    "paseena":               "sweating",
    "thithurana":            "chills",
    "kaanpna":               "chills",
    "sardi":                 "chills",
    "khujli":                "itching",
    "sozish":                "swelling",
    "sujan":                 "swelling",
    "jalan":                 "burning sensation",
    "daane":                 "rash",
    "muh sukha":             "dry mouth",
    "pyas jyada":            "excessive thirst",
    "baar baar peshab":      "frequent urination",
    "peshab me jalan":       "painful urination",
    "dil ka tez dhakna":     "palpitations",
    "dil dhadakna":          "palpitations",
    "behoshi":               "fainting",
    "murcha":                "fainting",
    "dard":                  "pain",
    "peeda":                 "pain",
    "takleef":               "pain",
    "gas":                   "bloating",
    "acidity":               "acidity",
}

HINDI_MAP: Dict[str, str] = {
    # Fever / Temperature
    "तेज बुखार":               "high fever",
    "बुखार":                   "fever",
    "बुखार है":                "fever",
    # Head
    "सिर दर्द":                "headache",
    "सर दर्द":                 "headache",
    "सिर में दर्द":            "headache",
    "माइग्रेन":                "migraine",
    "सिर भारी":                "heaviness in head",
    # Chest / Body pain
    "सीने में दर्द":           "chest pain",
    "सीने का दर्द":            "chest pain",
    "कमर दर्द":                "back pain",
    "कमर में दर्द":            "back pain",
    "पेट में दर्द":            "abdominal pain",
    "पेट दर्द":                "abdominal pain",
    "जोड़ों में दर्द":          "joint pain",
    "मांसपेशियों में दर्द":    "muscle pain",
    "बदन दर्द":                "body ache",
    "शरीर में दर्द":           "body ache",
    "गले में दर्द":            "sore throat",
    "कान में दर्द":            "ear pain",
    "आँखों में दर्द":          "eye pain",
    "दर्द":                    "pain",
    # GI
    "उल्टी":                   "vomiting",
    "उल्टी हो रही है":         "vomiting",
    "मतली":                    "nausea",
    "जी मचलना":                "nausea",
    "दस्त":                    "diarrhea",
    "कब्ज":                    "constipation",
    "पेट में गैस":             "bloating",
    "भूख नहीं":                "loss of appetite",
    "अपच":                     "indigestion",
    # Respiratory
    "खाँसी":                   "cough",
    "खांसी":                   "cough",
    "सूखी खाँसी":              "dry cough",
    "सांस लेने में तकलीफ":    "difficulty breathing",
    "सांस लेने में दिक्कत":   "difficulty breathing",
    "सांस फूलना":              "shortness of breath",
    "नाक बंद":                 "nasal congestion",
    "नाक से पानी":             "runny nose",
    "छींक":                    "sneezing",
    "जुकाम":                   "cold",
    "जुखाम":                   "cold",
    # Neurological / General
    "चक्कर आना":               "dizziness",
    "चक्कर":                   "dizziness",
    "चक्कर आ रहे हैं":         "dizziness",
    "थकान":                    "fatigue",
    "थकावट":                   "fatigue",
    "कमजोरी":                  "weakness",
    "सुन्न":                   "numbness",
    "झनझनाहट":                 "tingling",
    "नींद नहीं":               "insomnia",
    "बेहोशी":                  "fainting",
    "भ्रम":                    "confusion",
    # Eyes / Skin
    "धुंधला दिखना":            "blurred vision",
    "आँखों से पानी":           "watery eyes",
    "खुजली":                   "itching",
    "सूजन":                    "swelling",
    "जलन":                     "burning sensation",
    "दाने":                    "rash",
    "पीलिया":                  "jaundice",
    # Cardio / Other
    "धड़कन तेज":               "palpitations",
    "दिल धड़कना":              "palpitations",
    "पसीना":                   "sweating",
    "ठंड लगना":                "chills",
    "कंपकंपी":                 "chills",
    "बार बार पेशाब":           "frequent urination",
    "पेशाब में जलन":           "painful urination",
    "प्यास ज्यादा":            "excessive thirst",
    "वजन कम":                  "weight loss",
    "मुँह सूखा":               "dry mouth",
    # Negation words (Devanagari) — also needed for negation detection
    "नहीं":                    "nahi",
    "नही":                     "nahi",
    "कोई नहीं":                "koi nahi",
    # Conjunctions / filler words — preserve scope-terminator semantics
    "लेकिन":                   "lekin",
    "परन्तु":                  "parantu",
    "मगर":                     "magar",
    "और":                      "aur",
    "लेकिन कि":                "lekin",
    # Common filler words to avoid garbled output
    "मुझे":                    "",
    "मुझको":                   "",
    "मेरे को":                 "",
    "मेरा":                    "",
    "हो रहा है":               "",
    "हो रही है":               "",
    "हो रहे हैं":              "",
    "है":                      "",
    "हैं":                     "",
    "से":                      "",
    "में":                     "",
    "कल":                      "",
    "आज":                      "",
    "बहुत":                    "",
}

def _build_hindi_regex() -> re.Pattern:
    keys = sorted(HINDI_MAP.keys(), key=len, reverse=True)
    return re.compile("(?:" + "|".join(re.escape(k) for k in keys) + ")")

_HINDI_RE = _build_hindi_regex()

def translate_hindi(text: str) -> str:
    """Replace Devanagari Hindi phrases with their English equivalents."""
    return _HINDI_RE.sub(lambda m: HINDI_MAP[m.group(0)], text)


SYMPTOM_DICTIONARY: List[str] = [
    "fever", "high fever", "low grade fever", "chills", "fatigue", "weakness",
    "weight loss", "weight gain", "night sweats", "excessive sweating", "sweating",
    "loss of appetite", "excessive thirst", "fainting", "body ache",
    "headache", "migraine", "chest pain", "abdominal pain", "back pain",
    "neck pain", "joint pain", "muscle pain", "leg pain", "arm pain",
    "ear pain", "eye pain", "sore throat", "throat ache",
    "painful urination", "blurred vision",
    "burning sensation", "heaviness in head", 
    "cough", "dry cough", "productive cough", "shortness of breath",
    "difficulty breathing", "wheezing", "runny nose", "nasal congestion",
    "sneezing", "cold", "hoarseness",
    "nausea", "vomiting", "diarrhea", "constipation", "bloating", "acidity",
    "indigestion", "blood in stool", "stomach cramps",
    "dizziness", "vertigo", "insomnia", "excessive sleepiness",
    "confusion", "memory loss", "tingling", "numbness", "tremors",
    "watery eyes", "ear fullness",
    "rash", "itching", "swelling", "jaundice", "frequent urination",
    "palpitations", "dry mouth",
]

SYMPTOM_SET: Set[str] = set(SYMPTOM_DICTIONARY)

# NegEx pre-negation triggers (Chapman et al., 2001)
PRE_NEG_CUES = [
    r"\bno\b", r"\bnot\b", r"\bnahi\b", r"\bnaheen\b",
    r"\bna\b(?!\w)", r"\bwithout\b", r"\bdenies\b", r"\bdenied\b",
    r"\babsence of\b", r"\bfree of\b", r"\bnegative for\b", r"\bnever\b",
    r"\bkoi nahi\b", r"\bnahi hai\b", r"\bkoi\b",
]

# Post-negation triggers (symptom comes before cue)
POST_NEG_CUES = [
    r"\bnahi hai\b", r"\bnahi\b", r"\bnaheen\b",
    r"\bnot present\b", r"\bwas not\b", r"\bis not\b",
]

# ConText extension: conjunctions that TERMINATE negation scope
# Harkema et al. (2009) — termination cues reset the negation window
SCOPE_TERMINATORS = re.compile(
    r"\b(?:but|however|although|except|yet|still|lekin|magar|par|parantu|aur|and|also|hai)\b",
    re.I
)

PRE_NEGATION_WINDOW  = 6   # tokens after pre-neg cue
POST_NEGATION_WINDOW = 3   # tokens before post-neg cue (tighter to avoid over-capture)
MAX_NGRAM            = 5
FUZZY_THRESHOLD      = 88


# =============================================================================
# COMPOUND WORD SPLITTER
# =============================================================================

# Known body-part and symptom root words. When two of these are fused without
# a space (e.g. "throatpain", "stomachache", "headpain"), we insert a space.
_BODY_PARTS = (
    "throat", "chest", "stomach", "abdomen", "belly", "back", "neck",
    "head", "ear", "eye", "leg", "arm", "shoulder", "knee", "ankle",
    "wrist", "hip", "groin", "skin", "scalp", "nose", "mouth", "hand",
    "foot", "feet", "finger", "toe",
)
_SYMPTOM_ROOTS = (
    "pain", "ache", "aches", "sore", "burn", "burning", "cramp", "cramps",
    "spasm", "swelling", "bleeding", "discharge", "loss", "vision",
    "breath", "breathing",
)

_COMPOUND_RE = re.compile(
    r"\b(" + "|".join(_BODY_PARTS) + r")(" + "|".join(_SYMPTOM_ROOTS) + r")\b",
    re.I,
)

def _split_compounds(text: str) -> str:
    """Insert a space between fused body-part + symptom-root tokens."""
    return _COMPOUND_RE.sub(r"\1 \2", text)


# =============================================================================
# STAGE 1: PREPROCESSING
# =============================================================================

def preprocess(text: str) -> str:
    """Normalize Unicode, lowercase, strip punctuation / extra whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = _split_compounds(text)          # split e.g. "throatpain" → "throat pain"
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\-']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# =============================================================================
# STAGE 2: HINGLISH -> ENGLISH TRANSLATION
# =============================================================================

def _build_hinglish_regex() -> re.Pattern:
    keys = sorted(HINGLISH_MAP.keys(), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b")

_HINGLISH_RE = _build_hinglish_regex()

# English body/symptom qualifier words that may precede a Hinglish word in
# code-mixed speech, e.g. "throat pain hai".  We skip Hinglish substitution
# when the match is immediately preceded by one of these qualifiers so the
# full English phrase (e.g. "throat pain") survives for fuzzy matching.
_ENG_QUALIFIERS = re.compile(
    r"\b(?:throat|chest|back|ear|eye|leg|arm|neck|knee|head|stomach|"
    r"joint|muscle|body|shoulder|ankle|wrist|hip|groin|skin|scalp)\s*$",
    re.I
)

def translate_hinglish(text: str) -> str:
    def _replacer(m: re.Match) -> str:
        # If an English body-part qualifier immediately precedes this match,
        # leave it untouched so "throat pain", "chest dard" etc. stay intact.
        preceding = text[: m.start()]
        if _ENG_QUALIFIERS.search(preceding):
            return m.group(0)
        return HINGLISH_MAP[m.group(0)]
    return _HINGLISH_RE.sub(_replacer, text)


# =============================================================================
# STAGE 3: NEGATION DETECTION  (NegEx + ConText)
# =============================================================================

_PRE_NEG_RE  = re.compile("|".join(f"(?:{c})" for c in PRE_NEG_CUES),  re.I)
_POST_NEG_RE = re.compile("|".join(f"(?:{c})" for c in POST_NEG_CUES), re.I)


def _token_char_starts(text: str, tokens: List[str]) -> List[int]:
    starts, pos = [], 0
    for tok in tokens:
        idx = text.index(tok, pos)
        starts.append(idx)
        pos = idx + len(tok)
    return starts


def _char_to_token(char_pos: int, starts: List[int]) -> int:
    """Token index whose start position is <= char_pos (token containing char_pos)."""
    best = 0
    for i, s in enumerate(starts):
        if s <= char_pos:
            best = i
        else:
            break
    return best


def _terminator_positions(tokens: List[str]) -> Set[int]:
    """Return token indices that are scope-terminating conjunctions."""
    return {i for i, t in enumerate(tokens) if SCOPE_TERMINATORS.fullmatch(t)}


def get_negated_spans(text: str) -> List[Tuple[int, int]]:
    """
    Return character spans under negation scope.
    NegEx: pre/post window.  ConText: window stops at scope terminators.
    """
    tokens = text.split()
    n = len(tokens)
    if n == 0:
        return []

    starts = _token_char_starts(text, tokens)
    terminators = _terminator_positions(tokens)
    negated: Set[int] = set()

    # Pre-negation: symptom comes AFTER cue
    for m in _PRE_NEG_RE.finditer(text):
        cue_tok = _char_to_token(m.start(), starts)
        for i in range(cue_tok + 1, min(n, cue_tok + 1 + PRE_NEGATION_WINDOW)):
            if i in terminators:
                break   # ConText: halt at conjunction
            negated.add(i)

    # Post-negation: symptom comes BEFORE cue (tight window = 3)
    for m in _POST_NEG_RE.finditer(text):
        cue_tok = _char_to_token(m.start(), starts)
        # Walk backwards from cue, stop at terminator
        for i in range(cue_tok - 1, max(-1, cue_tok - 1 - POST_NEGATION_WINDOW), -1):
            if i < 0 or i in terminators:
                break
            negated.add(i)

    return [(starts[i], starts[i] + len(tokens[i])) for i in negated if i < n]


def is_negated(sym: str, text: str, neg_spans: List[Tuple[int, int]]) -> bool:
    for m in re.finditer(re.escape(sym), text):
        for ns, ne in neg_spans:
            if m.start() < ne and m.end() > ns:
                return True
    return False


# Synonym aliases: map alternate phrasings to canonical symptom names.
# Applied as a pre-check inside fuzzy_match before fuzzy scoring.
SYMPTOM_ALIASES: Dict[str, str] = {
    "throat pain":      "sore throat",
    "throat ache":      "sore throat",
    "neck pain":        "neck pain",
    "stomach pain":     "abdominal pain",
    "stomach ache":     "abdominal pain",
    "belly pain":       "abdominal pain",
    "tummy pain":       "abdominal pain",
    "dizzy":            "dizzyness",
    "eye ache":         "eye pain",
    "ear ache":         "ear pain",
    "earache":          "ear pain",
    "leg ache":         "leg pain",
    "arm ache":         "arm pain",
    "back ache":        "back pain",
    "backache":         "back pain",
    "chest ache":       "chest pain",
    "head pain":        "headache",
    "head ache":        "headache",
    "throwing up":      "vomiting",
    "throw up":         "vomiting",
    "feel like vomiting": "nausea",
    "hard to breathe":  "difficulty breathing",
    "trouble breathing":"difficulty breathing",
    "cant breathe":     "difficulty breathing",
    "spinning":         "dizziness",
    "no energy":        "fatigue",
    "tired":            "fatigue",
    "running nose":     "runny nose",
    "watery nose":      "runny nose",
}

# =============================================================================
# STAGE 4: FUZZY MATCHING
# =============================================================================

def fuzzy_match(candidate: str) -> Optional[str]:
    if not candidate.strip():
        return None
    if candidate in SYMPTOM_SET:
        return candidate
    if candidate in SYMPTOM_ALIASES:
        return SYMPTOM_ALIASES[candidate]
    if _RAPIDFUZZ:
        res = process.extractOne(
            candidate, SYMPTOM_DICTIONARY,
            scorer=fuzz.token_sort_ratio, score_cutoff=FUZZY_THRESHOLD
        )
        return res[0] if res else None
    else:
        matches = difflib.get_close_matches(
            candidate, SYMPTOM_DICTIONARY, n=1, cutoff=FUZZY_THRESHOLD / 100
        )
        return matches[0] if matches else None


# =============================================================================
# STAGE 5: CANDIDATE EXTRACTION + OVERLAP RESOLUTION
# =============================================================================

def extract_candidates(text: str) -> List[Tuple[str, int, int]]:
    tokens = text.split()
    n = len(tokens)
    starts = _token_char_starts(text, tokens)
    candidates = []

    for size in range(MAX_NGRAM, 0, -1):
        for i in range(n - size + 1):
            window = " ".join(tokens[i: i + size])
            matched = fuzzy_match(window)
            if matched:
                char_start = starts[i]
                char_end   = starts[i + size - 1] + len(tokens[i + size - 1])
                candidates.append((matched, char_start, char_end))

    return candidates


def resolve_overlaps(candidates: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    """Keep longest non-overlapping matches (maximal munch)."""
    candidates = sorted(candidates, key=lambda c: c[2] - c[1], reverse=True)
    kept: List[Tuple[str, int, int]] = []
    used: Set[int] = set()

    for sym, s, e in candidates:
        span = set(range(s, e))
        if span & used:
            continue
        kept.append((sym, s, e))
        used |= span

    return kept


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================

def extract_symptoms(raw_text: str) -> Dict:
    """
    Extract patient-reported symptoms from free-form multilingual text.

    Parameters
    ----------
    raw_text : str
        English, Hinglish, or mixed patient input.

    Returns
    -------
    dict  {"symptoms": [sorted list of normalized English symptom strings]}
    """
    if not raw_text or not raw_text.strip():
        return {"symptoms": []}

    text = translate_hindi(raw_text)       # Stage 0 — Devanagari → English (before preprocess)
    text = preprocess(text)               # Stage 1
    text = translate_hinglish(text)       # Stage 2
    neg_spans = get_negated_spans(text)   # Stage 3
    candidates = extract_candidates(text) # Stage 4
    candidates = resolve_overlaps(candidates)

    seen: Set[str] = set()
    confirmed: List[str] = []
    for sym, s, e in candidates:          # Stage 5
        if is_negated(sym, text, neg_spans):
            continue
        if sym not in seen:
            confirmed.append(sym)
            seen.add(sym)

    confirmed.sort()
    return {"symptoms": confirmed}


def extract_symptoms_json(raw_text: str) -> str:
    """Return symptoms as a formatted JSON string."""
    return json.dumps(extract_symptoms(raw_text), ensure_ascii=False, indent=2)


# =============================================================================
# DEMO / TEST SUITE
# =============================================================================

if __name__ == "__main__":
    tests = []
    n = int(input("enter no of cases: "))
    for i in range(n):
        string_input = input("Enter string:\n")
        tests.append(("custom_input",string_input))

    sep = "=" * 70
    print(sep)
    print("  MULTILINGUAL SYMPTOM EXTRACTOR  (NegEx + ConText + Hinglish Dict)")
    print(f"  Fuzzy backend : {'RapidFuzz' if _RAPIDFUZZ else 'difflib (stdlib fallback)'}")
    print(f"  Symptom dict  : {len(SYMPTOM_DICTIONARY)} entries")
    print(f"  Hinglish map  : {len(HINGLISH_MAP)} entries")
    print(sep)

    for (label, txt) in tests:
        result = extract_symptoms(txt)
        got = result["symptoms"]

        print(f"  Input    : {txt}")
        print(f"  Got      : {got}")

    print(f"\nCompleted")
    print(sep)