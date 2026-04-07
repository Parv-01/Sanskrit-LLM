
import re, json, unicodedata, difflib
from typing import List, Dict, Set, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# A.  SYMPTOM LEXICON  (488 canonical terms, sourced from IAST dataset)
# =============================================================================
SYMPTOM_LEXICON: List[str] = [
    # Pain
    "pain","severe pain","mild pain","moderate pain","acute pain","chronic pain",
    "throbbing pain","stabbing pain","pricking pain","burning pain","radiating pain",
    "colicky pain","colicky abdominal pain","spasmodic pain","cutting pain",
    "chest pain","abdominal pain","back pain","backache","headache","migraine",
    "neck pain","joint pain","muscle pain","leg pain","arm pain","ear pain",
    "earache","eye pain","toothache","dental pain","flank pain","temporal pain",
    "groin pain","heel pain","ankle pain","foot pain","lower abdominal pain",
    "epigastric pain","pain in gums","pain in loins","pain in forehead",
    "pain in anal region","vaginal pain","penile pain","scrotal pain",
    "painful defecation","painful menstruation","painful urination",
    "painful micturition","painful erection","pain of joints","bone pain",
    "pricking pain in temples","pricking pain in chest","pricking pain in swelling",
    "severe abdominal pain","severe abdominal colicky pain","severe dental pain",
    "moderate to severe pain","pain radiating to ankle joint",
    "pain starting from hip","pain in great toe","pain in foot","tenderness",
    "tenderness of foot","pain in heel","pain in neck","sore throat","throat pain",
    "throat soreness","throat irritation",
    # Swelling / Edema
    "swelling","edema","pitting edema","non-pitting swelling","hard swelling",
    "soft swelling","painful swelling","painless swelling","reddish swelling",
    "spindle shaped swelling","swelling of foot","swelling around eyes",
    "swelling in throat","swelling in anal region","nasal swelling",
    "swelling in conjunctiva","cystic swelling","nodular swelling",
    "swelling redness in throat","swelling with yellowish discolouration",
    "inflammatory swelling","coppery-colored swelling","swelling of body parts",
    "blackish discoloration of eyelid with swelling","non-pitting swelling of legs",
    # Fever
    "fever","high fever","low grade fever","mild fever","continuous fever",
    "irregular fever","episodes of fever","fever with chills",
    "persistent low grade fever","sudden episodes of heat and cold",
    # Respiratory
    "cough","dry cough","persistent cough","cough with difficulty to expel sputum",
    "breathlessness","shortness of breath","difficulty in breathing","difficult breathing",
    "rapid breathing","deep rapid breathing","shallow breathing","noisy breathing",
    "labored breathing","wheezing","respiratory distress","chest tightness",
    "severe dyspnea","mild difficulty in breathing","occasional breathlessness",
    "breathlessness after exertion","snoring","loud breathing sounds",
    "convulsive chest movements","inability to inhale deeply",
    "foul smelling sputum","green colored sputum","yellow sputum","blood in sputum",
    "pus in sputum","sputum with green color and blood",
    "difficulty expelling sputum","hiccoughs","uncontrolled hiccoughs",
    "choking sensation in throat","abdominal enlargement with cough",
    # Gastrointestinal
    "nausea","vomiting","bilious vomiting","profuse vomiting","excessive vomiting",
    "projectile vomiting","feeling of nausea","feeling of vomiting",
    "sour eructation nausea","nausea after meals",
    "diarrhea","loose stools","constipation","difficult defecation",
    "undigested foul smelling stools","formed stools",
    "loss of appetite","anorexia","aversion to food","excessive hunger",
    "increased hunger","tastelessness","tastelessness of mouth",
    "sweet taste in mouth","bitter sensation in mouth",
    "abdominal distension","abdominal distension with gurgling","gurgling","borborygmus",
    "indigestion","difficulty to digest food","reduced digestive function",
    "heartburn","water brash","loss of digestive power",
    "blood in stool","bleeding per rectum","spitting of pus and blood",
    # Neurological / Cognitive
    "confusion","mental confusion","delirium","irrational talk","irrelevant talk",
    "difficulty concentrating","fickleness","loss of intelligence","loss of memory",
    "possible loss of memory","incoherent speech","difficulty in speech",
    "loss of speech clarity","suffocated voice","weak or broken voice",
    "loss of consciousness","temporary loss of consciousness",
    "transient loss of consciousness","fainting","frequent fainting","unconsciousness",
    "convulsions","tremors","tremor","spasms","muscle cramps",
    "dizziness","vertigo","giddiness","dizziness on standing",
    "numbness","loss of sensation","partial loss of sensation",
    "impaired tactile sensation","tingling","tingling sensation",
    "restlessness","apathy","irritability","abusive behavior",
    "insomnia","loss of sleep","drowsiness","excessive sleepiness",
    "upward gaze","downward gaze","rolling of eyes","abnormal eye movements",
    "impaired cognitive functions",
    # Motor / Musculoskeletal
    "weakness","generalized weakness","body weakness","gradual loss of strength",
    "fatigue","body fatigue","general weakness","body pain","body ache",
    "generalized body ache",
    "flaccidity","flaccidity of one side of the body","atony",
    "stiffness","stiffness of neck","stiffness in low back","stiffness in upper back",
    "stiffness in thigh","stiffness in knee","stiffness in calf muscles",
    "stiffness in feet","stiffness in gluteal region","rod like stiffness",
    "incurable stiffness of thighs","stiffness of extremities in infants",
    "stiffness of tongue","paralysis","hemiplegia","facial palsy",
    "loss of mobility on one side of the body","loss of function in lower limbs",
    "restriction of thigh movement","log-like immobility of the thighs",
    "restricted movement","reduced physical activity",
    "wasting","emaciation","muscle wasting","loss of muscle",
    "gradual loss of body weight","weight loss","weight gain",
    "deformity","lameness","limping","joint looseness",
    "inability to close eyes","inability to close lips",
    "difficulty in swallowing","difficulty in movements of eyelids",
    "labored breathing with body stretching","contraction","stretching sensation",
    "frequent throbbing sensation","frequent twitching sensation",
    "pulsation","pulsating sensation",
    "cramp like pain in upper limb","cramp like pain in lower limb",
    # Skin / Dermatological
    "itching","pruritus","itching of body","itching sensation in ear",
    "itching in throat","itching in palate","itching in lips",
    "itching in ears","itching in eyes","itching in nose","itching in head",
    "red papular eruptions with itching","reddish raised rashes with itching",
    "reddish circular lesion","skin eruptions","nasal eruptions",
    "rash","skin rash","hairloss","hair loss",
    "burning sensation","burning sensation in bladder","burning sensation in chest",
    "burning sensation in throat","burning sensation of skin",
    "burning sensation in anal region","burning sensation in parotid region",
    "burning sensation of interdigital areas","internal burning sensation",
    "severe burning sensation","burning micturition","intense burning",
    "discoloration","skin discoloration","facial discoloration","loss of complexion",
    "yellowish discoloration","blackish discoloration","reddish skin discoloration",
    "brownish-black skin discoloration","coppery skin discoloration",
    "bluish-black discoloration","coppery discoloration","pallor",
    "whitish appearance of body","jaundice","yellow discoloration of skin",
    "yellow discoloration of eyes","yellow discoloration of nails",
    "loss of body luster","thickness of skin",
    "crawling sensation in forehead","sensation of being covered with wet clothes",
    # Bleeding / Discharge
    "bleeding","bleeding from ear","bleeding from nose","bleeding from eyes",
    "bleeding from mouth","bleeding per rectum","bleeding from gums",
    "bleeding through penis","bleeding through vagina","subcutaneous bleeding",
    "bleeding from body orifices","blood discharge",
    "blackish blood discharge","reddish blood discharge","frothy blood discharge",
    "thin blood discharge","non-unctuous blood discharge",
    "ochre-colored blood discharge","black-colored blood discharge",
    "soot-colored blood discharge","collyrium-colored blood discharge",
    "viscid blood discharge","pale blood discharge","oily blood discharge",
    "slimy blood discharge","mucoid blood discharge",
    "blood discharge resembling cow urine color","vaginal discharge",
    "excessive whitish vaginal discharge","foul smelling discharge",
    "purulent discharge from ear","pus discharge","discharge of pus",
    "watery discharge","discharge from eyes","discharge from nose",
    "discharge from uterus","moist discharge from swelling","blood depletion",
    # Eyes
    "eye pain","watery eyes","redness of eyes","redness of one eye",
    "dilated eyes","bulging eyes","whitish eyes","drooping eyelid","ptosis",
    "loss of vision","loss of vision during daytime","loss of vision during night",
    "adherence of eyelids with itching","abrupt mild eye swelling",
    "swelling in conjunctiva","redness at root of eyelashes",
    "difficulty in movements of eyelids",
    "big hard painless movable swelling on eyelid",
    "swelling around eyes","itching in eyes","numbness in chest",
    # Ear / Nose / Throat
    "tinnitus","ringing in ears","hearing sounds","complete loss of hearing",
    "temporary loss of hearing","itching sensation in ear",
    "nasal blockage","nasal congestion","nasal obstruction","blocked nose",
    "nasal dryness","nasal moistness",
    "nasal pain","intermittent nasal blockage","intermittent nasal clearing",
    "sensation of nose being filled","loss of smell","watery nasal discharge",
    "runny nose","yellow nasal discharge","warm nasal discharge",
    "cold mucus discharge","pus-like nasal discharge","thick mucus discharge",
    "blackish mucus discharge","reddish mucus discharge",
    "unctuous mucus discharge","white mucus discharge",
    "dry mucus discharge","hot mucus discharge","copper-colored nasal discharge",
    "foul smell from nose","foul breath","presence of worms in nasal discharge",
    "watery discharge from nose","sneezing","excessive sneezing",
    "hoarseness","hoarseness of voice","voice change","suffocated voice",
    "dryness of throat","dryness of mouth","dryness of palate",
    "dryness of lips","dry throat","swelling in throat",
    "choking sensation in throat","itching in throat","itching in palate",
    "pricking pain in temples","crawling sensation in forehead",
    "sensation of heat rising in nose","suppuration of nose tip",
    "delayed suppuration","sensory discomfort","halitosis","cold and catarrh",
    # Urinary / Reproductive
    "urinary retention","retention of urine","frequent urination",
    "weak urinary stream","difficulty in urination","burning urination",
    "painful urination","yellow urine","burning sensation in bladder",
    "bleeding through penis","bleeding through vagina","vaginal discharge",
    "excessive whitish vaginal discharge","constant vaginal pain",
    "painful menstruation","nodular swelling in genital organs","scrotal swelling",
    # Systemic / General
    "excessive sweating","sweating","excessive thirst","increased thirst",
    "cold sensation","coldness","sensation of heat","desire for hot or cold",
    "sudden onset","recurrent episodes","intermittent occurrence",
    "sensation of heart displacement","unctuous face","glossy palms",
    "glossy soles","loss of facial wrinkling","drooling of saliva","salivation",
    "deviation of one side of face","distorted face","toothache",
    "loose teeth with pain","foul breath","loss of complexion",
    "excess fat accumulation in abdomen","excess fat accumulation in buttocks",
    "excess fat accumulation in breasts","excessive generalized fat accumulation",
    "obesity","muscle wasting","wasting of body parts","weakness of bones",
    "presence of worms","palpitations","chills","chest discomfort",
    "numbness in chest","reduced lifespan",
]

# De-duplicate
_seen: Set[str] = set()
_dd: List[str] = []
for _s in SYMPTOM_LEXICON:
    if _s not in _seen: _seen.add(_s); _dd.append(_s)
SYMPTOM_LEXICON = _dd
SYMPTOM_SET = set(SYMPTOM_LEXICON)

# =============================================================================
# B.  SYNONYM / PARAPHRASE EXPANSION TABLE
#     Catches "head ache" -> "headache", "pain in eyes" -> "eye pain", etc.
#     Applied BEFORE retrieval as a phrase-level rewrite.
# =============================================================================
SYNONYMS: Dict[str, str] = {
    # Compound rewrites (space-separated variants -> canonical)
    "head ache": "headache",
    "head pain": "headache",
    "head dard": "headache",
    "throat ache": "sore throat",
    "throat pain": "sore throat",
    "throat irritation": "sore throat",
    "throat soreness": "sore throat",
    "stomach ache": "abdominal pain",
    "stomach pain": "abdominal pain",
    "tummy ache": "abdominal pain",
    "tummy pain": "abdominal pain",
    "belly ache": "abdominal pain",
    "belly pain": "abdominal pain",
    "back ache": "back pain",
    "chest ache": "chest pain",
    "ear ache": "ear pain",
    "tooth ache": "toothache",
    "eye ache": "eye pain",
    "eyes pain": "eye pain",
    "pain in eyes": "eye pain",
    "pain in eye": "eye pain",
    "pain in ear": "ear pain",
    "pain in ears": "ear pain",
    "pain in throat": "sore throat",
    "pain in stomach": "abdominal pain",
    "pain in abdomen": "abdominal pain",
    "pain in belly": "abdominal pain",
    "pain in chest": "chest pain",
    "pain in back": "back pain",
    "pain in head": "headache",
    "pain in neck": "neck pain",
    "pain in knee": "knee pain",
    "pain in leg": "leg pain",
    "pain in arm": "arm pain",
    "pain in foot": "foot pain",
    "pain in hand": "arm pain",
    "pain in joints": "joint pain",
    "joint ache": "joint pain",
    "muscle ache": "muscle pain",
    "leg ache": "leg pain",
    "knee ache": "knee pain",
    "body ache": "body ache",
    "body pain": "body pain",
    # Nose
    "nose block": "nasal blockage",
    "nose blocked": "nasal blockage",
    "nose blockage": "nasal blockage",
    "nose bund": "nasal blockage",
    "nose band": "nasal blockage",
    "nose jam": "nasal blockage",
    "nasal block": "nasal blockage",
    "blocked nose": "nasal blockage",
    "stuffy nose": "nasal congestion",
    "runny nose": "runny nose",
    "nose running": "runny nose",
    "nose water": "watery nasal discharge",
    "nose watery": "watery nasal discharge",
    "nose closure": "nasal blockage",
    "nose closed": "nasal blockage",
    "nasal congestion": "nasal congestion",
    # Eye
    "eye pain": "eye pain",
    "eye ache": "eye pain",
    "eyes hurt": "eye pain",
    "eye swelling": "swelling around eyes",
    "eye water": "watery eyes",
    "eye watery": "watery eyes",
    "eye redness": "redness of eyes",
    "eye red": "redness of eyes",
    "red eye": "redness of eyes",
    "eye discharge": "discharge from eyes",
    "eye itching": "itching in eyes",
    # Ear
    "ear pain": "ear pain",
    "ear ache": "ear pain",
    "ears ringing": "tinnitus",
    "ear ringing": "tinnitus",
    "ear discharge": "purulent discharge from ear",
    "ear itching": "itching in ears",
    "ear itching sensation": "itching sensation in ear",
    # Skin
    "skin itch": "itching",
    "skin itching": "itching",
    "skin rash": "rash",
    "skin burn": "burning sensation of skin",
    "skin burning": "burning sensation of skin",
    # GI
    "loose motion": "diarrhea",
    "loose motions": "diarrhea",
    "watery stool": "diarrhea",
    "watery stools": "diarrhea",
    # Breathing
    "breathing problem": "difficulty in breathing",
    "breathing difficulty": "difficulty in breathing",
    "breathing trouble": "difficulty in breathing",
    "difficulty breathing": "difficulty in breathing",
    "cant breathe": "difficulty in breathing",
    "short of breath": "shortness of breath",
    # Misc
    "high temp": "fever",
    "high temperature": "fever",
    "running nose": "runny nose",
    "heavy head": "heaviness of head",
    "light headed": "dizziness",
    "lightheaded": "dizziness",
    "blurred vision": "loss of vision",
    "blurry vision": "loss of vision",
    "double vision": "loss of vision",
    "ringing ears": "tinnitus",
    "buzzing ears": "tinnitus",
}

# =============================================================================
# C.  HINGLISH / HINDI → ENGLISH MAP  (200+ entries)
# =============================================================================
HINGLISH_MAP: Dict[str, str] = {
    # Fever
    "bukhar": "fever", "bukhaar": "fever", "tez bukhar": "high fever",
    "halka bukhar": "mild fever", "jwara": "fever", "jvara": "fever",
    # Head
    "sir dard": "headache", "sar dard": "headache",
    "sir me dard": "headache", "sar me dard": "headache",
    "sir darda": "headache", "sar bhaari": "headache",
    "sir bhaari": "headache", "sir bharipan": "headache",
    # Pain general
    "dard": "pain", "peeda": "pain", "takleef": "pain",
    "vedana": "pain", "shool": "pain", "sula": "pain",
    # Body parts + pain
    "pet dard": "abdominal pain", "pet me dard": "abdominal pain",
    "pait dard": "abdominal pain",
    "seene me dard": "chest pain", "seene ka dard": "chest pain",
    "sine me dard": "chest pain",
    "kamar dard": "back pain", "kamar me dard": "back pain",
    "peeth dard": "back pain", "peeth me dard": "back pain",
    "gale me dard": "sore throat", "gale me kharaash": "sore throat",
    "gale me jalan": "sore throat", "gala kharaab": "sore throat",
    "gale me sujan": "swelling in throat",
    "jodon me dard": "joint pain", "jod dard": "joint pain",
    "muscles me dard": "muscle pain", "haddi dard": "bone pain",
    "badan dard": "body ache", "badan me dard": "body ache",
    "body me dard": "body ache", "poora badan dard": "body ache",
    "pair dard": "leg pain", "haath dard": "arm pain",
    "kaan me dard": "ear pain", "kaan me takleef": "ear pain",
    "kaan dard": "ear pain",
    "aankhon me dard": "eye pain", "aankh me dard": "eye pain",
    "aankh dard": "eye pain",
    "dant dard": "toothache", "daant dard": "toothache",
    "ghutne me dard": "knee pain", "ghutna dard": "knee pain",
    "sir": "head",  # partial — only used in multi-word context
    # Vomiting / Nausea
    "ulti": "vomiting", "ulti hona": "vomiting",
    "ulti ho rahi": "vomiting", "ulti aa rahi": "vomiting",
    "matli": "nausea", "ji machalna": "nausea",
    "ji ghabhrana": "nausea", "chardi": "vomiting", "ubkaayi": "nausea",
    # GI
    "dast": "diarrhea", "loose motion": "diarrhea", "patla mal": "diarrhea",
    "qabz": "constipation", "kabz": "constipation",
    "pet me gas": "bloating", "gas banna": "bloating",
    "acidity": "acidity",
    "bhook nahi": "loss of appetite", "bhooke nahi": "loss of appetite",
    "aruchi": "loss of appetite", "khana nahi khana": "loss of appetite",
    # Respiratory
    "khansi": "cough", "khaansi": "cough",
    "sukhhi khansi": "dry cough", "sukhi khansi": "dry cough",
    "balgam wali khansi": "productive cough",
    "sans phoolna": "shortness of breath",
    "sans lene me takleef": "difficulty in breathing",
    "sans lene me dikkat": "difficulty in breathing",
    "naak bund": "nasal blockage", "naak band": "nasal blockage",
    "naak jam gaya": "nasal blockage", "naak jam": "nasal blockage",
    "naak se paani": "runny nose", "naak se pani": "runny nose",
    "naak beh rahi": "runny nose",
    "chheenk": "sneezing", "chheenke": "sneezing", "chheenk aana": "sneezing",
    "zukam": "cold and catarrh", "jukam": "cold and catarrh",
    "awaaz band": "hoarseness", "bhaari awaaz": "hoarseness",
    "awaaz baithi": "hoarseness", "gale mein kharaash": "sore throat",
    # Neurological
    "chakkar": "dizziness", "chakkara": "dizziness",
    "chakkar aana": "dizziness", "sar ghoomna": "dizziness",
    "behoshi": "fainting", "murcha": "fainting",
    "behosh ho jaana": "loss of consciousness",
    "hosh kho dena": "loss of consciousness",
    "jhunzhuni": "tingling", "jhunjhuni": "tingling",
    "sunn hona": "numbness", "sun ho jaana": "numbness",
    "neend nahi": "insomnia", "neend na aana": "insomnia",
    "bahut neend": "excessive sleepiness", "nindasi": "drowsiness",
    "yaaddast kamzor": "loss of memory", "bhram": "confusion",
    "kaanpna": "tremors", "kaanpna hath": "tremors",
    # Weakness / Fatigue
    "kamzori": "weakness", "kamzoori": "weakness",
    "thakaan": "fatigue", "thakan": "fatigue",
    "thakaawat": "fatigue", "alasya": "fatigue",
    # Skin
    "khujli": "itching", "khujali": "itching",
    "daane": "rash", "chaane": "rash", "phode": "rash",
    "jalan": "burning sensation", "jalan hona": "burning sensation",
    "sujan": "swelling", "sozish": "swelling", "phulana": "swelling",
    # Eyes
    "aankhon se paani": "watery eyes", "aankh se paani": "watery eyes",
    "aankhon me lali": "redness of eyes", "aankh laal hona": "redness of eyes",
    "aankh laal": "redness of eyes",
    "dhundhla dikhna": "blurred vision", "nazar kamzor": "loss of vision",
    "aankhon me sujan": "swelling around eyes",
    "aankhon me khujli": "itching in eyes",
    # ENT
    "kaan me ghanti": "tinnitus", "kaan bajana": "tinnitus",
    "kaan me awaaz": "tinnitus",
    "naak se khoon": "bleeding from nose", "naak se blood": "bleeding from nose",
    "naak se badbu": "foul smell from nose",
    "muh se badbu": "foul breath", "muh sukha": "dryness of mouth",
    "muh ka sukha": "dryness of mouth",
    # Systemic
    "paseena": "sweating", "bahut paseena": "excessive sweating",
    "thithurana": "chills", "sardi lagna": "chills", "kaanpna": "chills",
    "pyas jyada": "excessive thirst", "jyada pyas": "excessive thirst",
    "baar baar peshab": "frequent urination",
    "peshab me jalan": "burning urination",
    "dil ka tez dhakna": "palpitations", "dil dhadakna": "palpitations",
    "wazn kamm": "weight loss", "vajan kam hona": "weight loss",
    "bhaar badhna": "weight gain", "vajan badhna": "weight gain",
    # Devanagari
    "बुखार": "fever", "सिर दर्द": "headache",
    "पेट दर्द": "abdominal pain", "खांसी": "cough",
    "उल्टी": "vomiting", "मतली": "nausea",
    "दस्त": "diarrhea", "कब्ज": "constipation",
    "चक्कर": "dizziness", "कमजोरी": "weakness",
    "थकान": "fatigue", "खुजली": "itching",
    "सूजन": "swelling", "जलन": "burning sensation",
    "दर्द": "pain", "बेहोशी": "fainting",
    "नींद नहीं": "insomnia", "भूख नहीं": "loss of appetite",
    "छींक": "sneezing", "नाक बंद": "nasal blockage",
    "सांस लेने में तकलीफ": "difficulty in breathing",
    "छाती में दर्द": "chest pain", "पीठ दर्द": "back pain",
    "जोड़ों में दर्द": "joint pain", "शरीर में दर्द": "body ache",
    "आंखों में दर्द": "eye pain", "कान में दर्द": "ear pain",
    "गले में दर्द": "sore throat", "दांत दर्द": "toothache",
    "नाक से पानी": "runny nose", "पसीना": "sweating",
    "कंपकंपी": "tremors", "सुन्नपन": "numbness", "झुनझुनी": "tingling",
    "याददाश्त कमजोर": "loss of memory", "भ्रम": "confusion",
    "मुंह सूखा": "dryness of mouth", "प्यास ज्यादा": "excessive thirst",
    "बार बार पेशाब": "frequent urination",
    "नाक से खून": "bleeding from nose", "आँखों से पानी": "watery eyes",
    "गला खराब": "sore throat", "गले में खराश": "sore throat",
}

# =============================================================================
# D.  PREPROCESSING
# =============================================================================
def _build_h_re():
    keys = sorted(HINGLISH_MAP, key=len, reverse=True)
    return re.compile(
        r"(?<![a-zA-Z\u0900-\u097F])(?:" +
        "|".join(re.escape(k) for k in keys) +
        r")(?![a-zA-Z\u0900-\u097F])"
    )

def _build_syn_re():
    keys = sorted(SYNONYMS, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b")

_H_RE   = _build_h_re()
_SYN_RE = _build_syn_re()


def _translate_hinglish(text: str) -> str:
    return _H_RE.sub(lambda m: HINGLISH_MAP[m.group(0)], text)

def _expand_synonyms(text: str) -> str:
    """Apply phrase-level synonym expansion (longest match first)."""
    return _SYN_RE.sub(lambda m: SYNONYMS[m.group(0)], text)

def preprocess(raw: str) -> str:
    t = unicodedata.normalize("NFC", raw)
    t = re.sub(r"[\u200b-\u200d\ufeff\u00ad]", "", t)
    t = t.lower().strip()
    t = _translate_hinglish(t)
    # Strip residual Devanagari after translation
    t = re.sub(r"[\u0900-\u097f]+", " ", t)
    t = re.sub(r"[^\w\s\-',]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # Synonym expansion AFTER cleaning
    t = _expand_synonyms(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# =============================================================================
# E.  TYPO CORRECTION  (word-level difflib)
# =============================================================================
_STOPWORDS: Set[str] = {
    "the","a","an","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","shall","should",
    "may","might","must","can","could","of","in","on","at","to","for",
    "with","by","from","up","about","into","through","during","before",
    "after","above","below","between","each","all","both","few","more",
    "most","other","some","such","no","nor","not","only","own","same",
    "than","too","very","just","this","that","these","those","and","or",
    "but","if","while","as","until","although","because","since","though",
    "hai","hain","mujhe","mere","meri","mera","aur","bhi","lekin",
    "magar","par","ya","toh","se","ka","ki","ke","jo","jab","tab",
    "kya","nahi","ho","raha","rahi","aa","hona","gaya","gayi",
    "i","my","me","you","your","he","she","it","we","they","patient",
    "have","has","since","feel","feeling","having","got","getting",
}

_SYM_WORDS: List[str] = sorted(set(
    w for s in SYMPTOM_LEXICON for w in s.split() if len(w) > 3 and w not in _STOPWORDS
))

def _correct_word(word: str) -> str:
    if len(word) < 4 or word in _STOPWORDS: return word
    if any(word in s for s in SYMPTOM_LEXICON): return word
    close = difflib.get_close_matches(word, _SYM_WORDS, n=1, cutoff=0.80)
    return close[0] if close else word

def typo_correct(text: str) -> str:
    return " ".join(_correct_word(w) for w in text.split())

# =============================================================================
# F.  SEMANTIC INDEX  (TF-IDF dual encoder, DPR-style retrieval)
# =============================================================================
SIM_THRESHOLD_MULTI  = 0.32   # for 2+ token windows
SIM_THRESHOLD_SINGLE = 0.92   # single tokens must near-exactly match
TOP_K = 5
MAX_NGRAM_WIN = 5

def _paraphrases() -> List[str]:
    return [
        "ache soreness tenderness discomfort hurts sore painful",
        "swollen puffy inflamed oedema fluid retention",
        "coughing phlegm mucus sputum expectorating",
        "haemorrhage blood loss discharge bleed",
        "dyspnea respiratory difficulty inhale exhale breath",
        "nausea queasy sick stomach upset",
        "dizzy lightheaded spinning giddy vertigo",
        "weak tired exhausted fatigue lethargy sluggish",
        "fever temperature pyrexia febrile hot",
        "paralysis palsy immobile unable move loss mobility",
        "stiff rigid tight restricted range motion",
        "itching pruritus scratching urge itch",
        "burning scalding heat sensation fire",
        "numb anaesthetic loss sensation hypoesthesia tingling",
        "confused disorientated clouded delirium irrational",
        "anorexia appetite loss not hungry",
        "vomit emesis throwing up retching bilious",
        "constipate hard stool difficulty passing bowel",
        "diarrhoea loose stool frequent bowel loose",
        "oedema fluid retention puffiness puffy",
        "flaccid lax loose muscle tone low",
        "atrophy wasting shrinking muscle loss",
        "jaundice yellow icterus eyes skin nails",
        "pale pallor white anaemia bloodless",
        "hoarse voice change dysphonia raspy broken",
        "insomnia sleeplessness difficulty sleeping awake",
        "thirst polydipsia excessive water intake",
        "palpitation heart racing fast beat",
        "tremor shaking shivering quivering",
        "rash eruption hives urticaria skin lesion",
        "discharge pus oozing secretion fluid leaking",
        "nasal runny blocked stuffy congested sneezing",
        "earache tinnitus ringing buzzing ear",
        "throat sore painful swallowing difficulty",
        "blurred vision vision loss sight impaired",
        "muscle cramps spasms convulsions seizure",
        "swallowing difficulty dysphagia choke",
        "hair loss alopecia bald hairloss",
        "weight loss wasting emaciation thin",
        "excessive sweating diaphoresis perspiration",
        "chills rigors shivering cold",
        "headache head ache pain head",
        "eye pain pain in eye eyes hurt",
        "nose block nasal congestion stuffy nose blocked nose nose bund",
        "sore throat throat pain throat ache",
        "stomach ache tummy ache belly pain abdominal pain",
        "ear ache earache ear pain",
    ]

class SymptomIndex:
    def __init__(self):
        corpus = SYMPTOM_LEXICON + _paraphrases()
        self.wv = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 3), sublinear_tf=True
        )
        self.cv = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
        )
        w_all = self.wv.fit_transform(corpus)
        c_all = self.cv.fit_transform(corpus)
        n = len(SYMPTOM_LEXICON)
        self.idx_w = w_all[:n]
        self.idx_c = c_all[:n]

    def retrieve(self, query: str, top_k: int = TOP_K,
                 threshold: float = SIM_THRESHOLD_MULTI) -> List[Tuple[str, float]]:
        qw = self.wv.transform([query])
        qc = self.cv.transform([query])
        sim = (0.65 * cosine_similarity(qw, self.idx_w).flatten() +
               0.35 * cosine_similarity(qc, self.idx_c).flatten())
        idxs = np.argsort(sim)[::-1][:top_k]
        return [(SYMPTOM_LEXICON[i], float(sim[i]))
                for i in idxs if sim[i] >= threshold]

# =============================================================================
# G.  NEGATION  (NegEx + ConText)
# =============================================================================
_PRE_NEG = re.compile(
    r"\b(?:no|not|nahi|naheen|nahi\s+hai|na(?!\w)|without|absence\s+of|"
    r"free\s+of|negative\s+for|never|denies|denied|koi\s+nahi|koi)\b", re.I
)
_POST_NEG = re.compile(
    r"\b(?:nahi\s+hai|nahi|naheen|not\s+present|was\s+not|is\s+not|absent)\b",
    re.I
)
_SCOPE_BREAK_PRE = re.compile(
    r"\b(?:but|however|except|although|yet|still|aur|and|also|lekin|magar)\b",
    re.I
)
_SCOPE_BREAK_POST = re.compile(
    r"\b(?:however|except|although|yet|still|aur|and|also|bhi|hai)\b", re.I
)
_NEG_WIN_PRE  = 6
_NEG_WIN_POST = 3


def _tok_pos(text: str, tokens: List[str]) -> List[int]:
    starts, pos = [], 0
    for tok in tokens:
        i = text.index(tok, pos); starts.append(i); pos = i + len(tok)
    return starts

def _c2t(cp: int, starts: List[int]) -> int:
    best = 0
    for i, s in enumerate(starts):
        if s <= cp: best = i
        else: break
    return best

def negated_spans(text: str) -> List[Tuple[int, int]]:
    tokens = text.split()
    if not tokens: return []
    starts = _tok_pos(text, tokens)
    n = len(tokens)
    pre_breaks  = {i for i, t in enumerate(tokens) if _SCOPE_BREAK_PRE.fullmatch(t)}
    post_breaks = {i for i, t in enumerate(tokens) if _SCOPE_BREAK_POST.fullmatch(t)}
    neg: Set[int] = set()

    for m in _PRE_NEG.finditer(text):
        ct = _c2t(m.start(), starts)
        for i in range(ct + 1, min(n, ct + 1 + _NEG_WIN_PRE)):
            if i in pre_breaks: break
            neg.add(i)

    for m in _POST_NEG.finditer(text):
        ct = _c2t(m.start(), starts)
        for i in range(max(0, ct - _NEG_WIN_POST), ct):
            if i in post_breaks: break
            neg.add(i)

    return [(starts[i], starts[i] + len(tokens[i])) for i in neg if i < n]

def _is_neg(phrase: str, text: str, spans: List[Tuple[int, int]]) -> bool:
    for m in re.finditer(re.escape(phrase), text):
        for ns, ne in spans:
            if m.start() < ne and m.end() > ns: return True
    return False

# =============================================================================
# H.  CANDIDATE EXTRACTION
#     Key change from v2: single-token windows use a much higher threshold
#     so stray words like "head" or "ache" don't fire independently.
# =============================================================================
def _is_all_stopwords(tokens: List[str]) -> bool:
    return all(t in _STOPWORDS for t in tokens)

def _get_candidates(text: str, index: SymptomIndex) -> List[Tuple[str, float, int, int]]:
    tokens = text.split()
    n = len(tokens)
    if n == 0: return []
    starts = _tok_pos(text, tokens)
    found = []

    for size in range(MAX_NGRAM_WIN, 0, -1):
        for i in range(n - size + 1):
            window_toks = tokens[i:i + size]
            if _is_all_stopwords(window_toks): continue
            window = " ".join(window_toks)
            if len(window.strip()) < 3: continue

            # Single-token windows: use near-exact threshold to prevent noise
            thresh = SIM_THRESHOLD_SINGLE if size == 1 else SIM_THRESHOLD_MULTI

            matches = index.retrieve(window, top_k=1, threshold=thresh)
            if matches:
                sym, score = matches[0]
                cs = starts[i]
                ce = starts[i + size - 1] + len(tokens[i + size - 1])
                found.append((sym, score, cs, ce))
    return found

def _resolve(found: List[Tuple[str, float, int, int]]) -> List[Tuple[str, float, int, int]]:
    found = sorted(found, key=lambda x: x[1], reverse=True)
    kept, used = [], set()
    for sym, score, cs, ce in found:
        span = set(range(cs, ce))
        if span & used: continue
        kept.append((sym, score, cs, ce))
        used |= span
    return kept

# =============================================================================
# I.  MAIN PIPELINE
# =============================================================================
_INDEX: Optional[SymptomIndex] = None

def _get_index() -> SymptomIndex:
    global _INDEX
    if _INDEX is None: _INDEX = SymptomIndex()
    return _INDEX


def extract(raw_text: str) -> Dict:
    """
    Extract symptoms from English / Hindi / Hinglish free text.
    Returns {"symptoms": [...], "count": N}
    """
    if not raw_text or not raw_text.strip():
        return {"symptoms": [], "count": 0}

    index = _get_index()
    text  = preprocess(raw_text)       # Hinglish map + synonym expand + clean
    text  = typo_correct(text)          # word-level fuzzy correction
    text  = _expand_synonyms(text)      # re-run synonyms AFTER typo fix
    text  = re.sub(r"\s+", " ", text).strip()
    spans = negated_spans(text)         # NegEx negation spans
    found = _get_candidates(text, index)
    found = _resolve(found)             # maximal munch overlap removal

    seen: Set[str] = set()
    out:  List[str] = []
    for sym, score, cs, ce in sorted(found, key=lambda x: x[2]):
        if _is_neg(sym, text, spans): continue
        if sym not in seen:
            out.append(sym)
            seen.add(sym)

    return {"symptoms": out, "count": len(out)}


def extract_json(raw_text: str) -> str:
    return json.dumps(extract(raw_text), ensure_ascii=False, indent=2)


# =============================================================================
# J.  DEMO
# =============================================================================
if __name__ == "__main__":
    import time

    TESTS = [
        # Core v2 tests
        ("Hindi basic",         "Mujhe bukhar aur sir dard hai"),
        ("English + negation",  "I have chest pain but no fever"),
        ("Hinglish GI",         "pet me dard aur ulti ho rahi hai"),
        ("English typos",       "I have fevr and hedache since yesterday"),
        ("Negated + positive",  "mujhe koi bukhar nahi hai but weakness hai"),
        ("Complex neg Hing.",   "aankhon me dard hai aur naak se paani aa raha hai khansi bhi hai lekin bukhar nahi"),
        ("Devanagari",          "मुझे बुखार है और सिर दर्द भी है उल्टी भी हो रही है"),
        ("Mixed script neg",    "Mujhe बुखार है aur chest pain bhi hai ulti nahi hai"),
        # User-reported failures
        ("pain in eyes",        "pain in eyes"),
        ("eyes pain",           "eyes pain"),
        ("head ache",           "head ache"),
        ("head ache + throat",  "i have head ache and pain in throat"),
        ("throatt ache",        "throatt ache"),
        ("nose band",           "nose band"),
        ("nose bund",           "nose bund"),
        ("nose closure",        "nose closure"),
        ("eyespain (typo)",     "eyespain"),
        # New edge cases
        ("hand pain",           "arm pain"),
        ("shiverings",          "shivering"),
        ("stomach ache",        "stomach ache"),
        ("back ache",           "back ache"),
        ("ear ache",            "ear ache"),
        ("running nose",        "I have running nose and sore throat"),
        ("Devanagari full",     "मुझे खांसी है, थकान है, सांस लेने में तकलीफ है और पेट दर्द भी है"),
    ]

    sep = "=" * 72
    print(sep)
    print("  MULTILINGUAL SYMPTOM EXTRACTION PIPELINE  v3.0")
    print(f"  Lexicon  : {len(SYMPTOM_LEXICON)} canonical symptoms")
    print(f"  Hinglish : {len(HINGLISH_MAP)} translation mappings")
    print(f"  Synonyms : {len(SYNONYMS)} phrase rewrites")
    print("  Building index...", end=" ", flush=True)
    t0 = time.time()
    _get_index()
    print(f"done ({time.time()-t0:.2f}s)")
    print(sep)

    for label, txt in TESTS:
        t0 = time.time()
        result = extract(txt)
        ms = (time.time() - t0) * 1000
        print(f"\n[{label}]  ({ms:.0f} ms)")
        print(f"  Input   : {txt}")
        print(f"  Symptoms: {result['symptoms']}")

    print(f"\n{sep}")
    print("  Interactive — type text, press Enter  (Ctrl+C to quit)")
    print(sep)
    try:
        while True:
            user = input("\n> ").strip()
            if user:
                print(extract_json(user))
    except (KeyboardInterrupt, EOFError):
        print("\nCompleted.")