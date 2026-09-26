"""
preprocess.py — Text normalization for business entity resolution.
"""

import re
from unidecode import unidecode

# ── Legal suffix normalization map ────────────────────────────────────────────
LEGAL_SUFFIXES = [
    (r'\bpvt\b\.?', 'private'),
    (r'\bltd\b\.?', 'limited'),
    (r'\bcorp\b\.?', 'corporation'),
    (r'\binc\b\.?', 'incorporated'),
    (r'\bco\b\.?',  'company'),
    (r'\bassoc\b\.?', 'association'),
    (r'\bintl\b\.?', 'international'),
    (r'\bmfg\b\.?', 'manufacturing'),
    (r'\bsvcs?\b\.?', 'services'),
    (r'\bgrp\b\.?', 'group'),
    (r'\benterp?\b\.?', 'enterprise'),
]

# ── Address abbreviation map ──────────────────────────────────────────────────
ADDRESS_ABBREVS = [
    (r'\brd\b\.?',   'road'),
    (r'\bst\b\.?',   'street'),
    (r'\bave\b\.?',  'avenue'),
    (r'\bblvd\b\.?', 'boulevard'),
    (r'\bdr\b\.?',   'drive'),
    (r'\bct\b\.?',   'court'),
    (r'\bln\b\.?',   'lane'),
    (r'\bpkwy\b\.?', 'parkway'),
    (r'\bhwy\b\.?',  'highway'),
    (r'\bapt\b\.?',  'apartment'),
    (r'\bste\b\.?',  'suite'),
    (r'\bfl\b\.?',   'floor'),
    (r'\bmt\b\.?',   'mount'),
    (r'\bnr\b\.?',   'near'),
    (r'\bopp\b\.?',  'opposite'),
    (r'\bph\b\.?',   'phase'),
    (r'\bdist\b\.?', 'district'),
    (r'\bno\b\.?',   'number'),
]

# ── US state abbreviations ────────────────────────────────────────────────────
US_STATES = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas',
    'ca': 'california', 'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware',
    'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii', 'id': 'idaho',
    'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas',
    'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
    'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
    'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york',
    'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma',
    'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
    'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah',
    'vt': 'vermont', 'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia',
    'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia',
}

# ── Indian state abbreviations ────────────────────────────────────────────────
INDIAN_STATES = {
    'up': 'uttar pradesh', 'mp': 'madhya pradesh', 'ap': 'andhra pradesh',
    'tn': 'tamil nadu', 'wb': 'west bengal', 'mh': 'maharashtra',
    'rj': 'rajasthan', 'ka': 'karnataka', 'gj': 'gujarat', 'hr': 'haryana',
    'pb': 'punjab', 'jk': 'jammu and kashmir', 'uk': 'uttarakhand',
    'hp': 'himachal pradesh', 'ts': 'telangana', 'cg': 'chhattisgarh',
    'jh': 'jharkhand', 'dl': 'delhi', 'or': 'odisha',
}


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, strip, collapse whitespace."""
    if not isinstance(text, str):
        return ''
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def transliterate(text: str) -> str:
    """Convert non-ASCII characters to closest ASCII equivalents."""
    if not isinstance(text, str):
        return ''
    return unidecode(text)


def normalize_name(name: str, do_transliterate: bool = True) -> str:
    """
    Normalize a business name:
      1. Lowercase + strip
      2. Transliterate non-ASCII
      3. Remove punctuation (keep alphanumeric + spaces)
      4. Expand legal suffixes
      5. Remove common noise tokens
      6. Collapse whitespace
    """
    if not isinstance(name, str) or not name.strip():
        return ''

    text = clean_text(name)

    if do_transliterate:
        text = transliterate(text)

    # Remove URLs / domains
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\b\w+\.(com|org|net|co|io)\b', '', text)

    # Replace & with 'and'
    text = re.sub(r'&', ' and ', text)

    # Expand legal suffixes
    for pattern, replacement in LEGAL_SUFFIXES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Remove punctuation but keep alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Remove common noise tokens
    noise = {'llc', 'llp', 'dba', 'the', 'of', 'and', 'a', 'an'}
    tokens = text.split()
    tokens = [t for t in tokens if t not in noise]

    text = ' '.join(tokens)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_address(address: str, do_transliterate: bool = True) -> str:
    """
    Normalize a business address:
      1. Lowercase + strip
      2. Transliterate non-ASCII
      3. Expand abbreviations
      4. Remove punctuation
      5. Sort tokens for order-invariant matching
    """
    if not isinstance(address, str) or not address.strip():
        return ''

    text = clean_text(address)

    if do_transliterate:
        text = transliterate(text)

    # Replace & with 'and'
    text = re.sub(r'&', ' and ', text)

    # Expand address abbreviations
    for pattern, replacement in ADDRESS_ABBREVS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Remove 'null' strings
    text = re.sub(r'\bnull\b', '', text)

    # Remove punctuation but keep alphanumeric, spaces, and hyphens
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_address_numbers(address: str) -> str:
    """Extract all numeric tokens from an address (street numbers, PIN codes)."""
    if not isinstance(address, str):
        return ''
    return ' '.join(re.findall(r'\b\d+\b', address))


def create_blocking_key(name_norm: str, country: str) -> str:
    """Create a simple blocking key: first 4 chars of normalized name + country."""
    prefix = name_norm[:4] if len(name_norm) >= 4 else name_norm
    return f"{prefix}_{country.lower()}" if prefix else f"__{country.lower()}"


def preprocess_dataframe(df, source_name=''):
    """
    Apply all preprocessing to a source DataFrame.
    Adds normalized columns in-place.
    """
    import time
    t0 = time.time()
    print(f"  Preprocessing {source_name} ({len(df):,} rows)...")

    # Transliterate + normalize name
    df['name_norm'] = df['business_name'].fillna('').apply(normalize_name)

    # Transliterate + normalize address
    df['addr_norm'] = df['business_address'].fillna('').apply(normalize_address)

    # Combined text for TF-IDF blocking
    df['combined_text'] = df['name_norm'] + ' ' + df['addr_norm']

    # Address numbers
    df['addr_numbers'] = df['business_address'].fillna('').apply(extract_address_numbers)

    # Blocking key
    df['block_key'] = df.apply(
        lambda r: create_blocking_key(r['name_norm'], r['country']), axis=1)

    elapsed = time.time() - t0
    print(f"    ✓ Done in {elapsed:.1f}s")
    return df
