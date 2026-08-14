"""Feature extraction for the Ecoself NLP instrument.

Scores every text in ``corpus/`` on a grammatical predictor and an independent
lexical criterion, plus a set of descriptive text-level features, then writes
the result to ``corpus_scores.csv``.

Predictor
---------
ExperienceAttribution
    Rate per 1,000 words at which a nature noun occupies the subject position
    of an experiential verb. The study's single grammatical measure of
    experience attribution to nature.

Criterion
---------
SelfTranscendence
    Ji & Raney's Self-Transcendent Emotion Dictionary, restricted to the awe
    and gratitude sub-categories, matched by stem prefix and reported per
    1,000 words. All six sub-categories (awe, gratitude, admiration, elevation,
    hope, general inspiration) are written as separate columns with the same
    curation, and the unmodified full dictionary as a further column; these are
    descriptive comparisons only and never used as the criterion.

Descriptive features (never entering a hypothesis)
--------------------------------------------------
AgencyAttribution
    Rate per 1,000 words at which a nature noun occupies the subject position
    of an action verb. The doer counterpart to ExperienceAttribution: both
    attribution measures read the first WordNet sense of the noun and of the
    verb, their supersense sets are disjoint, and so no clause is counted by
    both.
ActiveVoice_ratio
    One minus the PassivePy passive-sentence proportion, computed over every
    sentence in the text with no reference to what the sentence is about. A
    text-wide marker of grammatical voice and register, not a measure of
    nature's agency: "the report was commissioned" counts against it exactly
    as much as "the forest was cleared". Formerly named EcoAgency; the values
    are unchanged by the rename.
Verb_noun_ratio
    VERB count over NOUN count. High values indicate process-heavy prose.
Sensorimotor, Concreteness, EpistemicOpenness
    Norm- and lexicon-based extras, with coverage diagnostics for the two
    norm-based measures.

Run from the project root::

    python pipeline/extract_features.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import nltk
import pandas as pd
import spacy
from nltk.corpus import wordnet as wn
from PassivePySrc.PassivePy import PassivePyAnalyzer

# Windows defaults stdout to cp1252, which cannot print the Yoruba diacritics in
# the corpus metadata.
sys.stdout.reconfigure(encoding="utf-8")

# --------------------------------------------------------------------------- #
# Paths and configuration
# --------------------------------------------------------------------------- #

CORPUS_DIR = Path("corpus")
METADATA_PATH = Path("corpus_metadata.csv")
LEXICON_DIR = Path("pipeline/lexicons")
OUTPUT_PATH = Path("corpus_scores.csv")

SPACY_MODEL = "en_core_web_lg"

# Audit thresholds. Chunks were cut to 900-1,000 words, so anything outside
# this wider band is worth inspecting; below this coverage floor a norm-based
# extra is not reliable for that text.
WORD_COUNT_MIN = 800
WORD_COUNT_MAX = 1100
COVERAGE_FLOOR = 0.65
STEM_DOMINANCE_THRESHOLD = 0.30

# --------------------------------------------------------------------------- #
# Criterion configuration (STED)
# --------------------------------------------------------------------------- #

# Ji & Raney's six sub-categories, tagged in the source JSON. The criterion uses
# awe and gratitude; the other four are scored for descriptive comparison only
# and never enter a hypothesis. Ordered criterion pair first.
STED_CATEGORIES = [
    "awe",
    "gratitude",
    "admiration",
    "elevation",
    "hope",
    "general_inspiration",
]

# The dictionary was built on US English and misses these forms.
STED_UK_ADDITIONS = {"honour", "honourable", "honoured"}

# Stems whose dominant use across this corpus is policy or academic register
# rather than self-transcendent emotion.
STED_STEM_EXCLUSIONS = {"engag", "acknowledg", "aesthetic", "recogn"}

# Surface forms that match a retained stem in a non-emotional sense:
# "kind(s)" as a category term, "help*" as the policy-register auxiliary, and
# grand-kin terms matching the kinship stem.
STED_SURFACE_STOPLIST = {
    "kind",
    "kinds",
    "help",
    "helps",
    "helped",
    "helping",
    "grandfather",
    "grandmother",
    "grandparent",
    "grandparents",
    "grandchild",
    "grandchildren",
    "granddaughter",
    "grandson",
}

# --------------------------------------------------------------------------- #
# Attribution configuration
#
# The nature-noun test below is shared by ExperienceAttribution and
# AgencyAttribution. The two measures differ only in the verb test.
# --------------------------------------------------------------------------- #

# A noun counts as a nature noun when its first WordNet sense falls in one of
# these supersenses.
NATURE_SUPERSENSES = {
    "noun.animal",
    "noun.plant",
    "noun.phenomenon",
    "noun.object",
}

# Nouns whose first WordNet sense misclassifies them for this purpose.
NATURE_SUPPLEMENT = {
    "forest",
    "branch",
    "soil",
    "lichen",
    "fire",
    "flame",
    "wave",
    "tide",
    "ice",
    "snowfall",
    "sunrise",
    "sunset",
    "plant",
    "animal",
    "creature",
    "organism",
    "wildlife",
    "fauna",
    "being",
    "species",
    "specie",
    "beast",
    "nature",
    "life",
    "rock",
    "stone",
    "cliff",
    "peak",
    "more-than-human",
    "place",
    "whale",
    "urchin",
    "water",
    "air",
    "nectar",
}

# Blocked regardless of what WordNet returns, since attributing experience to a
# human is not the construct being measured.
HUMAN_NOUNS = {
    "human",
    "person",
    "people",
    "man",
    "woman",
    "child",
    "individual",
    "mum",
    "mom",
    "mother",
    "dad",
    "father",
}

# A verb counts as experiential when its first sense falls in one of these
# supersenses.
EXPERIENTIAL_SUPERSENSES = {
    "verb.cognition",
    "verb.perception",
    "verb.emotion",
    "verb.communication",
}

# Verbs whose first sense is a communication or perception sense but whose use
# is expository metadiscourse rather than attributed inner life: "the report
# suggests", "the data show", "the model determines".
EXPERIENTIAL_VERB_STOPLIST = {
    "face",
    "show",
    "determine",
    "describe",
    "imply",
    "suggest",
}

# A verb counts as an action verb when its first sense falls in one of these
# supersenses. WordNet has fifteen verb supersenses; these are the ten that
# remain once the four experiential ones and verb.stative are removed. Standing,
# remaining and consisting are states rather than doings, so verb.stative is
# deliberately outside both sets and a verb whose dominant sense is stative is
# counted by neither measure.
ACTION_SUPERSENSES = {
    "verb.body",
    "verb.change",
    "verb.competition",
    "verb.consumption",
    "verb.contact",
    "verb.creation",
    "verb.motion",
    "verb.possession",
    "verb.social",
    "verb.weather",
}

# Verbs whose first sense lands in an action supersense but whose use is
# stative, eventive or semi-copular rather than agentive. "Storms occur"
# reports an event and "the days get shorter" is a change of state; neither
# casts the subject as a doer. Checked against audit_agency_verbs.
AGENCY_VERB_STOPLIST = {"have", "get", "occur", "happen", "remain"}

# Parts of speech carrying lexical content, used for the norm-based extras.
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}

# --------------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------------- #

# The six category columns are derived from STED_CATEGORIES so that the schema
# cannot drift from the scorer. SelfTranscendence_full comes last because it is
# the only column matched without the exclusions or the surface stoplist, so it
# does not reconcile with the six.
CATEGORY_COLUMNS = [f"SelfTranscendence_{name}" for name in STED_CATEGORIES]

FEATURE_COLUMNS = (
    [
        "Verb_noun_ratio",
        "ActiveVoice_ratio",
        "ExperienceAttribution",
        "AgencyAttribution",
        "SelfTranscendence",
    ]
    + CATEGORY_COLUMNS
    + [
        "SelfTranscendence_full",
        "Sensorimotor",
        "Concreteness",
        "EpistemicOpenness",
    ]
)

OUTPUT_COLUMNS = (
    ["id", "band", "orientation", "author", "source", "word_count"]
    + FEATURE_COLUMNS
    + ["lancaster_coverage", "brysbaert_coverage"]
)

# --------------------------------------------------------------------------- #
# Resource loading
# --------------------------------------------------------------------------- #


def load_sted_stems(path):
    """Load and curate the STED stem lists.

    Multi-word entries are dropped because a stem is prefix-matched against a
    single token and cannot span a phrase.

    Args:
        path: Path to the category-tagged STED JSON file.

    Returns:
        A dict with four keys. ``primary`` is the criterion list, the union of
        awe and gratitude with the UK additions and minus the stem exclusions.
        ``categories`` holds each of the six sub-categories separately, curated
        the same way; the UK additions are absent from these because they carry
        no category tag, and stems cross-listed in two categories appear in
        both, so the six do not partition ``primary``. ``full`` is every
        single-word entry with no curation at all, the unmodified dictionary.
        ``tags`` is the raw stem-to-category mapping, for inspection.
    """
    with path.open(encoding="utf-8") as handle:
        categorised = json.load(handle)

    primary = [
        stem
        for stem, categories in categorised.items()
        if ("awe" in categories or "gratitude" in categories) and " " not in stem
    ]
    # Sorted so the list order, which the audit helpers use to attribute a
    # match to its first stem, is the same on every run.
    primary += [stem for stem in sorted(STED_UK_ADDITIONS) if stem not in primary]
    primary = [stem for stem in primary if stem not in STED_STEM_EXCLUSIONS]

    # The exclusions were chosen against awe and gratitude and are applied to
    # all six uniformly, so elevation loses "engag" and admiration loses
    # "recogn" on a decision made for another category.
    categories = {
        name: [
            stem
            for stem, tags in categorised.items()
            if name in tags
            and " " not in stem
            and stem not in STED_STEM_EXCLUSIONS
        ]
        for name in STED_CATEGORIES
    }

    return {
        "primary": primary,
        "categories": categories,
        "full": [stem for stem in categorised if " " not in stem],
        "tags": categorised,
    }


def load_single_word_lexicon(path):
    """Load a JSON word list, keeping single-word entries in lower case."""
    with path.open(encoding="utf-8") as handle:
        return {word.lower() for word in json.load(handle) if " " not in word}


def load_norms(path, word_column, value_column):
    """Map lower-cased words to their norm value from a lexicon CSV."""
    norms = pd.read_csv(path)
    return dict(zip(norms[word_column].str.lower(), norms[value_column]))


nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

NLP = spacy.load(SPACY_MODEL)

STED_STEMS = load_sted_stems(LEXICON_DIR / "sted_categorised.json")
HEDGES = load_single_word_lexicon(LEXICON_DIR / "hedges.json")
BOOSTERS = load_single_word_lexicon(LEXICON_DIR / "boosters.json")
LANCASTER_NORMS = load_norms(
    LEXICON_DIR / "Sensorimotor_norms.csv", "Word", "Minkowski3.sensorimotor"
)
BRYSBAERT_NORMS = load_norms(
    LEXICON_DIR / "Brysbaert_concreteness.csv", "Word", "Conc.M"
)

# --------------------------------------------------------------------------- #
# Feature primitives
# --------------------------------------------------------------------------- #


def is_nature_noun(lemma):
    """Return True if the lemma denotes a non-human natural entity."""
    if lemma in HUMAN_NOUNS:
        return False
    if lemma in NATURE_SUPPLEMENT:
        return True
    senses = wn.synsets(lemma, pos=wn.NOUN)
    if not senses:
        return False
    return senses[0].lexname() in NATURE_SUPERSENSES


def first_verb_supersense(lemma):
    """Return the WordNet supersense of the lemma's first verb sense.

    WordNet orders senses by frequency in the sense-tagged corpus, so the first
    sense is the dominant reading. Reading it is a type-level approximation to
    word sense disambiguation, which would resolve the verb against its actual
    sentence. Nothing here disambiguates in context, and both verb tests carry
    that limitation equally.
    """
    senses = wn.synsets(lemma, pos=wn.VERB)
    return senses[0].lexname() if senses else "none"


def is_experiential_verb(lemma):
    """Return True if the lemma's first sense denotes inner experience."""
    if lemma in EXPERIENTIAL_VERB_STOPLIST:
        return False
    return first_verb_supersense(lemma) in EXPERIENTIAL_SUPERSENSES


def is_action_verb(lemma):
    """Return True if the lemma's first sense denotes a material action."""
    if lemma in AGENCY_VERB_STOPLIST:
        return False
    return first_verb_supersense(lemma) in ACTION_SUPERSENSES


def is_experience_attribution(token):
    """Return True if the token is a nature noun subject of an experiential verb.

    The subject must be tagged NOUN, which excludes proper nouns, pronouns,
    determiners and nominalised adjectives before the lexicon is consulted.
    This matches the gate audit_nature_nouns applies, so the scorer and its
    audit count the same population.
    """
    return (
        token.dep_ == "nsubj"
        and token.pos_ == "NOUN"
        and token.head.pos_ == "VERB"
        and is_nature_noun(token.lemma_.lower())
        and is_experiential_verb(token.head.lemma_.lower())
    )


def is_agency_attribution(token):
    """Return True if the token is a nature noun subject of an action verb.

    The mirror image of is_experience_attribution: same subject slot, same
    nature-noun test, and a verb test built the same way on the same first
    sense. Since the experiential and action supersense sets are disjoint and
    both tests read one sense, no token can satisfy both. Together they split
    the nature-noun subjects of a verb into experiencers and doers, leaving
    uncounted only those whose verb is dominantly stative or stoplisted.
    """
    return (
        token.dep_ == "nsubj"
        and token.pos_ == "NOUN"
        and token.head.pos_ == "VERB"
        and is_nature_noun(token.lemma_.lower())
        and is_action_verb(token.head.lemma_.lower())
    )


def matched_stems(token, stems):
    """Return the stems that prefix-match the token's lemma."""
    lemma = token.lemma_.lower()
    return [stem for stem in stems if lemma.startswith(stem)]


def stem_rate_per_1k(tokens, word_count, stems, stoplist):
    """Count stem matches across the tokens and express them per 1,000 words.

    Proper nouns are skipped so that place and organisation names cannot inflate
    the criterion, and any surface form on the stoplist is skipped even when its
    lemma matches a retained stem.
    """
    count = sum(
        1
        for token in tokens
        if token.pos_ != "PROPN"
        and token.text.lower() not in stoplist
        and any(token.lemma_.lower().startswith(stem) for stem in stems)
    )
    return (count / word_count) * 1000 if word_count else 0


def lookup_norm(token, norms):
    """Look up a token in a norm dictionary, trying the lemma before the surface form."""
    lemma = token.lemma_.lower()
    if lemma in norms:
        return norms[lemma]
    surface = token.text.lower()
    if surface in norms:
        return norms[surface]
    return None


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def score_text(text):
    """Score a single text on every feature and return them as a Series."""
    doc = NLP(text)
    words = [token for token in doc if not token.is_punct and not token.is_space]
    word_count = len(words)

    # Verb_noun_ratio. spaCy tags auxiliaries as AUX, so this counts lexical
    # verbs only.
    verb_count = sum(1 for token in doc if token.pos_ == "VERB")
    noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
    verb_noun_ratio = verb_count / noun_count if noun_count else 0

    # ExperienceAttribution.
    attribution_count = sum(1 for token in doc if is_experience_attribution(token))
    attribution_rate = (
        (attribution_count / word_count) * 1000 if word_count else 0
    )

    # AgencyAttribution, the doer counterpart over the same subject slot. The
    # two verb tests are disjoint, so a nature-noun subject is counted by at
    # most one of them.
    agency_count = sum(1 for token in doc if is_agency_attribution(token))
    agency_rate = (agency_count / word_count) * 1000 if word_count else 0

    # SelfTranscendence and its variants. The full dictionary is matched without
    # the surface stoplist so that the transparency column stays uncurated.
    self_transcendence = stem_rate_per_1k(
        words, word_count, STED_STEMS["primary"], STED_SURFACE_STOPLIST
    )
    category_rates = {
        f"SelfTranscendence_{name}": stem_rate_per_1k(
            words, word_count, stems, STED_SURFACE_STOPLIST
        )
        for name, stems in STED_STEMS["categories"].items()
    }
    self_transcendence_full = stem_rate_per_1k(
        words, word_count, STED_STEMS["full"], set()
    )

    # Sensorimotor and Concreteness as mean norm values over content words.
    content_tokens = [token for token in words if token.pos_ in CONTENT_POS]

    sensorimotor_values = []
    concreteness_values = []
    for token in content_tokens:
        sensorimotor = lookup_norm(token, LANCASTER_NORMS)
        if sensorimotor is not None:
            sensorimotor_values.append(sensorimotor)
        concreteness = lookup_norm(token, BRYSBAERT_NORMS)
        if concreteness is not None:
            concreteness_values.append(concreteness)

    mean_sensorimotor = (
        sum(sensorimotor_values) / len(sensorimotor_values)
        if sensorimotor_values
        else 0
    )
    mean_concreteness = (
        sum(concreteness_values) / len(concreteness_values)
        if concreteness_values
        else 0
    )

    # Coverage records how much of each text the norms actually reached, which
    # is what makes the two extras interpretable.
    lancaster_coverage = (
        len(sensorimotor_values) / len(content_tokens) if content_tokens else 0
    )
    brysbaert_coverage = (
        len(concreteness_values) / len(content_tokens) if content_tokens else 0
    )

    # EpistemicOpenness as hedge density minus booster density per 1,000 words.
    hedge_count = sum(1 for token in words if token.lemma_.lower() in HEDGES)
    booster_count = sum(1 for token in words if token.lemma_.lower() in BOOSTERS)
    epistemic_openness = (
        ((hedge_count - booster_count) / word_count) * 1000 if word_count else 0
    )

    return pd.Series(
        {
            "word_count": word_count,
            "Verb_noun_ratio": verb_noun_ratio,
            "ExperienceAttribution": attribution_rate,
            "AgencyAttribution": agency_rate,
            "SelfTranscendence": self_transcendence,
            **category_rates,
            "SelfTranscendence_full": self_transcendence_full,
            "Sensorimotor": mean_sensorimotor,
            "Concreteness": mean_concreteness,
            "EpistemicOpenness": epistemic_openness,
            "lancaster_coverage": lancaster_coverage,
            "brysbaert_coverage": brysbaert_coverage,
        }
    )


def load_corpus():
    """Read the corpus text files and join them to the metadata on ``id``."""
    rows = [
        {"id": path.stem, "text": path.read_text(encoding="utf-8")}
        for path in sorted(CORPUS_DIR.glob("*.txt"))
    ]
    metadata = pd.read_csv(METADATA_PATH)
    return pd.DataFrame(rows).merge(metadata, on="id", how="left")


def add_active_voice(df):
    """Add the ActiveVoice_ratio column from PassivePy's corpus-level output.

    PassivePy runs over the whole corpus rather than text by text and returns
    rows positionally, so the ``id`` order is checked before assignment.
    ``passive_percentages`` is a 0-1 proportion, and this inverts it so that
    high values mean a text written mostly in the active voice. It counts every
    sentence in the text, whatever the sentence is about, so it says nothing
    about who or what is doing the acting.
    """
    analyzer = PassivePyAnalyzer(spacy_model=SPACY_MODEL)
    results = analyzer.match_corpus_level(df[["id", "text"]].copy(), column_name="text")

    assert results["id"].tolist() == df["id"].tolist(), (
        "PassivePy returned rows in a different order to the corpus"
    )

    # PassivePy returns this column as object dtype, which .round() silently
    # skips, so this would be the only column written at full precision.
    df["ActiveVoice_ratio"] = pd.to_numeric(1 - results["passive_percentages"]).values
    return df


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main():
    """Score the corpus, write the scores file, and print the descriptives."""
    df = load_corpus()
    print(f"Loaded {len(df)} texts")
    print(df.groupby("band").size())

    counts = ", ".join(
        f"{name} {len(stems)}" for name, stems in STED_STEMS["categories"].items()
    )
    print(
        f"STED stems: primary {len(STED_STEMS['primary'])}, "
        f"full {len(STED_STEMS['full'])}"
    )
    print(f"  by category: {counts}")
    print(
        f"Hedges: {len(HEDGES)}  Boosters: {len(BOOSTERS)}  "
        f"Lancaster: {len(LANCASTER_NORMS)}  Brysbaert: {len(BRYSBAERT_NORMS)}"
    )

    print("Scoring...")
    df = pd.concat([df, df["text"].apply(score_text)], axis=1)
    df = add_active_voice(df)

    scores = df[OUTPUT_COLUMNS].round(3)

    # Rounding promotes word_count to a float, so put it back.
    scores["word_count"] = scores["word_count"].astype(int)

    scores.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(scores)} rows to {OUTPUT_PATH}")

    pd.set_option("display.width", None)
    pd.set_option("display.max_columns", None)

    print("\nScores by text:")
    print(
        scores[
            [
                "id",
                "author",
                "band",
                "Verb_noun_ratio",
                "ActiveVoice_ratio",
                "ExperienceAttribution",
                "AgencyAttribution",
                "SelfTranscendence",
            ]
        ]
        .sort_values(["band", "author"])
        .to_string(index=False)
    )

    # decoupled sorts first alphabetically, so reindex to keep the gradient in
    # order and the control band last.
    band_order = ["instrumental", "middle", "relational", "decoupled"]
    print("\nBand means:")
    print(scores.groupby("band")[FEATURE_COLUMNS].mean().round(3).reindex(band_order))

    return df


# --------------------------------------------------------------------------- #
# Utilities
#
# Manual audits and per-text inspection. These are not part of the scoring run
# and are called by hand from an interactive session.
# --------------------------------------------------------------------------- #


def first_noun_supersense(word):
    """Return the WordNet supersense of the word's first noun sense."""
    senses = wn.synsets(word, pos=wn.NOUN)
    return senses[0].lexname() if senses else "none"


def audit_corpus(df):
    """Print the corpus-level quality checks against a scored DataFrame."""
    print("\n--- Word count outliers ---")
    outliers = df[
        (df["word_count"] < WORD_COUNT_MIN) | (df["word_count"] > WORD_COUNT_MAX)
    ][["id", "band", "word_count"]].sort_values("word_count")
    print(
        outliers.to_string(index=False)
        if len(outliers)
        else f"  All within {WORD_COUNT_MIN}-{WORD_COUNT_MAX}"
    )

    for column in ["lancaster_coverage", "brysbaert_coverage"]:
        print(f"\n--- {column} below {COVERAGE_FLOOR:.0%} ---")
        low_coverage = df[df[column] < COVERAGE_FLOOR][
            ["id", "band", column]
        ].sort_values(column)
        print(
            low_coverage.to_string(index=False)
            if len(low_coverage)
            else f"  All above {COVERAGE_FLOOR:.0%}"
        )

    # Parsing is the expensive step, so everything that needs a parsed document
    # is collected in one pass over the corpus rather than a loop per check.
    dominance_lines = []
    experience_lines = []
    agency_lines = []

    for _, row in df.iterrows():
        doc = NLP(row["text"])

        # A single stem driving most of a text's criterion score is the
        # signature of a false positive, so flag it for manual inspection.
        hits = Counter()
        for token in doc:
            if token.is_punct or token.is_space or token.pos_ == "PROPN":
                continue
            matches = matched_stems(token, STED_STEMS["primary"])
            if matches:
                hits[matches[0]] += 1

        total = sum(hits.values())
        if total >= 3:
            top = hits.most_common(3)
            flag = "  ! " if top[0][1] / total > STEM_DOMINANCE_THRESHOLD else "    "
            summary = ", ".join(f"{stem}({count})" for stem, count in top)
            dominance_lines.append(
                f"{flag}{row['id']:15s} {row['band']:12s} "
                f"total={total:2d}  top: {summary}"
            )

        experience = [
            f"{token.text}->{token.head.text}"
            for token in doc
            if is_experience_attribution(token)
        ]
        if experience:
            experience_lines.append(
                f"  {row['id']:15s} {row['band']:12s} "
                f"[{len(experience):2d}] {', '.join(experience)}"
            )

        agency = [
            f"{token.text}->{token.head.text}"
            for token in doc
            if is_agency_attribution(token)
        ]
        if agency:
            agency_lines.append(
                f"  {row['id']:15s} {row['band']:12s} "
                f"[{len(agency):2d}] {', '.join(agency)}"
            )

    print(
        f"\n--- STED stem dominance "
        f"(! = top stem above {STEM_DOMINANCE_THRESHOLD:.0%} of the score) ---"
    )
    print("\n".join(dominance_lines))

    print("\n--- All ExperienceAttribution firings ---")
    print("\n".join(experience_lines))

    print("\n--- All AgencyAttribution firings ---")
    print("\n".join(agency_lines))


def debug_text(text_id):
    """Print a full feature breakdown for a single text, read from disk."""
    matches = list(CORPUS_DIR.glob(f"{text_id}.txt"))
    if not matches:
        print(f"No file found for {text_id}")
        return

    doc = NLP(matches[0].read_text(encoding="utf-8"))
    words = [token for token in doc if not token.is_punct and not token.is_space]
    word_count = len(words)

    print(f"\n{'=' * 60}\n  {text_id}  ({word_count} words)\n{'=' * 60}")

    verb_count = sum(1 for token in doc if token.pos_ == "VERB")
    noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
    if noun_count:
        ratio = verb_count / noun_count
        print(f"\nV:N: verbs={verb_count} nouns={noun_count} ratio={ratio:.3f}")
    else:
        print("\nV:N: no nouns")

    print("\nExperienceAttribution firings:")
    firings = [token for token in doc if is_experience_attribution(token)]
    for index, token in enumerate(firings, start=1):
        print(f"  [{index}] '{token.text}' -> '{token.head.text}'")
        print(f"       {token.sent.text.strip()[:120]}")
    if not firings:
        print("  None")
    print(f"  Rate: {(len(firings) / word_count) * 1000:.3f} per 1k")

    print("\nAgencyAttribution firings:")
    agency_firings = [token for token in doc if is_agency_attribution(token)]
    for index, token in enumerate(agency_firings, start=1):
        print(f"  [{index}] '{token.text}' -> '{token.head.text}'")
        print(f"       {token.sent.text.strip()[:120]}")
    if not agency_firings:
        print("  None")
    # The two verb tests are disjoint by construction, so this is a guard: if it
    # ever prints anything but zero, one of the supersense sets has drifted.
    shared = sum(1 for token in agency_firings if is_experience_attribution(token))
    print(f"  Also counted as ExperienceAttribution: {shared} (should be 0)")
    print(f"  Rate: {(len(agency_firings) / word_count) * 1000:.3f} per 1k")

    print("\nSelfTranscendence (awe + gratitude) matches:")
    sted_count = 0
    for token in words:
        if token.pos_ == "PROPN" or token.text.lower() in STED_SURFACE_STOPLIST:
            continue
        matches = matched_stems(token, STED_STEMS["primary"])
        if matches:
            sted_count += 1
            tags = STED_STEMS["tags"].get(matches[0], "UK spelling")
            print(f"  [{sted_count}] '{token.text}' <- stem '{matches[0]}' ({tags})")
    if not sted_count:
        print("  None")
    print(f"  Rate: {(sted_count / word_count) * 1000:.3f} per 1k")

    content_tokens = [token for token in words if token.pos_ in CONTENT_POS]
    sensorimotor_values = []
    concreteness_values = []
    misses = []
    for token in content_tokens:
        sensorimotor = lookup_norm(token, LANCASTER_NORMS)
        if sensorimotor is None:
            misses.append(token.text)
        else:
            sensorimotor_values.append(sensorimotor)
        concreteness = lookup_norm(token, BRYSBAERT_NORMS)
        if concreteness is not None:
            concreteness_values.append(concreteness)

    coverage = len(sensorimotor_values) / len(content_tokens) if content_tokens else 0
    mean_sensorimotor = (
        sum(sensorimotor_values) / len(sensorimotor_values)
        if sensorimotor_values
        else 0
    )
    mean_concreteness = (
        sum(concreteness_values) / len(concreteness_values)
        if concreteness_values
        else 0
    )
    print(
        f"\nSensorimotor: coverage {len(sensorimotor_values)}/{len(content_tokens)} "
        f"= {coverage:.1%}  mean {mean_sensorimotor:.3f}"
    )
    print(
        f"Concreteness: mean {mean_concreteness:.3f} "
        f"over {len(concreteness_values)} words"
    )
    if misses:
        trailing = "..." if len(misses) > 20 else ""
        print(f"  Not in Lancaster ({len(misses)}): {', '.join(misses[:20])}{trailing}")

    hedge_hits = [token.text for token in words if token.lemma_.lower() in HEDGES]
    booster_hits = [token.text for token in words if token.lemma_.lower() in BOOSTERS]
    hedge_rate = (len(hedge_hits) / word_count) * 1000
    booster_rate = (len(booster_hits) / word_count) * 1000
    print(
        f"\nEpistemicOpenness: hedges {len(hedge_hits)} ({hedge_rate:.2f}/1k), "
        f"boosters {len(booster_hits)} ({booster_rate:.2f}/1k), "
        f"diff {hedge_rate - booster_rate:.3f}"
    )


def audit_agency_verbs(df, min_count=2):
    """List the verbs driving AgencyAttribution and its overlap with the other.

    The stoplist is a judgement call, so this is the check that keeps it
    honest: run it, read the verbs, and add anything stative or auxiliary in
    use rather than agentive. The closing line confirms the two attribution
    measures are disjoint and gives the corpus totals behind both.
    """
    verbs = Counter()
    subjects = defaultdict(set)
    agency_total = 0
    experience_total = 0
    shared_total = 0

    for text in df["text"]:
        doc = NLP(text)
        for token in doc:
            agency = is_agency_attribution(token)
            experience = is_experience_attribution(token)
            agency_total += agency
            experience_total += experience
            shared_total += agency and experience
            if agency:
                verb = token.head.lemma_.lower()
                verbs[verb] += 1
                subjects[verb].add(token.lemma_.lower())

    print("\n--- AgencyAttribution verbs (check for stative or auxiliary uses) ---")
    for verb, count in verbs.most_common():
        if count < min_count:
            continue
        nouns = ", ".join(sorted(subjects[verb])[:4])
        print(f"  {verb:16s} x{count:<3d} {first_verb_supersense(verb):18s} <- {nouns}")

    print(
        f"\n  Firings across the corpus: agency {agency_total}, "
        f"experience {experience_total}, counted by both {shared_total} (should be 0)"
    )


def audit_nature_nouns(df, min_count=2):
    """Compare nouns the nature test accepts against those it rejects.

    Applies the shared attribution gate with the nature test removed, keeping
    any subject whose verb passes either the experiential or the action test.
    The rejected nouns are the candidate additions to ``NATURE_SUPPLEMENT`` and
    the accepted ones can be scanned for false positives. The lexicon serves
    both attribution measures, so the audit covers both; human nouns appearing
    in the rejected list confirm the block is working.
    """
    missed = Counter()
    counted = Counter()
    missed_verbs = defaultdict(set)
    counted_verbs = defaultdict(set)

    for text in df["text"]:
        doc = NLP(text)
        for token in doc:
            if token.dep_ != "nsubj" or token.pos_ != "NOUN":
                continue
            if token.head.pos_ != "VERB":
                continue
            verb = token.head.lemma_.lower()
            if not (is_experiential_verb(verb) or is_action_verb(verb)):
                continue

            lemma = token.lemma_.lower()
            if is_nature_noun(lemma):
                counted[lemma] += 1
                counted_verbs[lemma].add(verb)
            else:
                missed[lemma] += 1
                missed_verbs[lemma].add(verb)

    print("\n--- Rejected (candidate supplement additions) ---")
    for lemma, count in missed.most_common():
        if count < min_count:
            continue
        verbs = ", ".join(sorted(missed_verbs[lemma])[:4])
        print(f"  {lemma:16s} x{count:<2d} {first_noun_supersense(lemma):18s} -> {verbs}")

    print("\n--- Accepted (firing as nature, check for false positives) ---")
    for lemma, count in counted.most_common():
        if count < min_count:
            continue
        verbs = ", ".join(sorted(counted_verbs[lemma])[:4])
        print(f"  {lemma:16s} x{count:<2d} {first_noun_supersense(lemma):18s} -> {verbs}")


if __name__ == "__main__":
    df = main()

    # Inspection after the scoring run. Comment these three out for a fast run:
    # audit_corpus reparses all 69 texts, which roughly doubles the runtime.
    # debug_text()
    # audit_corpus(df)
