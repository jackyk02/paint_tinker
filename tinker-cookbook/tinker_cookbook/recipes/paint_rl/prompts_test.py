from collections import Counter

from tinker_cookbook.recipes.paint_rl.prompts import (
    COLORS,
    N_TEST_PROMPTS,
    N_TRAIN_PROMPTS,
    SKILL_PROMPTS,
    STYLES,
    SUBJECTS,
    build_prompt_pool,
    build_prompt_splits,
    system_prompt,
)

# The held-out set every arm (verifier and judge) is evaluated on (seed 0,
# 50 prompts spread over every subject). Pinned on purpose: an edit to
# the pool or the splitter that moves these prompts makes runs incomparable,
# so it has to be a deliberate change here as well.
PINNED_HELDOUT_SEED0 = [
    ("s12c06t0", "Paint a violet poppy field under a summer sky in wet-on-wet watercolor wash."),
    ("s33c09t3", "Paint an olive stack of old books on a windowsill in layered watercolor glazes."),
    ("s29c04t1", "Paint a teal watering can in minimal watercolor with lots of white paper."),
    (
        "s36c12t2",
        "Paint a blue violet straw hat hanging on a chair beside an open window in watercolor with soft ink outlines.",
    ),
    ("s44c03t3", "Paint an emerald sailboat on a calm sea in layered watercolor glazes."),
    ("s42c08t0", "Paint a magenta mountain lake at dawn in wet-on-wet watercolor wash."),
    (
        "s47c05t2",
        "Paint an indigo village street leading to a church under a crescent moon in watercolor with soft ink outlines.",
    ),
    ("s05c07t1", "Paint a sepia lily in minimal watercolor with lots of white paper."),
    (
        "s07c10t2",
        "Paint a sky blue orchid in a terracotta pot in watercolor with soft ink outlines.",
    ),
    (
        "s51c01t1",
        "Paint a coral flock of birds in flight across the sky in minimal watercolor with lots of white paper.",
    ),
    (
        "k04t0",
        "Paint a round ceramic bowl with a narrow base and wide opening, in wet-on-wet watercolor wash.",
    ),
    (
        "k06t3",
        "Paint a winding road that becomes narrower toward the horizon, in layered watercolor glazes.",
    ),
    ("s23c02t3", "Paint a golden pomegranate in layered watercolor glazes."),
    (
        "s27c11t0",
        "Paint a charcoal gray bunch of bananas beside a striped teacup in wet-on-wet watercolor wash.",
    ),
    ("s28c00t2", "Paint a crimson hot air balloon in watercolor with soft ink outlines."),
    (
        "k02t1",
        "Paint a blue vase to the left of a yellow teacup, in minimal watercolor with lots of white paper.",
    ),
    ("s00c11t1", "Paint a charcoal gray sunflower in minimal watercolor with lots of white paper."),
    ("s01c03t3", "Paint an emerald rose in layered watercolor glazes."),
    ("s41c00t2", "Paint a crimson barn in watercolor with soft ink outlines."),
    ("s38c08t0", "Paint a magenta lighthouse in wet-on-wet watercolor wash."),
    (
        "s14c02t2",
        "Paint a golden bouquet of peonies in a ceramic jug in watercolor with soft ink outlines.",
    ),
    (
        "s15c07t0",
        "Paint a sepia wisteria vine trailing over a garden gate in wet-on-wet watercolor wash.",
    ),
    (
        "s52c09t1",
        "Paint an olive fish swimming in a clear stream in minimal watercolor with lots of white paper.",
    ),
    (
        "s46c12t3",
        "Paint a blue violet lighthouse with a sailboat passing in front of it in layered watercolor glazes.",
    ),
    (
        "s20c05t2",
        "Paint an indigo field of flowers with a path leading to a distant cottage in watercolor with soft ink outlines.",
    ),
    (
        "s35c04t1",
        "Paint a teal bowl of lemons beside a glass bottle in minimal watercolor with lots of white paper.",
    ),
    (
        "s10c01t0",
        "Paint a coral lotus blossom floating on a still pond in wet-on-wet watercolor wash.",
    ),
    ("s45c06t3", "Paint a violet forest in autumn in layered watercolor glazes."),
    (
        "s11c10t1",
        "Paint a sky blue cherry blossom branch against a pale sky in minimal watercolor with lots of white paper.",
    ),
    ("s03c04t3", "Paint a teal daisy in layered watercolor glazes."),
    ("s17c06t0", "Paint a violet field of flowers in wet-on-wet watercolor wash."),
    (
        "k05t2",
        "Paint a diamond-shaped kite with a long ribbon tail, in watercolor with soft ink outlines.",
    ),
    ("s21c09t0", "Paint an olive pear in wet-on-wet watercolor wash."),
    ("s04c12t3", "Paint a blue violet iris in layered watercolor glazes."),
    ("s40c03t1", "Paint an emerald cottage in minimal watercolor with lots of white paper."),
    (
        "s49c10t2",
        "Paint a sky blue classical building with columns and arched windows in watercolor with soft ink outlines.",
    ),
    (
        "s09c02t0",
        "Paint a golden lavender field stretching toward a distant farmhouse in wet-on-wet watercolor wash.",
    ),
    (
        "s34c01t3",
        "Paint a coral basket of apples on a checked tablecloth in layered watercolor glazes.",
    ),
    (
        "k07t2",
        "Paint a row of trees along a canal, shrinking into the distance, in watercolor with soft ink outlines.",
    ),
    ("s39c07t1", "Paint a sepia windmill in minimal watercolor with lots of white paper."),
    ("s06c11t2", "Paint a charcoal gray hydrangea in watercolor with soft ink outlines."),
    ("s43c08t0", "Paint a magenta old oak tree alone in a meadow in wet-on-wet watercolor wash."),
    (
        "s18c00t1",
        "Paint a crimson field of flowers on a rolling hillside in minimal watercolor with lots of white paper.",
    ),
    ("s08c05t3", "Paint an indigo foxgloves along a cottage wall in layered watercolor glazes."),
    ("k00t3", "Paint exactly five red tulips in a vase, in layered watercolor glazes."),
    (
        "s13c06t1",
        "Paint a violet magnolia branch in a glass vase in minimal watercolor with lots of white paper.",
    ),
    ("s37c05t0", "Paint an indigo mailbox in wet-on-wet watercolor wash."),
    ("s22c07t2", "Paint a sepia peach in watercolor with soft ink outlines."),
    (
        "s30c12t2",
        "Paint a blue violet cup of coffee on a wooden table in watercolor with soft ink outlines.",
    ),
    ("s19c02t3", "Paint a golden field of flowers at sunset in layered watercolor glazes."),
]


def test_pool_shape():
    pool = build_prompt_pool()
    # skill prompts are crossed with the styles only
    base = len(SUBJECTS) * len(COLORS) * len(STYLES)
    assert len(pool) == base + len(SKILL_PROMPTS) * len(STYLES) == 2788
    assert len({p.id for p in pool}) == len(pool)
    assert len({s.phrase for s in SUBJECTS}) == len(SUBJECTS)
    assert Counter(s.family for s in SUBJECTS) == {
        "flower": 21, "object": 17, "scene": 13, "animal": 2,
    }  # fmt: skip
    assert Counter(s.tier for s in SUBJECTS) == {1: 19, 2: 24, 3: 10}


def test_animal_family():
    assert {s.name for s in SUBJECTS if s.family == "animal"} == {
        "flock of birds", "fish",
    }  # fmt: skip


def test_subject_variants():
    pool = build_prompt_pool()
    assert sum(s.name == "field of flowers" for s in SUBJECTS) == 4
    assert len([p for p in pool if p.subject == "field of flowers"]) == len(COLORS) * len(STYLES)


def test_articles():
    pool = build_prompt_pool()
    for p in pool:
        if p.family == "skill":  # worded in full, no color slot
            continue
        article = p.text.split()[1]
        assert article == ("an" if p.color[0] in "aeiou" else "a"), p.text


def _spread(counts: Counter[str]) -> int:
    return max(counts.values()) - min(counts.values())


def test_default_split_is_pinned():
    _, test = build_prompt_splits(seed=0)
    assert [(p.id, p.text) for p in test] == PINNED_HELDOUT_SEED0
    assert all(p.kind == "test" for p in test)


def test_split_invariants():
    train, test = build_prompt_splits(seed=0)
    train2, test2 = build_prompt_splits(seed=0)
    assert [p.id for p in train] == [p.id for p in train2]
    assert [p.id for p in test] == [p.id for p in test2]
    assert not {p.id for p in train} & {p.id for p in test}
    assert len(train) == N_TRAIN_PROMPTS and len(test) == N_TEST_PROMPTS
    # no held-out subjects: every subject and skill prompt is trained on, and
    # the 50 held-out prompts are 50 different ones of them
    everything = {s.phrase for s in SUBJECTS} | {k.request for k in SKILL_PROMPTS}
    assert {p.subject for p in train} == everything
    assert len({p.subject for p in test}) == min(N_TEST_PROMPTS, len(everything))
    # the simple subject base is balanced over subjects, colors and styles
    for prompts in (train, test):
        base = [p for p in prompts if p.family != "skill"]
        assert _spread(Counter(p.subject for p in base)) <= 1
        colors = Counter(p.color for p in base)
        styles = Counter(p.style for p in prompts)
        assert set(colors) == set(COLORS) and _spread(colors) <= 1
        assert set(styles) == set(STYLES) and _spread(styles) <= 2
    assert all(p.kind == "train" for p in train)


def test_skill_prompts_are_a_small_share():
    train, test = build_prompt_splits(seed=0)
    skill_train = [p for p in train if p.family == "skill"]
    assert 0.05 <= len(skill_train) / len(train) <= 0.12
    assert {k.category for k in SKILL_PROMPTS} == {
        "object counting", "spatial relationships", "shape and geometry", "depth and perspective",
    }  # fmt: skip
    for p in build_prompt_pool():
        if p.family == "skill":
            assert p.color == "" and p.text.endswith(f", in {p.style}."), p.text
            assert not any(
                p.text.startswith(f"Paint {a} {c} ") for a in ("a", "an") for c in COLORS
            )


def test_other_seeds_and_sizes_work():
    train, test = build_prompt_splits(n_train=40, n_test=8, seed=7)
    assert len(train) == 40 and len(test) == 8
    assert not {p.id for p in train} & {p.id for p in test}


def test_system_prompt_mentions_canvas_size():
    text = system_prompt(512)
    assert "createCanvas(512, 512, WEBGL)" in text
    assert "-256" in text and "{size}" not in text


def test_system_prompt_does_not_ask_for_noloop():
    # The page stops the draw loop itself; a noLoop() rule invites brush.noLoop().
    assert "noLoop" not in system_prompt(512)
