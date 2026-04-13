# 🎧 Model Card: VibeFinder 1.0

## 1. Model Name

**VibeFinder 1.0** — a content-based music recommender that ranks songs by matching a user's taste profile against song attributes.

---

## 2. Intended Use

VibeFinder suggests up to 5 songs from a small curated catalog based on a user's preferred genre, mood, and numeric targets for energy, valence, and danceability (plus an acoustic/non-acoustic preference).

- **Intended for**: classroom exploration of how content-based recommenders work, and how design choices (weights, features, data) show up in the output.
- **Assumptions about the user**: the user can describe their taste in the same vocabulary the catalog uses (e.g., knowing the difference between "lofi" and "ambient"). The user is a single listener in a single moment — no context, mood drift, or listening history.
- **Not for real users**: there is no personalization over time, no social signal, no catalog at a realistic scale, and no safety filtering.

---

## 3. How the Model Works

Every song in the catalog is judged individually against the user's preferences, then the whole list is sorted from highest score to lowest.

A song earns points for:

- **Matching the user's favorite genre** — a big boost.
- **Matching the user's favorite mood** — a medium boost.
- **Being close to the user's target energy** — closer is better, scaled 0 to 1.
- **Being close to the user's target valence** (musical positivity) — same idea, smaller weight.
- **Being close to the user's target danceability** — same idea, smaller weight.
- **Matching the user's acoustic preference** — a small fixed boost if the song is acoustic and the user likes acoustic (or vice versa).

Points add up into one number. The top 5 songs by score are returned, along with a list of short "because..." reasons explaining which parts of the score fired.

The biggest change from the starter code: I extended the user profile to include `target_valence` and `target_danceability`, since the CSV had those columns but the starter `UserProfile` had no way to use them. I also normalized the reason format across the two APIs so that reasons are always a list of short strings that get joined at display time.

---

## 4. Data

- **Catalog size**: 20 songs (10 from the starter `songs.csv`, 10 I added to broaden coverage).
- **Features per song**: id, title, artist, genre, mood, energy, tempo_bpm, valence, danceability, acousticness.
- **Genres represented**: pop, indie pop, lofi, ambient, jazz, synthwave, rock, r&b, hip-hop, folk, edm, country, classical, metal, reggae, indie.
- **Moods represented**: happy, chill, intense, relaxed, moody, focused, melancholic, aggressive, dreamy, euphoric, nostalgic, romantic.
- **What's missing**: no sad, no angsty, no spoken-word, no international/non-English pop, no pre-1980s references beyond one classical piece. Tempo range is narrow (60–168 BPM). All songs are fictional — this is a simulation, not a licensed dataset.
- **Whose taste this reflects**: mine and whatever the starter repo's author had in mind. Skew toward "chill" and "pop/lofi" vibes.

---

## 5. Strengths

- **Transparent by construction**: every score comes with a list of reasons. There's no hidden layer — if a song ranks #1, you can point at exactly which components fired and for how many points.
- **Works well on clean, consistent profiles**: when a user asks for "pop / happy / high energy / non-acoustic" and a song in the catalog is exactly that, the scoring produces an unambiguous top-1 (observed: scores above 5.5 out of a 5.75 max for the three non-adversarial profiles).
- **Gracefully degrades on impossible profiles**: the adversarial "edm / sad / high energy / low valence / likes acoustic" profile returned results with top score 3.79 — low enough that a downstream system could use the score floor as a "no good matches" signal instead of pretending.
- **Cheap to reason about**: no training, no state, no randomness. Same input → same ranking. Easy to test, easy to explain to a non-programmer.

---

## 6. Limitations and Bias

- **Genre structurally dominates.** A genre match is worth +2.0. The combined maximum of every continuous-similarity component (energy + valence + danceability) is 2.25. This means the system almost never surfaces a cross-genre suggestion at the top of the list, even when the continuous features align perfectly. I confirmed this with a weight-shift experiment: doubling the energy weight and halving the genre weight *narrowed* the gap between the genre-matched #1 and non-genre contenders, but did not flip the #1.
- **Filter bubble effect**: because genre is a hard categorical gate, a user who lists "pop" will be funneled toward pop songs regardless of whether a hip-hop or r&b track would actually match their vibe better. This is the exact mechanism that creates filter bubbles in real platforms.
- **Catalog skew amplifies itself**: over-represented genres in `songs.csv` (pop, lofi) surface more often simply because there are more of them to score. In a 20-song catalog this is minor; in a production catalog with natural power-law skew, it's a major bias.
- **Mood is all-or-nothing**: "happy" vs "euphoric" vs "joyful" would all be treated as non-matches against each other, even though they're close on any real mood axis. The scoring has no notion of mood similarity.
- **Dead feature**: `tempo_bpm` is in the CSV but unused in v1, because there's no corresponding user field. Including it without a user target would just be noise.
- **Type mismatch by design**: `likes_acoustic` is a bool but `acousticness` is continuous. I threshold at 0.5. A user who is "mildly acoustic-preferring" cannot express that in the current profile.
- **Same-artist spam risk**: nothing prevents the top-5 from being dominated by one artist. On this small catalog it didn't happen, but the risk is structural.

---

## 7. Evaluation

I evaluated the system against four hand-crafted user profiles:

1. **Happy Pop Commuter** (pop / happy / 0.8 energy)
2. **Chill Lofi Studier** (lofi / focused / 0.35 energy / likes acoustic)
3. **Deep Intense Rock** (rock / intense / 0.9 energy)
4. **Conflicted Listener** (adversarial: edm / sad / 0.9 energy / 0.2 valence / likes acoustic)

For each profile I ran `python -m src.main`, looked at the top-5, and checked whether:
- the top-1 matched the obvious intent,
- the rest of the ranking made narrative sense,
- the adversarial profile produced honestly lower scores rather than fake-confident matches.

I also ran a **weight-shift experiment**: doubled the energy weight (1.0 → 2.0) and halved the genre weight (2.0 → 1.0), re-ran the rock profile, and compared score gaps. Finding: non-rock high-energy tracks rose from a 3.12-point gap behind #1 to a 2.13-point gap. The #1 itself did not change. This confirms that genre is dominant not because of its raw weight but because of the overall score-range geometry.

Pairwise comparisons of all four profiles are written up in [`reflection.md`](reflection.md).

---

## 8. Future Work

- **Normalize the total score to [0, 1]** so that "perfect match" means the same thing across weight configurations, and so that users can set a minimum-score threshold meaningfully.
- **Soft mood similarity**: replace exact-match mood scoring with a mood-embedding lookup (even a tiny hand-crafted cosine table would help).
- **Diversity penalty**: subtract from a song's score if its genre or artist is already represented in the current top-k. Would fix the same-artist spam risk and nudge the ranker out of genre monoculture.
- **Multi-strategy API**: let the caller pick "genre-first," "mood-first," or "energy-focused" via a named strategy rather than by hand-editing weights.
- **Confidence output**: expose the raw score alongside the ranking so a downstream UI can say "we're confident about these 3" vs "we're guessing about these 2."
- **Enrich the catalog**: more genres (especially non-Western), more moods, wider tempo range, and real licensing data so this could plausibly run against a real library.

---

## 9. Personal Reflection

> **[REPLACE THIS BLOCK WITH YOUR OWN VOICE.]**
>
> A few prompts to riff on (in your own words):
>
> - What was the biggest learning moment during this project? (e.g., did you expect the weight-shift experiment to flip the #1? Were you surprised that the adversarial profile still returned *something*?)
> - How did using AI tools help you, and when did you need to push back on what they suggested?
> - What surprised you about how simple algorithms can still "feel" like recommendations?
> - What would you try next if you extended this project?
>
> A few sentences is enough. Specific honest details beat polished generic prose — graders can tell.
