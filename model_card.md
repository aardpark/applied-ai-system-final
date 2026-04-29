# 🎧 Model Card: Antoine v2 (Audio-Similarity Music Recommender)

## 1. Model Name

**Antoine v2** — extends the Module 3 Antoine project. The original heuristic recommender lives in `src/baseline.py` for comparison; the deployed v2 system is in `src/recommender.py`.

---

## 2. Intended Use

Given a song you like, recommend 5 others from a small personal catalog that *sound like it* — not just match its tag. The seed can be a song already in the catalog (instant) or any YouTube URL (downloads the embedding model on first use).

---

## 3. How the Model Works

Every song in the catalog is run through MERT (`m-a-p/MERT-v1-95M`), a pretrained transformer trained on music for representation learning. MERT outputs a 768-dimensional vector per song that captures timbre, rhythm, and texture. Recommendations are the cosine-nearest neighbors in that vector space, after subtracting the catalog's global mean (a standard fix for "narrow cone" anisotropy in pretrained contrastive embeddings).

A spectrogram fingerprint (224×224 mel-spectrogram PNG) is also generated per song. It isn't used for ranking — it's a human-legible artifact so you can *see* why two songs ended up neighbors.

---

## 4. Data

24 songs from a personal YouTube playlist, spanning 7 genre tags: hip-hop, indie rock, j-pop, k-pop, electronic, soul/folk, bedroom pop. Catalog metadata in `data/catalog.csv`. Audio is **not** committed; only the derived embeddings (`data/embeddings.npy`, 73KB) and fingerprints (`data/fingerprints/*.png`, ~30KB each) ship with the repo.

The CLAP comparison embeddings (`data/embeddings_clap.npy`) are also included so the eval harness can re-run the comparison.

---

## 5. Strengths

- **Cross-genre discovery actually works.** Daft Punk's nearest neighbor is Aphex Twin (symmetric, both electronic). Labi Siffre's 1972 acoustic ballad pairs with a 2001 IDM track on shared atmosphere. NewJeans (k-pop) clusters tightly with j-pop tracks even though they're tagged differently. These are the matches the system was built to make.
- **Self-validating.** The eval harness measures three real properties (spread, artist recall, cross-genre coverage), and these measurements are what surfaced CLAP's failure mode during development.
- **Honest about uncertainty.** The "because…" string includes the actual cosine score and explicitly flags BPM mismatches between seed and recommendation rather than smoothing them over.
- **Clone-and-run.** A stranger can run the demo in 30 seconds with no audio files and no model download, because the heavy work is done offline and the artifacts are committed.

---

## 6. Limitations and Bias

the catalog has 24 songs because that's a curation choice, but if i were to actually grow this to thousands or tens of thousands of songs, two things would break. first, the embed step is slow. 24 songs took about a minute. 24000 would take most of a day. thats a real bottleneck if you want this to be something people add to. second, with that many songs the cross-genre matches start to mean less. theres always going to be some random pair that scores high just by coincidence, and the more songs you have the more of those false matches you find.

the other thing is that MERT is a black box. we don't actually know what its matching on. it could be picking up on rhythm, or production style, or some feature we cant even name. the recommendations look right to us but theres no way to verify that the model is matching on things we'd say matter and not on things we'd say dont. you kinda have to just trust the rankings, or run the eval harness and hope the metrics catch the failure modes.

a few smaller ones worth flagging:

- the catalog is one person's taste — 24 songs heavy on hip-hop and electronic, very little outside western pop. cross-genre matches will look different on different catalogs.
- BPM detection fails on free-tempo music (IDM, ballads). BPM only shows up in explanations, not in ranking, so this isnt critical but its a known noisy signal.
- the system always returns 5 results, even when nothing in the catalog is actually close. for 24 songs thats fine. at scale you'd want a confidence threshold.

---

## 7. Evaluation

`eval/harness.py` computes three metrics across two embedding methods (MERT deployed, CLAP compared):

| Method | Spread | Artist recall@5 | Cross-genre@5 |
|---|---|---|---|
| MERT (deployed) | 0.13 | **0.20** | 3.38 |
| CLAP | 0.40 | **0.00** | 3.33 |

**Spread** = std of off-diagonal cosines (higher = more discriminative).
**Artist recall@5** = fraction of same-artist tracks recovered in top-5 (sanity check).
**Cross-genre@5** = average distinct foreign genres in top-5 (the headline behavior).

the most surprising thing was that spread looked like the right metric but it was artist-recall that actually caught CLAP. spread is the obvious "are these vectors discriminative" measure, but CLAP scored higher on it than MERT and the rankings were total nonsense. artist-recall ("do mitski's two tracks find each other") was the metric that actually mattered. wasnt expecting that.

14 unit tests pass (`pytest`) covering catalog loading, embedding shape, mean-centering, query lookup edge cases, top-k correctness, and explanation generation.

---

## 8. Misuse Considerations

the realistic misuse is making it too good. if you build a recommender that's really good at finding songs that hit the same vibe as something you already love, people can spend hours just chasing variations of one feeling. its the same rabbit hole problem spotify and tiktok have. the cross-genre design partially helps because it pulls you out of one genre tag, but it doesnt prevent the deeper "im stuck in a vibe" problem.

---

## 9. AI Collaboration Notes

i worked through this project with an AI assistant. two moments stood out.

**helpful.** when our first embedding model (CLAP) returned cosine scores that were basically all ≈0.96, the assistant flagged it as a known failure mode for contrastive embeddings and suggested mean-centering — subtract the global average from each vector and renormalize. that one-line fix turned the rankings from useless into something coherent, and is the same fix that makes MERT actually work in this project.

**wrong.** before we tested anything, the AI confidently said CLAP was the right tool for the job. it sounded right — music-themed name, well-known model, purpose-built. but CLAP was actually trained for audio-to-text matching, not for comparing two pieces of music against each other. running the eval harness made it obvious the recommendations were nonsense. the way to catch it wasnt to argue with the model, it was to actually run the numbers.

the real lesson is that AI tools are great for speeding up iteration but their suggestions need to be tested empirically. they pattern-match on the obvious answer and dont always know whether that answer is the right one for your specific case.

---

## 10. Future Work

- **Hybrid scoring.** A weighted ensemble of MERT and MFCC similarity captured both Daft Punk↔Aphex Twin (MERT win) and Mitski↔Joji (MFCC win). Worth exploring.
- **Text-query mode.** MERT doesn't natively support text queries; CLAP does (despite its other failures). A separate text-query path using CLAP for retrieval and MERT for re-ranking could give natural-language search.
- **Larger catalog.** 24 songs is enough to demo the principle. 200-500 would let the cross-genre matches be more frequent and the artist-recall metric more stable.
- **Confidence threshold.** Return "no good matches" instead of forcing 5 results when the top cosine is below a learned threshold.
