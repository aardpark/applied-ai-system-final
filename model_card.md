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

- **Catalog is one person's taste.** 24 songs heavy on hip-hop and electronic, light on country, jazz, classical, ambient. Cross-genre matches will look different on different catalogs. With more songs the recommender's biases would surface in different places.
- **MERT's training data is Western-pop-heavy.** Embeddings are likely to be richer for pop/rock/electronic than for non-Western genres. This catalog has very little non-Western music outside of j-pop/k-pop, so the bias is masked rather than absent.
- **BPM detection fails on free-tempo music.** librosa's beat tracker handles 4/4 dance music well. For IDM (Aphex Twin) and acoustic ballads it can be off by a factor of 2. BPM appears in explanations only; it doesn't steer ranking.
- **No "no good matches" outcome.** The system always returns 5 results. With a 24-track catalog that's fine; at 24,000 you'd want a confidence threshold.
- **Pretrained-model opacity.** Unlike v1's hand-tuned weights, v2's notion of similarity is opaque. The eval harness is the only check on whether the model is doing something real or hallucinating coherence.

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

CLAP looked plausible by spread but artist recall exposed it as noise. MERT has a smaller spread but every individual ranking is coherent.

In addition: 14 unit tests pass (`pytest`), covering catalog loading, embedding shape, mean-centering, query lookup edge cases, top-k correctness, and explanation generation.

---

## 8. Misuse Considerations

- **Filter bubble.** Like any similarity-based recommender, Antoine will steer a user toward what they already like. The cross-genre design partially counters this but doesn't eliminate it.
- **Audio-source copyright.** The system embeds from publicly accessible YouTube URLs and never stores audio. Anyone redeploying this with non-public sources should think about whether the resulting embeddings are derivative works.
- **Catalog injection.** If extended into a multi-user system where users can add tracks to a shared catalog, the embedding step downloads arbitrary URLs — that's a sandboxing concern. The current single-user design avoids this entirely.

---

## 9. AI Collaboration Notes

I worked through this project with an AI assistant. Two specific moments are worth recording.

**One time the AI was helpful.** When CLAP's embeddings turned out to be uniformly ≈0.96 cosine-similar (the "narrow cone" failure), the assistant correctly diagnosed it as anisotropy and proposed mean-centering as the standard fix. That took the rankings from random-looking to genuinely coherent, and it's the same one-line fix that makes MERT work too. Without that suggestion I'd have either accepted the bad rankings or burned hours on the wrong fix.

**One time the AI was wrong.** Before testing, the assistant confidently recommended CLAP as "the right tool for music-music similarity, purpose-built for this" — it pattern-matched on "music + audio + similarity" and skipped that CLAP is specifically a *contrastive audio-text* model, not a music-music similarity model. The recommendation was plausible-sounding and well-formatted and turned out to make the entire pipeline produce nonsense. The fix wasn't to argue with the model; it was to write the eval harness, run the numbers, and switch to MERT when the numbers didn't lie. Lesson: AI judgment is fast at suggesting next steps and bad at predicting which suggestion will actually work. Validate empirically.

---

## 10. Future Work

- **Hybrid scoring.** A weighted ensemble of MERT and MFCC similarity captured both Daft Punk↔Aphex Twin (MERT win) and Mitski↔Joji (MFCC win). Worth exploring.
- **Text-query mode.** MERT doesn't natively support text queries; CLAP does (despite its other failures). A separate text-query path using CLAP for retrieval and MERT for re-ranking could give natural-language search.
- **Larger catalog.** 24 songs is enough to demo the principle. 200-500 would let the cross-genre matches be more frequent and the artist-recall metric more stable.
- **Confidence threshold.** Return "no good matches" instead of forcing 5 results when the top cosine is below a learned threshold.
