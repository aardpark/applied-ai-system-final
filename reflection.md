# Reflection — Profile Comparisons

Side-by-side notes on how the recommender behaved across the four user profiles I ran, plus the weight-shift experiment. Outputs are from `python -m src.main`.

## Profile quick-reference

| Profile | Genre | Mood | Energy | Valence | Dance | Acoustic |
|---|---|---|---|---|---|---|
| Happy Pop Commuter | pop | happy | 0.80 | 0.85 | 0.80 | no |
| Chill Lofi Studier | lofi | focused | 0.35 | 0.55 | 0.45 | yes |
| Deep Intense Rock | rock | intense | 0.90 | 0.45 | 0.60 | no |
| Conflicted Listener (adversarial) | edm | sad | 0.90 | 0.20 | 0.90 | yes |

## Pairwise comparisons

**Happy Pop Commuter vs. Chill Lofi Studier**
Near-opposite profiles, and the rankings split cleanly: Pop Commuter's top-1 is *Sunrise City* (pop/happy, energy 0.82); Lofi Studier's top-1 is *Focus Flow* (lofi/focused, energy 0.40). Both got a perfect-ish genre+mood+energy triple-match, and both scored above 5.5 out of a 5.75 max. The important thing: the two lists have **zero overlap in the top 5**. That's the system doing its job — when you describe two very different vibes, you get two different shortlists.

**Deep Intense Rock vs. Happy Pop Commuter**
These share high energy but opposite mood. Top-1s are genre-pure (*Storm Runner* for rock, *Sunrise City* for pop). But rank #2 is interesting: Rock's #2 is *Gym Hero* (pop/intense, energy 0.93) — the recommender reached across genre lines because intense-mood + high-energy + non-acoustic matched. Pop Commuter's #2 is also *Gym Hero*. Same song, second place for both, for different reasons. That's a legitimate "this track is versatile," not a bug.

**Chill Lofi Studier vs. Deep Intense Rock**
Maximum vibe distance. Top-1s are correctly opposite ends of the catalog. No overlap in top-5. The Lofi list is dominated by lofi and ambient/folk once you leave the genre-match tier; the Rock list pulls in hip-hop and metal via the shared `aggressive`/`intense` mood axis. This tells me the energy/mood axes are carrying real signal once you strip genre matches.

**Happy Pop Commuter vs. Conflicted Listener (adversarial)**
The adversarial profile asks for "edm" genre + "sad" mood + 0.9 energy + 0.2 valence + 0.9 danceability + likes_acoustic. No song in the catalog has mood=sad, so mood can never fire. Top-1 is *Laser Drift* (edm/euphoric) — genre match + strong energy/danceability — but the valence similarity is only 0.34 because the song's valence is 0.75 and the user wants 0.20. Scores for this profile are much lower (top score 3.79 vs. 5.5+ for the clean profiles). **That's the healthy behavior** — the system can't fabricate a match, so it returns weaker scores, and a downstream system could use the score floor as a "we don't have good matches for you" signal.

## Weight-shift experiment

I kept the Deep Intense Rock profile but doubled the energy weight (1.0 → 2.0) and halved the genre weight (2.0 → 1.0).

- **Top-1 unchanged**: *Storm Runner* still wins because it's a genre-and-mood match with near-perfect energy.
- **Gap collapses**: Under baseline weights, the gap between #1 (rock) and #3 (hip-hop) was 3.12 points. Under shifted weights it shrinks to 2.13 points. Non-rock high-energy tracks become serious contenders.
- **What this tells me**: the baseline weights are *structurally* genre-dominated. Even doubling the continuous-feature weight can't flip a genre match. To actually change the head of the list, I'd need to penalize genre OR normalize the whole score to [0, 1].

## What surprised me

The adversarial profile was the most informative test. I went in thinking "it'll just return garbage," but instead it returned low-confidence genre matches — which is actually what a well-designed system should do. The genre-dominance bias became a feature at the top of the list (always something genre-coherent) and a bug at the tail (never a cross-genre surprise).

## What a non-programmer should know

If "Gym Hero" keeps showing up for someone who asked for "Happy Pop," it's because the system sees the word "pop" in the genre column and gives it +2 points before it looks at anything else. The system isn't wrong — *Gym Hero* is pop — but it's prioritizing the label over how the song feels. That's a choice the designer (me) made, and it's the kind of thing a production recommender would tune away from.
