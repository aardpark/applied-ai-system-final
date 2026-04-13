# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**Antoine**

---

## 2. Intended Use

Based on the genres you enjoy, Antoine figures out 5 songs to recommend to you that fit. You describe your taste across 6 categories — favorite genre, favorite mood, target energy, target valence, target danceability, and whether you prefer acoustic — and a sophisticated matching algorithm ensures a good fit.

---

## 3. How the Model Works

Songs have multiple attributes. When the attributes match what you asked for, the song earns a point. Songs with similar attribute points are in similar genres. Antoine returns the 5 closest songs for the attributes you gave.

---

## 4. Data

20 songs total. 10 came with the starter, I added 10 more to get more genres and moods in there (hip hop, edm, metal, folk, country, classical, reggae, rnb, indie). Each song has genre, mood, energy, tempo, valence, danceability, acousticness. All the song titles are made up so none of this is real listener data.

---

## 5. Strengths

Antoine works pretty well when you give it a clean profile. Asking for pop happy high energy got me Sunrise City at the top which makes sense. The scores also come with reasons so you can see exactly why a song ranked where it did. And when I gave it a weird profile that had no good match it returned low scores instead of pretending to be confident which is probably the right move.

---

## 6. Limitations and Bias

Genre is probably given too much weight. Genre overwhelmingly overpowers attributes so the algorithm picks genre over everything else. Also, emotions that are similar are treated uniquely, so there's a missed opportunity there.

---

## 7. Evaluation

tested it with four profiles, three normal and one weird on purpose (edm + sad mood that doesn't even exist in the catalog). checked if the top pick made sense for each. also did an experiment where i doubled energy weight and halved genre to see if the ranking would flip. it didnt. genre still won. so the dominance isnt just the number its structural.

---

## 8. Future Work

would normalize the total score so perfect match means the same thing no matter how i tune the weights. would also do something smarter with mood so happy and euphoric don't read as totally unrelated. and a diversity penalty would be good, nothing stops the top 5 from all being the same artist right now.

---

## 9. Personal Reflection

biggest thing i learned is that a recommender is basically just a scoring function plus a sort. everything else is plumbing. the "intelligence" isnt in the algorithm its in how you decide what gets points. the weight shift experiment surprised me, i assumed doubling energy would flip the top pick and it didnt, which made me realize the actual question isnt "how much is this feature worth" its "how does this feature stack against the ceiling of the others." also now when spotify recommends me something i think about what invisible weights decided that.
