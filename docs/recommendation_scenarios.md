# Recommendation scenarios

## 1. Guest

Guests are not tracked. No guest profile, history, recent-movie list, user ID, or session recommendation model exists. The API-facing baseline reads precomputed `ALL` or genre weighted-rating rankings using `get_guest_recommendations(genre, top_k)`.

## 2. First-login onboarding

The input contains `selected_movie_ids`, `selected_genres`, and `top_k`. The baseline validates canonical IDs and normalized genres, ignores invalid values with warnings, combines selected-movie TF-IDF profiles with genre-token preferences, excludes selected movies, and returns deterministic unique ranked results with scores.

It does not read guest history or `guest_recent_movie_ids`. Empty or unusable input falls back to the guest global top-rated ranking.

## 3. Returning user

Returning-user data uses available historical interactions. This dataset contains explicit ratings only, so every output interaction has `interaction_type = rating` and preserves the observed rating and UTC timestamp. No click, watch, like, completion, or session event is invented.

Chronological validation/test holdouts are created only for users with adequate training history. Sparse users remain train-only.

