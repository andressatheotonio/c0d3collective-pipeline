"""
Maps native platform difficulty values to the Codeforces scale (800–3500).

Codeforces already uses its own scale; AtCoder and Kattis use different systems:
- AtCoder (Kenkoooo IRT difficulty): roughly -2000 to 4000+, mean ~1000 for ABC-C
- Kattis: 1.0–10.0 continuous scale

Linear fits derived from comparative analyses in the literature.
Problems outside the CF scale range are clamped to [800, 3500].
"""

CF_MIN, CF_MAX = 800, 3500


def _clamp(value: float) -> float:
    return max(CF_MIN, min(CF_MAX, value))


class RatingScaler:
    def scale(self, platform: str, native_rating: float) -> float | None:
        """Return cf_scaled_difficulty for a native platform rating."""
        if platform == "codeforces":
            return _clamp(native_rating)
        elif platform == "atcoder":
            return self._atcoder(native_rating)
        elif platform == "kattis":
            return self._kattis(native_rating)
        return None

    def _atcoder(self, irt_difficulty: float) -> float:
        # AtCoder IRT ~0 → CF ~1200; IRT ~2000 → CF ~2600 (empirical linear fit)
        cf = 1200 + (irt_difficulty / 2000) * 1400
        return _clamp(cf)

    def _kattis(self, kattis_score: float) -> float:
        # Kattis 1.0 → CF 800; Kattis 10.0 → CF 3500 (linear)
        cf = 800 + (kattis_score - 1.0) / 9.0 * (CF_MAX - CF_MIN)
        return _clamp(cf)
