# Fix: Model Biased Toward "Lie" — Improve Pre-trained Model

The current synthetic pre-trained model almost always predicts "Lie" on real recordings. Fix this comprehensively.

## Root Causes to Fix

### 1. Synthetic data doesn't match real audio feature ranges
The synthetic data uses hardcoded Gaussian distributions that don't match what librosa actually extracts from real recordings. Real speech features (especially MFCCs, spectral features) have very different ranges.

**Fix in `detector/model.py` → `_sample_features()`:**
- The MFCC ranges are way off. Real MFCCs from librosa have: mfcc_1 ~= -200 to -100, mfcc_2+ ~= -50 to 50. Currently using means near 0.
- Spectral centroid: real values from librosa are typically 500-3000 Hz (current is ok)
- RMS: real values typically 0.01-0.3 (check current)
- f0_mean: real speech is typically 85-255 Hz (male ~85-180, female ~165-255)
- HNR: the log-ratio calculation in audio_features.py produces values roughly in range -1 to 2, not -0.5 to 0.5
- speech_rate: real values are typically 1-6 syllables/sec
- Make distributions wider with MORE OVERLAP between truth/lie — deception differences are subtle, not dramatic

### 2. Prediction thresholds too aggressive
**Fix in `detector/model.py` → `predict()`:**
- Current: truth if >70%, lie if lie_prob >70%
- Change to: truth if >60%, lie if lie_prob >60%, otherwise "Uncertain"
- Add calibration: when model is synthetic/pretrained, bias toward "Uncertain" rather than confident predictions
- The default (no model) should lean toward 50/50 (uncertain), NOT toward lie

### 3. Feature mismatch between synthetic and real
**Fix in `detector/model.py` → `create_pretrained_model()`:**
- After generating synthetic data, apply StandardScaler stats to understand the expected ranges
- When predicting on real data, if feature values are far outside synthetic training range, reduce confidence

### 4. Better indicator analysis
**Fix in `detector/model.py` → `_analyze_indicators()`:**
- Use relative comparisons (vs baseline) not absolute thresholds
- Label indicators as "Slightly elevated", "Within normal range" rather than just "High"/"Low"

## Specific Code Changes Needed

### In `detector/model.py`:

1. **`_sample_features()`** — Fix ALL distributions to match real librosa output:
   - MFCC_1_mean: truth ~= -180±30, lie ~= -170±30 (subtle difference!)
   - MFCC_2_mean: truth ~= 30±15, lie ~= 35±15
   - Other MFCCs: similar small shifts, large variance
   - MFCC_*_std: truth ~= 15±5, lie ~= 18±5
   - f0_mean: widen range, 100-200 Hz
   - HNR: real range -1 to 2
   - All differences between truth/lie should be SUBTLE (research shows ~5-15% shift, not 50%+)
   - Increase noise/variance so the model learns subtle patterns, not dramatic ones

2. **`predict()`** — Better calibration:
   ```python
   # When synthetic model, apply confidence dampening
   if self.metadata.get("synthetic", False):
       # Dampen toward 0.5 (uncertain)
       truth_prob = 0.5 + (truth_prob - 0.5) * 0.6
       lie_prob = 1.0 - truth_prob
   
   # Lower thresholds for classification
   if truth_prob >= 0.58:
       prediction = "Truth"
   elif lie_prob >= 0.58:
       prediction = "Lie"
   else:
       prediction = "Uncertain"
   ```

3. **`_analyze_indicators()`** — Use ranges that match real data:
   - Pitch variability: low < 15, high > 45 (was 10/30)
   - Jitter: low < 0.005, high > 0.02 (was 0.003/0.015)
   - Shimmer: real range is wider
   - HNR: adjust to real range (-1 to 2)

4. **Increase `_SYNTHETIC_N_PER_CLASS`** to 1000 for better generalization

### In `detector/audio_features.py`:
- No changes needed, extraction is correct

### In `ui/components.py` and `ui/app.py`:
- No changes needed

## Testing
After making changes, verify by:
1. Delete existing model: `rm models/model_*`
2. The app should regenerate a better pretrained model
3. On neutral speech, it should predict closer to 50/50 or "Uncertain"

## Git
```bash
rm -f models/model_*.joblib
git add -A && git commit -m "fix: improve pretrained model calibration - reduce lie bias, realistic feature distributions"
git push
```

## IMPORTANT
- Read the existing code first, understand it fully
- Make TARGETED edits — don't rewrite entire files
- The key insight: subtle differences between truth/lie + confidence dampening for synthetic model
- When completely finished, run: openclaw system event --text "Done: Fixed lie bias - improved synthetic distributions and prediction calibration" --mode now
