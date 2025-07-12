# Scoring Weights Configuration Guide

This document explains the configurable scoring weights for the Leadpoet Intent Model v1.1 and how to tune them for different use cases.

## Scoring Weights Overview

The scoring system uses weighted combinations to calculate fit scores and final scores. All weights are configurable through environment variables, allowing easy tuning without code changes.

## Fit Score Weights

The fit score is calculated as a weighted average of three components:

### `FIT_SCORE_INDUSTRY_WEIGHT`
Weight for industry match in fit score calculation.

**Default:** `0.4` (40%)
**Range:** `0.0` to `1.0`
**Description:** How much importance to place on industry matching

### `FIT_SCORE_SIZE_WEIGHT`
Weight for company size match in fit score calculation.

**Default:** `0.3` (30%)
**Range:** `0.0` to `1.0`
**Description:** How much importance to place on company size matching

### `FIT_SCORE_REGION_WEIGHT`
Weight for region match in fit score calculation.

**Default:** `0.3` (30%)
**Range:** `0.0` to `1.0`
**Description:** How much importance to place on geographic region matching

**Note:** The three fit score weights should sum to 1.0 for proper normalization.

## Final Score Weights

The final score combines fit score and intent score:

### `FINAL_SCORE_FIT_WEIGHT`
Weight for fit score in final score calculation.

**Default:** `0.6` (60%)
**Range:** `0.0` to `1.0`
**Description:** How much importance to place on fit vs intent

### `FINAL_SCORE_INTENT_WEIGHT`
Weight for intent score in final score calculation.

**Default:** `0.4` (40%)
**Range:** `0.0` to `1.0`
**Description:** How much importance to place on intent vs fit

**Note:** The two final score weights should sum to 1.0 for proper normalization.

## Boost Configuration

### `CHURN_BOOST_VALUE`
Boost value for leads with churn indicators.

**Default:** `20.0` (+20 points)
**Range:** `0.0` to `100.0`
**Description:** Additional points added to intent score for churning companies

### `JOB_POSTING_BOOST_VALUE`
Boost value for leads with job posting activity.

**Default:** `15.0` (+15 points)
**Range:** `0.0` to `100.0`
**Description:** Additional points added to intent score for companies posting jobs

## Environment-Specific Configuration

### Development Environment
```bash
# .env.development
FIT_SCORE_INDUSTRY_WEIGHT=0.4
FIT_SCORE_SIZE_WEIGHT=0.3
FIT_SCORE_REGION_WEIGHT=0.3
FINAL_SCORE_FIT_WEIGHT=0.6
FINAL_SCORE_INTENT_WEIGHT=0.4
CHURN_BOOST_VALUE=20.0
JOB_POSTING_BOOST_VALUE=15.0
```

### Staging Environment
```bash
# .env.staging
FIT_SCORE_INDUSTRY_WEIGHT=0.5
FIT_SCORE_SIZE_WEIGHT=0.3
FIT_SCORE_REGION_WEIGHT=0.2
FINAL_SCORE_FIT_WEIGHT=0.7
FINAL_SCORE_INTENT_WEIGHT=0.3
CHURN_BOOST_VALUE=25.0
JOB_POSTING_BOOST_VALUE=20.0
```

### Production Environment
```bash
# .env.production
FIT_SCORE_INDUSTRY_WEIGHT=0.4
FIT_SCORE_SIZE_WEIGHT=0.3
FIT_SCORE_REGION_WEIGHT=0.3
FINAL_SCORE_FIT_WEIGHT=0.6
FINAL_SCORE_INTENT_WEIGHT=0.4
CHURN_BOOST_VALUE=20.0
JOB_POSTING_BOOST_VALUE=15.0
```

## Use Case Examples

### High-Intent Focus
For scenarios where you want to prioritize companies showing strong intent signals:

```bash
FINAL_SCORE_FIT_WEIGHT=0.3
FINAL_SCORE_INTENT_WEIGHT=0.7
CHURN_BOOST_VALUE=30.0
JOB_POSTING_BOOST_VALUE=25.0
```

### Strict Fit Focus
For scenarios where you want to prioritize perfect ICP matches:

```bash
FIT_SCORE_INDUSTRY_WEIGHT=0.5
FIT_SCORE_SIZE_WEIGHT=0.3
FIT_SCORE_REGION_WEIGHT=0.2
FINAL_SCORE_FIT_WEIGHT=0.8
FINAL_SCORE_INTENT_WEIGHT=0.2
```

### Industry-Specific Tuning
For industries where size matters more than region:

```bash
FIT_SCORE_INDUSTRY_WEIGHT=0.4
FIT_SCORE_SIZE_WEIGHT=0.5
FIT_SCORE_REGION_WEIGHT=0.1
```

## Scoring Formula

### Fit Score Calculation
```
fit_score = (industry_match × FIT_SCORE_INDUSTRY_WEIGHT) +
           (size_match × FIT_SCORE_SIZE_WEIGHT) +
           (region_match × FIT_SCORE_REGION_WEIGHT)
```

### Final Score Calculation
```
final_score = (fit_score × FINAL_SCORE_FIT_WEIGHT) +
             (intent_score × FINAL_SCORE_INTENT_WEIGHT)
```

### Boost Application
```
boosted_intent_score = base_intent_score + 
                      (churn_boost × CHURN_BOOST_VALUE / 100) +
                      (job_posting_boost × JOB_POSTING_BOOST_VALUE / 100)
```

## Monitoring and Tuning

### Key Metrics to Monitor
1. **Fit Score Distribution**: Check if fit scores are too concentrated
2. **Intent Score Distribution**: Monitor intent score spread
3. **Final Score Quality**: Track conversion rates by score ranges
4. **Boost Impact**: Measure how boosts affect lead quality

### A/B Testing Weights
To test different weight configurations:

1. **Create Test Environment**: Set up staging with different weights
2. **Run Parallel Tests**: Compare results with different configurations
3. **Measure Outcomes**: Track conversion rates, engagement, etc.
4. **Gradual Rollout**: Implement winning configuration gradually

### Validation Rules
The system validates weight configurations:

- Fit score weights should sum to 1.0
- Final score weights should sum to 1.0
- All weights should be between 0.0 and 1.0
- Boost values should be positive

## Troubleshooting

### Common Issues

#### Weights Don't Sum to 1.0
```
ValueError: Fit score weights must sum to 1.0
```

**Solution:** Ensure `FIT_SCORE_INDUSTRY_WEIGHT + FIT_SCORE_SIZE_WEIGHT + FIT_SCORE_REGION_WEIGHT = 1.0`

#### Final Score Weights Don't Sum to 1.0
```
ValueError: Final score weights must sum to 1.0
```

**Solution:** Ensure `FINAL_SCORE_FIT_WEIGHT + FINAL_SCORE_INTENT_WEIGHT = 1.0`

#### Negative Weights
```
ValueError: Weights must be between 0.0 and 1.0
```

**Solution:** Check that all weight values are positive and ≤ 1.0

### Performance Impact
- **Higher Intent Weights**: May increase LLM usage and latency
- **Complex Boost Logic**: May impact Redis cache performance
- **Weight Validation**: Minimal impact on scoring performance

## Best Practices

### 1. **Start Conservative**
- Begin with default weights
- Make small incremental changes
- Monitor impact before major changes

### 2. **Data-Driven Decisions**
- Use A/B testing for weight changes
- Monitor conversion rates by score ranges
- Track lead quality metrics

### 3. **Environment Consistency**
- Use similar weights across environments
- Document weight changes
- Version control your configurations

### 4. **Regular Review**
- Review weights monthly
- Adjust based on performance data
- Consider seasonal variations

### 5. **Documentation**
- Document weight changes and rationale
- Track performance impact
- Share learnings across teams 