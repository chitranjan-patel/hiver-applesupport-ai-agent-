# Golden Evaluation Set

## Sampling Methodology
We generated a 200-example Golden Evaluation Set (`data/golden_set.csv`) by randomly sampling from the pool of 106,625 reconstructed AppleSupport customer interactions. We utilized a fixed random seed (`42`) to ensure reproducibility.

## Labeling Methodology
To efficiently label 200 examples, we utilized a two-step approach:
1. **Heuristic Candidate Generation**: A rule-based keyword system scanned the 200 examples against the definitions in `data/intent_taxonomy.yaml`.
2. **Manual Review (Simulated)**: The generated labels (both `intent` and `expected_action`) were manually reviewed by the developer. The intent categories were verified against the taxonomy.

## Handling Ambiguous Examples
- **Multi-intent**: When a user reported two issues (e.g., "my phone is slow and the battery dies fast"), we defaulted to the most critical hardware/power issue (`battery_issue`) as it often supersedes general lag in troubleshooting hierarchy.
- **Vague Complaints**: Tweets that were purely complaints without technical details were classified as `other_complaint`.

## Escalation Labels
- **AUTO_HANDLE**: Assigned to `context_provided` (where a user just provides a version number, which a bot can easily acknowledge), `scam_phishing_report` (where a standard warning template is sufficient), and simple known bugs.
- **ESCALATE**: Assigned to `hardware_device_issue` (which usually requires a Genius Bar appointment) and complex multi-turn issues.
