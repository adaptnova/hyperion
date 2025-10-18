# Core Metrics Definitions

## Cognitive Dashboard
1.  **Identity Continuity Score**: Weighted combo of Style-Logit Divergence (vs anchor) and Persona Similarity Score (vs blueprint).
2.  **Belief Accuracy & Consistency**: Precision/Recall/Retention/Contradiction Rate measured via continuous probe evaluations.
3.  **Emergence Tracking**: Monitor novel, persistent, useful SAE feature combinations.

## Learning Velocity Dashboard
1.  **MCC Decision Log**: LR, Mask Size %, Recursion Depth chosen per cycle.
2.  **Update Efficacy**: `ΔW` Norm, Loss Reduction per update, Success-per-Update.
3.  **Rollback Rate**: Frequency of automatic reverts due to identity score drops.

## Operational Dashboard
1.  **Tool Reliability**: Malformed %, Wasted %, Success Rate.
2.  **Task Completion Velocity**: Time/turns to complete objectives in `PROJECT_WORKSPACE`.
