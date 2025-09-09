import pandas as pd
from typing import List, Optional
from .zero_shot import zero_shot_scores_for_assay

def compare_scoring_methods(
    df: pd.DataFrame,
    model_name: str,
    device: Optional[str] = None,
    methods: Optional[List[str]] = None,
    progress: bool = True
) -> pd.DataFrame:

    if methods is None:
        methods = ["masked", "mutant_marginal", "wildtype_marginal", 
                  "pseudo_likelihood", "global_probability"]
    
    result_df = df.copy()
    
    for method in methods:
        print(f"\nRunning {method} scoring...")
        scored_df = zero_shot_scores_for_assay(
            df=df,
            model_name=model_name,
            device=device,
            progress=progress,
            scoring_method=method
        )
        result_df[f'{method}_score'] = scored_df['delta_log_prob']
    
    # Calculate correlation matrix if multiple methods used
    if len(methods) > 1:
        score_cols = [f'{m}_score' for m in methods]
        correlation_matrix = result_df[score_cols].corr()
        print("\nCorrelation between scoring methods:")
        print(correlation_matrix)
    
    return result_df