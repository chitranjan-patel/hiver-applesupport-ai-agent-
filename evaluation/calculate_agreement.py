import pandas as pd
import numpy as np
import os

def calculate_agreement():
    file_path = 'evaluation/human_judge_sample.csv'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    df = pd.read_csv(file_path)
    
    dimensions = ['correctness', 'groundedness']
    
    print("--- Human vs LLM Judge Agreement Analysis ---")
    
    for dim in dimensions:
        human_col = f'human_{dim}'
        llm_col = f'llm_{dim}'
        
        if human_col not in df.columns or llm_col not in df.columns:
            continue
            
        # Filter out rows where human hasn't scored yet
        valid_df = df.replace(r'^\s*$', np.nan, regex=True).dropna(subset=[human_col, llm_col])
        
        total_rows = len(df)
        valid_rows = len(valid_df)
        missing_rows = total_rows - valid_rows
        
        print(f"\nDimension: {dim.upper()}")
        print(f"Total rows: {total_rows} | Valid scored rows: {valid_rows} | Missing human scores: {missing_rows}")
        
        if valid_rows == 0:
            print(f"Status: Waiting for human review on '{human_col}'.")
            continue
            
        # Ensure numeric
        human_scores = pd.to_numeric(valid_df[human_col], errors='coerce')
        llm_scores = pd.to_numeric(valid_df[llm_col], errors='coerce')
        
        # Calculate Absolute Difference
        abs_diff = (human_scores - llm_scores).abs()
        avg_diff = abs_diff.mean()
        
        # Agreement Percentage (Tolerance of +/- 1)
        agreement_mask = abs_diff <= 1
        agreement_pct = agreement_mask.mean() * 100
        
        # Exact Match Percentage
        exact_match_pct = (abs_diff == 0).mean() * 100
        
        print(f"Average Absolute Difference: {avg_diff:.2f} (Scale 1-5)")
        print(f"Agreement (Tolerance +/- 1): {agreement_pct:.1f}%")
        print(f"Exact Match: {exact_match_pct:.1f}%")
        
        if valid_rows > 5:
            corr = human_scores.corr(llm_scores)
            print(f"Pearson Correlation: {corr:.2f}")
            
if __name__ == "__main__":
    calculate_agreement()
