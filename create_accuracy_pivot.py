import pandas as pd

# Load the results
results_path = 'output_data/20251227_232747_gemini-2.5-flash_yelp_retrieval_random_random_contriever_s42_n5/retrieval_comparison_results.csv'
df = pd.read_csv(results_path)

# Create strategy column (same as in the script)
df['strategy'] = df.apply(
    lambda row: f"{row['retriever']} {row['cf_strategy']} ({row['pool_type']})" 
    if row['cf_strategy'] != '-' 
    else f"{row['retriever']} ({row['pool_type']})",
    axis=1
)

# Create accuracy pivot table
pivot_df = df.pivot_table(
    values='accuracy',
    index='strategy',
    columns='budget_used',
    aggfunc='first'
)

# Sort strategies in desired order
strategy_order = []
for retriever in ['static', 'bm25', 'contriever', 'bge_large']:
    for cf_strat in ['-', 'mixed', 'factual_anchored']:
        for pool in ['full', 'factuals_only']:
            if cf_strat == '-':
                strategy_order.append(f"{retriever} ({pool})")
            else:
                strategy_order.append(f"{retriever} {cf_strat} ({pool})")

# Reindex to match order
existing_strategies = [s for s in strategy_order if s in pivot_df.index]
pivot_df = pivot_df.reindex(existing_strategies)

# Sort columns (budgets) numerically
pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

# Save to CSV
output_path = 'output_data/20251227_232747_gemini-2.5-flash_yelp_retrieval_random_random_contriever_s42_n5/retrieval_comparison_pivot_accuracy.csv'
pivot_df.to_csv(output_path)
print(f'✅ Accuracy pivot table saved to: {output_path}')
print(f'\nAccuracy Pivot Table:')
print(pivot_df.round(4).to_string())


