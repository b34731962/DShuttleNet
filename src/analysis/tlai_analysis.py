import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 

def analyze_tlai(df_analysis, shots_to_end_limit=2, selected_type=['Failed Return', 'Opp Grounded within Boundaries', 'Net Fault', 'Out of Bounds'], threat_shot_types = ['Smash', 'Drop']):

    def get_absolute_team(row):
        pid = row['player']
        if pd.isna(pid): return np.nan
        if pid == row['A'] or pid == row['B']: return 0.0
        if pid == row['C'] or pid == row['D']: return 1.0
        return np.nan

    df_analysis['player_team_abs'] = df_analysis.apply(get_absolute_team, axis=1)

    df_analysis['player_team_set_status'] = np.where(
        df_analysis['player_team_abs'] == df_analysis['set_win'],
        0.0,
        1.0
    )

    df_analysis['did_opponent_team_win'] = df_analysis['player_team_set_status'] != df_analysis['score_team']

    df_analysis['defender_TLAI'] = np.where(
        df_analysis['player_team_abs'] == 0.0,
        df_analysis['match_Loser_TLAI'],
        df_analysis['match_Winner_TLAI']
    )
    
    df_analysis['defender_formation'] = np.where(
        df_analysis['player_team_abs'] == 0.0,
        df_analysis['match_Loser_formation'],
        df_analysis['match_Winner_formation']
    )
    
    hitter_won = (df_analysis['player_team_set_status'] == df_analysis['score_team'])

    shots_to_end = df_analysis['shot_count'] - df_analysis['shot_num']
    is_immediate_end = shots_to_end.isin(range(0, shots_to_end_limit + 1))
    is_lose_type = df_analysis['lose_reason'].isin(selected_type)
    
    df_analysis['is_defender_failure'] = hitter_won & is_immediate_end & is_lose_type
    
    df_analysis['is_threat_shot'] = df_analysis['ball_type'].astype(str).str.strip().str.lower().isin(
        [s.lower() for s in threat_shot_types]
    )

    df_def = df_analysis[
        (df_analysis['defender_formation'] == 'Defense') & 
        (df_analysis['defender_TLAI'].notna()) &
        (df_analysis['is_threat_shot'] == True)
    ].copy()

    if len(df_def) == 0:
        return

    print(f" There are {len(df_def)} valid records for defense analysis.")

    print("\n[Logistic Regression Analysis for Nonlinear TLAI Effect]")
    df_def['TLAI_scaled_10x'] = df_def['defender_TLAI'] * 10 
    df_def['TLAI_squared'] = df_def['TLAI_scaled_10x'] ** 2
    
    X = df_def[['TLAI_scaled_10x', 'TLAI_squared']]
    X = sm.add_constant(X) 
    Y = df_def['is_defender_failure'].astype(int)

    log_reg_nonlinear = sm.Logit(Y, X).fit(disp=0) 
    
    print(log_reg_nonlinear.summary().tables[1])
    
    p_val_sq = log_reg_nonlinear.pvalues['TLAI_squared']
    coef_sq = log_reg_nonlinear.params['TLAI_squared']
    
    print(f"\n>> Coefficient for quadratic term (TLAI_squared) : {coef_sq:.4f}")
    print(f">> P-value: {p_val_sq:.5f}")

    bins = np.arange(0, 1.05, 0.05)
    labels = [f"{b:.2f}" for b in bins[:-1]]
    
    df_def['TLAI_bin'] = pd.cut(df_def['defender_TLAI'], bins=bins, labels=labels, include_lowest=True)

    bin_stats = df_def.groupby('TLAI_bin', observed=False).agg(
        Failure_Rate=('is_defender_failure', lambda x: x.mean() * 100),
        Sample_Count=('is_defender_failure', 'count')
    ).reset_index()

    min_samples = 100
    valid_bins = bin_stats[bin_stats['Sample_Count'] >= min_samples].copy()

    print(f"\n[TLAI threshold analysis (only showing intervals with sample size >= {min_samples})]")
    print(valid_bins)

    if valid_bins.empty:
        return bin_stats

    plt.figure(figsize=(8, 6))

    ax = sns.lineplot(
        x='TLAI_bin', y='Failure_Rate',
        data=valid_bins,
        marker='o', color='#EA4335', linewidth=3, markersize=9
    )

    for i, row in valid_bins.iterrows():
        x_pos = list(valid_bins['TLAI_bin']).index(row['TLAI_bin'])
        plt.text(
            x=x_pos, y=row['Failure_Rate'] + 0.5,
            s=f"{row['Failure_Rate']:.1f}%\n(n={int(row['Sample_Count'])})",
            ha='center', va='bottom', fontsize=11, weight='bold', color='#333333'
        )

    avg_failure_rate = df_def['is_defender_failure'].mean() * 100
    plt.axhline(y=avg_failure_rate, color='gray', linestyle='--', alpha=0.7, label=f'Avg ({avg_failure_rate:.1f}%)')

    plt.xlabel('TLAI (0.0 = Balance, 1.0 = Imbalance)', fontsize=14, weight='bold')
    plt.ylabel('Immediate Defense Failure Rate (%)', fontsize=14, weight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.legend(loc='upper left', fontsize=14)
    
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    
    min_y = valid_bins['Failure_Rate'].min()
    max_y = valid_bins['Failure_Rate'].max()
    if pd.isna(min_y) or pd.isna(max_y):
        plt.ylim(0, 100)
    else:
        plt.ylim(max(0, min_y - 2), max_y + 4)
    
    plt.subplots_adjust(left=0.12, right=0.88, top=0.90, bottom=0.15)
    
    os.makedirs('./img/Coordination Analysis', exist_ok=True)
    plt.savefig('./img/Coordination Analysis/tlai_threshold_analysis.pdf', dpi=400) 
    plt.show()

    return bin_stats