import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.utils.data_utils import df_filter_by_conditions
from scipy.stats import chi2_contingency 

def global_cce_analysis(df, match_id=None, 
                                set_num=None, 
                                start_rally_id=None, 
                                end_rally_id=None,
                                event_type=None,
                                selected_formations='Attack'):
    
    df_filtered, filter_title_suffix = df_filter_by_conditions(
        df,
        match_id=match_id,
        set_num=set_num,
        start_rally_id=start_rally_id,
        end_rally_id=end_rally_id,
        player_id=None, 
        team=None,
        event_type=event_type    
    )
        
    required_cols = [
        'player', 'A', 'B', 'C', 'D', 'shot_num', 'shot_count',
        'match_Winner_formation', 'match_Loser_formation', 
        'match_Winner_mid_cover', 'match_Loser_mid_cover',
        'score_team', 'match_id', 'set_id', 'rally_id', 'set_win'
    ]
    
    if not all(col in df_filtered.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_filtered.columns]
        print(f"error: required columns missing: {missing}。")
        return
    
    df_analysis = df_filtered.copy()
    
    df_selected_formation = df_analysis[df_analysis['player_formation'] == selected_formations].copy()
    
    if df_selected_formation.empty:
        print("warning: no matching data found for the selected formation.")
        return
    else:
        print(f"There are {len(df_selected_formation)} shots with the {selected_formations} formation.")

    def calculate_stats(data, cover_col_name):
        valid_data = data.dropna(subset=[cover_col_name])
        valid_data[cover_col_name] = valid_data[cover_col_name].astype(bool)
        
        total = valid_data.groupby(cover_col_name).size()
        fails = valid_data[valid_data['is_defense_failure'] == True].groupby(cover_col_name).size()
        
        stats_df = pd.concat([total, fails], axis=1).fillna(0)
        stats_df.columns = ['total_attempts', 'total_failures']
        
        stats_df['failure_rate'] = (stats_df['total_failures'] / stats_df['total_attempts']) * 100
        stats_df['success_rate'] = 100 - stats_df['failure_rate']
        
        for status in [True, False]:
            if status not in stats_df.index:
                stats_df.loc[status] = [0, 0, 0, 0]
        
        return stats_df.reindex([True, False])

    stats_reaction = calculate_stats(df_selected_formation, 'is_reaction_cover')

    print("\n--- Reaction-Cover ---")
    print(stats_reaction)
    # (Contingency Table)
    #              (Fail)   (Success)
    # CCE (True)   a             b
    # CCE (False)  c             d
    
    fail_covered = int(stats_reaction.loc[True, 'total_failures'])
    total_covered = int(stats_reaction.loc[True, 'total_attempts'])
    success_covered = total_covered - fail_covered
    
    fail_no_cover = int(stats_reaction.loc[False, 'total_failures'])
    total_no_cover = int(stats_reaction.loc[False, 'total_attempts'])
    success_no_cover = total_no_cover - fail_no_cover
    
    obs = np.array([
        [fail_covered, success_covered], 
        [fail_no_cover, success_no_cover]
    ])
    
    chi2, p_val, dof, expected = chi2_contingency(obs)
    
    print(f"Chi-square: {chi2:.2f}")
    print(f"p-value: {p_val:.5f}")

    print(f"\n--- (Chi-Square Test) ---")
    print(f"p-value: {p_val:.5f}")
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['Successful Compensation', 'Failed Compensation']
    success_rates = [stats_reaction.loc[True, 'success_rate'], stats_reaction.loc[False, 'success_rate']]
    colors = ['#34A853', '#EA4335'] 
    
    bars = ax.bar(labels, success_rates, color=colors, width=0.6)
    
    title_str = f'Court Coverage Defense Success Rate (Offensive Formation)\n{filter_title_suffix} (p-value: {p_val:.4f} ***)'
    ax.set_title(title_str, fontsize=14, weight='bold', pad=15)
    ax.set_ylabel('Defense Success Rate (%)', fontsize=12, weight='bold')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        status = [True, False][i]
        
        # 1. 成功率
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, 
                f'{height:.1f}%', ha='center', va='bottom', fontsize=12, weight='bold')
        
        # 2. 成功數/總數
        fail = int(stats_reaction.loc[status, 'total_failures'])
        att = int(stats_reaction.loc[status, 'total_attempts'])
        success = att - fail
        label_text = f"Success: {success} / {att}"
        
        y_pos = height / 2 if height > 10 else height + 5
        text_color = 'white' if height > 20 else 'black'
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, 
                label_text, ha='center', va='center', fontsize=11, color=text_color, weight='bold')

    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.15)

    os.makedirs('./img/Coordination Analysis', exist_ok=True)
    plt.savefig('./img/Coordination Analysis/Court_Coverage.pdf', dpi=400)
    plt.show()
        

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency 

def formation_change_cce_analysis(df, match_id=None, 
                                set_num=None, 
                                start_rally_id=None, 
                                end_rally_id=None,
                                event_type=None,):
    df_filtered, filter_title_suffix = df_filter_by_conditions(
        df,
        match_id=match_id,
        set_num=set_num,
        start_rally_id=start_rally_id,
        end_rally_id=end_rally_id,
        player_id=None, 
        team=None,
        event_type=event_type    
    )

    if df_filtered.empty:
        print(f"No data found under the specified filter conditions ({filter_title_suffix}).")
        return
        
    required_cols = [
        'player', 'A', 'B', 'C', 'D', 'shot_num', 'shot_count',
        'match_Winner_formation', 'match_Loser_formation', 
        'match_Winner_mid_cover', 'match_Loser_mid_cover',
        'score_team', 'match_id', 'set_id', 'rally_id', 'set_win'
    ]
    
    if not all(col in df_filtered.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_filtered.columns]
        print(f"Error: Missing required columns for analysis: {missing}.")
        return

    df_analysis = df_filtered.copy()

    df_analysis['my_next_formation'] = np.where(
        df_analysis['player_team'] == 0.0,
        df_analysis['match_Winner_formation'].shift(-2),
        df_analysis['match_Loser_formation'].shift(-2)
    )

    df_analysis['is_same_rally_after_2'] = df_analysis['rally_id'] == df_analysis['rally_id'].shift(-2)

    df_analysis['my_next_formation'] = np.where(
        df_analysis['is_same_rally_after_2'],
        df_analysis['my_next_formation'],
        'Point End'  
    )

    df_attack = df_analysis[df_analysis['player_formation'] == 'Attack'].copy()
    
    if df_attack.empty:
        print("Warning: No hit data found under the 'Attack' formation.")
        return
    else:
        print(f"Analyzing: A total of {len(df_attack)} hits under the 'Attack' formation.")

    df_attack['is_reaction_cover'] = pd.to_numeric(df_attack['is_reaction_cover'], errors='coerce')

    df_att = df_analysis[
        (df_analysis['player_formation'] == 'Attack') & 
        (df_analysis['is_reaction_cover'].notna())
    ].copy()
    
    def categorize_strict_state(row):
        next_form = row['my_next_formation']
        
        if next_form == 'Point End':
            if not row['did_opponent_team_win']:
                return 'Exclude'
            else:
                return 'Point Lost'
        elif next_form == 'Attack':
            return 'Keep Attack'
        elif next_form == 'Defense':
            return 'To Defense'

    df_att['next_state_strict'] = df_att.apply(categorize_strict_state, axis=1)
    
    df_strict = df_att[df_att['next_state_strict'] != 'Exclude'].copy()
    
    print(f"Original attack samples: {len(df_att)} | After excluding opponent errors, actual tested samples: {len(df_strict)}")

    contingency_table = pd.crosstab(df_strict['is_reaction_cover'], df_strict['next_state_strict'])
    
    expected_cols = [
        'Keep Attack', 
        'To Defense', 
        'Point Lost', 
    ]
    for col in expected_cols:
        if col not in contingency_table.columns:
            contingency_table[col] = 0
            
    contingency_table = contingency_table[expected_cols]
    
    print("\n[Tested State Transition Frequency Table (Counts)]")
    print(contingency_table)
    
    transition_probs = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    
    print("\n[Strict Formation Transition Probability Analysis (%)]")
    transition_probs.index = ['No Coverage (CCE=False)', 'With Coverage (CCE=True)']
    print(transition_probs.round(2).to_string())

    def draw_stacked_bar(ct):
        categories = ['No Coverage(CCE = 0)', 'With Coverage(CCE = 1)']
        
        def get_counts(col_name):
            cce_false = ct.loc[ct.index.isin([False, 0, 0.0]), col_name].sum() if col_name in ct.columns else 0
            cce_true = ct.loc[ct.index.isin([True, 1, 1.0]), col_name].sum() if col_name in ct.columns else 0
            return np.array([cce_false, cce_true])
            
        keep_attack_counts = get_counts('Keep Attack')
        to_defense_counts = get_counts('To Defense')
        point_lost_counts = get_counts('Point Lost')
        
        total_counts = keep_attack_counts + to_defense_counts + point_lost_counts
        
        with np.errstate(divide='ignore', invalid='ignore'):
            keep_attack_pct = np.where(total_counts > 0, (keep_attack_counts / total_counts) * 100, 0)
            to_defense_pct = np.where(total_counts > 0, (to_defense_counts / total_counts) * 100, 0)
            point_lost_pct = np.where(total_counts > 0, (point_lost_counts / total_counts) * 100, 0)

        fig, ax = plt.subplots(figsize=(8, 6))
        width = 0.6
        
        colors = ['#4285F4', '#FBBC05', '#EA4335']

        p1 = ax.bar(categories, keep_attack_pct, width, label='Keep Attack', color=colors[0], edgecolor='white')
        p2 = ax.bar(categories, to_defense_pct, width, bottom=keep_attack_pct, label='To Defense', color=colors[1], edgecolor='white')
        p3 = ax.bar(categories, point_lost_pct, width, bottom=keep_attack_pct + to_defense_pct, label='Point Lost', color=colors[2], edgecolor='white')

        def add_labels(bars, counts, pcts):
            for bar, count, pct in zip(bars, counts, pcts):
                if count == 0: 
                    continue
                height = bar.get_height()
                y_pos = bar.get_y() + height / 2
                ax.text(bar.get_x() + bar.get_width()/2, y_pos-1,
                        f'{pct:.1f}%(N={count})', 
                        ha='center', va='center', color='black', weight='bold', fontsize=14)

        add_labels(p1, keep_attack_counts, keep_attack_pct)
        add_labels(p2, to_defense_counts, to_defense_pct)
        add_labels(p3, point_lost_counts, point_lost_pct)

        ax.set_ylabel('Transition Probability (%)', fontsize=14, weight='bold')
        ax.tick_params(axis='both', labelsize=14)
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False, fontsize=14)
        
        for i, total in enumerate(total_counts):
            ax.text(i, 102, f'Total N={total}', ha='center', weight='bold', color='#333333', fontsize=14)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.subplots_adjust(left=0.11, right=0.9, top=0.9, bottom=0.12)
        
        os.makedirs('./img/Coordination Analysis', exist_ok=True)
        plt.savefig('./img/Coordination Analysis/CCE_Formation_trans_probs.pdf', dpi=400)
        plt.show()

    draw_stacked_bar(contingency_table)

    return transition_probs