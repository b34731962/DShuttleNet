import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

COURT_WIDTH = 61.0
FULL_LENGTH = 134.0
HALF_LENGTH = 67.0


def draw_full_court(ax):
    COURT_COLOR = '#90C080'  
    LINE_COLOR = 'white'
    lw = 2  

    ax.set_facecolor(COURT_COLOR)

    rect = patches.Rectangle((0, 0), COURT_WIDTH, FULL_LENGTH, 
                             linewidth=lw, edgecolor=LINE_COLOR, facecolor='none', zorder=5)
    ax.add_patch(rect)
    
    margin = 4.6
    ax.plot([margin, margin], [0, FULL_LENGTH], color=LINE_COLOR, linewidth=lw, zorder=5)
    ax.plot([COURT_WIDTH-margin, COURT_WIDTH-margin], [0, FULL_LENGTH], color=LINE_COLOR, linewidth=lw, zorder=5)
    
    long_service = 7.6
    ax.plot([0, COURT_WIDTH], [long_service, long_service], color=LINE_COLOR, linewidth=lw, zorder=5)
    ax.plot([0, COURT_WIDTH], [FULL_LENGTH-long_service, FULL_LENGTH-long_service], color=LINE_COLOR, linewidth=lw, zorder=5)
    
    short_service_bottom = 47.2
    short_service_top = FULL_LENGTH - 47.2
    ax.plot([0, COURT_WIDTH], [short_service_bottom, short_service_bottom], color=LINE_COLOR, linewidth=lw, zorder=5)
    ax.plot([0, COURT_WIDTH], [short_service_top, short_service_top], color=LINE_COLOR, linewidth=lw, zorder=5)
    
    ax.plot([0, COURT_WIDTH], [HALF_LENGTH, HALF_LENGTH], color=LINE_COLOR, linewidth=lw+1.5, linestyle='-', zorder=5)
    
    mid_x = COURT_WIDTH / 2
    ax.plot([mid_x, mid_x], [0, short_service_bottom], color=LINE_COLOR, linewidth=lw, zorder=5)
    ax.plot([mid_x, mid_x], [short_service_top, FULL_LENGTH], color=LINE_COLOR, linewidth=lw, zorder=5)

def prepare_occupancy_data(df):
    df_clean = df.copy()
    is_ab_hitter = (df_clean['player'] == df_clean['A']) | (df_clean['player'] == df_clean['B'])
    
    df_clean['h1_x'] = np.where(is_ab_hitter, df_clean['player_A_x'], df_clean['player_C_x'])
    df_clean['h1_y'] = np.where(is_ab_hitter, df_clean['player_A_y'], df_clean['player_C_y'])
    df_clean['h2_x'] = np.where(is_ab_hitter, df_clean['player_B_x'], df_clean['player_D_x'])
    df_clean['h2_y'] = np.where(is_ab_hitter, df_clean['player_B_y'], df_clean['player_D_y'])

    df_clean['d1_x'] = np.where(is_ab_hitter, df_clean['player_C_x'], df_clean['player_A_x'])
    df_clean['d1_y'] = np.where(is_ab_hitter, df_clean['player_C_y'], df_clean['player_A_y'])
    df_clean['d2_x'] = np.where(is_ab_hitter, df_clean['player_D_x'], df_clean['player_B_x'])
    df_clean['d2_y'] = np.where(is_ab_hitter, df_clean['player_D_y'], df_clean['player_B_y'])

    for x, y in [('h1_x','h1_y'), ('h2_x','h2_y')]:
        mask = df_clean[y] > HALF_LENGTH
        df_clean.loc[mask, x] = COURT_WIDTH - df_clean.loc[mask, x]
        df_clean.loc[mask, y] = FULL_LENGTH - df_clean.loc[mask, y]

    for x, y in [('d1_x','d1_y'), ('d2_x','d2_y')]:
        mask = df_clean[y] < HALF_LENGTH
        df_clean.loc[mask, x] = COURT_WIDTH - df_clean.loc[mask, x]
        df_clean.loc[mask, y] = FULL_LENGTH - df_clean.loc[mask, y]

    dx_h = df_clean['h1_x'] - df_clean['h2_x']
    dy_h = df_clean['h1_y'] - df_clean['h2_y']
    mag_h = np.sqrt(dx_h**2 + dy_h**2)
    
    cos_h = np.clip(dx_h / mag_h, -1.0, 1.0)
    angle_h = np.degrees(np.arccos(cos_h))
    df_clean['hitter_angle'] = np.where(angle_h > 90, 180 - angle_h, angle_h)
    df_clean['hitter_angle'] = df_clean['hitter_angle'].round(0)

    # Defender Angle (0~90°) 
    dx_d = df_clean['d1_x'] - df_clean['d2_x']
    dy_d = df_clean['d1_y'] - df_clean['d2_y']
    mag_d = np.sqrt(dx_d**2 + dy_d**2)
    
    cos_d = np.clip(dx_d / mag_d, -1.0, 1.0)
    angle_d = np.degrees(np.arccos(cos_d))
    df_clean['defender_angle'] = np.where(angle_d > 90, 180 - angle_d, angle_d)
    df_clean['defender_angle'] = df_clean['defender_angle'].round(2)

    df_clean.loc[mag_h == 0, 'hitter_angle'] = np.nan
    df_clean.loc[mag_d == 0, 'defender_angle'] = np.nan

    return df_clean

def draw_attack_defense_map(df, title="Attack vs Defense Positioning", filename="smash_defense_map.png"):
    
    h_x = pd.concat([df['h1_x'], df['h2_x']]).dropna()
    h_y = pd.concat([df['h1_y'], df['h2_y']]).dropna()
    d_x = pd.concat([df['d1_x'], df['d2_x']]).dropna()
    d_y = pd.concat([df['d1_y'], df['d2_y']]).dropna()

    COURT_GREEN = '#90C080'

    fig, ax = plt.subplots(figsize=(6, 12), facecolor=COURT_GREEN)
    draw_full_court(ax)

    red_cmap = LinearSegmentedColormap.from_list('pub_red', ["#FAC9C900", "#FF7474", "#CC1A1A", "#800000"], N=256)
    blue_cmap = LinearSegmentedColormap.from_list('pub_blue', ["#7FABFC00", "#2862D8", "#002CAF", "#000066"], N=256)

    sns.kdeplot(
        x=h_x, y=h_y, fill=True, cmap=red_cmap,
        alpha=0.85, bw_adjust=0.75, thresh=0.05, levels=20, ax=ax, zorder=10
    )

    sns.kdeplot(
        x=d_x, y=d_y, fill=True, cmap=blue_cmap,
        alpha=0.85, bw_adjust=0.75, thresh=0.05, levels=10, ax=ax, zorder=10
    )

    # ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=15)
    
    legend_elements = [
        Line2D([0], [0], color='#CC1A1A', lw=4, label='Hitter'),
        Line2D([0], [0], color='#2862D8', lw=4, label='Defender')
    ]

    ax.legend(
        handles=legend_elements, 
        loc='upper center',          
        bbox_to_anchor=(0.5, 1.00),   
        ncol=2,                       
        frameon=False,                 
        fontsize=12,                    
        columnspacing=1.0,             
        handlelength=1.2,              
        handletextpad=0.5,             
        prop={'weight': 'bold', 'size': 16}
    )

    coord_style = {'color': 'white', 'fontsize': 16, 'fontweight': 'bold', 'zorder': 30}
    ax.text(0, -2.5, '(0, 0)', ha='center', va='top', **coord_style)
    ax.text(COURT_WIDTH, -2.5, '(61, 0)', ha='center', va='top', **coord_style)
    ax.text(0, FULL_LENGTH + 2.5, '(0, 134)', ha='center', va='bottom', **coord_style)
    ax.text(COURT_WIDTH, FULL_LENGTH + 2.5, '(61, 134)', ha='center', va='bottom', **coord_style)

    margin_view = 8
    ax.set_xlim(-margin_view, COURT_WIDTH + margin_view)
    ax.set_ylim(-margin_view, FULL_LENGTH + margin_view)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()

    plt.savefig(filename, dpi=400, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    print(f"{filename} (400 DPI) successfully saved!")