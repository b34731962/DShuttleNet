import matplotlib.pyplot as plt
import seaborn as sns
def normalize_halfcourt(df):
    position_cols = [
        ('player_A_x','player_A_y'),
        ('player_B_x','player_B_y'),
        ('player_C_x','player_C_y'),
        ('player_D_x','player_D_y')
    ]

    for x_col, y_col in position_cols:
        mask = df[y_col] > 67
        df.loc[mask, x_col] = 61 - df.loc[mask, x_col]
        df.loc[mask, y_col] = 134 - df.loc[mask, y_col]

    return df
def plot_angle_relation_scatter(df, title="Hitter vs Defender Positioning Angles",filename="img/angle/smash.png"):
    """
    繪製攻擊方與防守方站位角度的散佈圖
    x: Hitter Angle (0-90)
    y: Defender Angle (0-90)
    """
    plot_df = df.dropna(subset=['hitter_angle', 'defender_angle'])
    
    plt.figure(figsize=(8, 8))
    
    sns.kdeplot(
        data=plot_df, 
        x='hitter_angle', 
        y='defender_angle', 
        
        fill=True, 
        
        cmap='Reds', 
        
        bw_adjust=0.6, 
        
        thresh=0.01, 
        
        levels=20, 
        
        alpha=0.8, 
        
        zorder=0
    )

    sns.kdeplot(
        data=plot_df, 
        x='hitter_angle', 
        y='defender_angle', 
        fill=False, 
        color="black", 
        linewidths=0.5, 
        alpha=0.2, 
        levels=10, 
        bw_adjust=0.6, 
        zorder=1
    )

    plt.xlabel("Hitter TRA", fontsize=40)
    plt.ylabel("Defender TRA", fontsize=40)
    # plt.title(title, fontsize=15, fontweight='bold')
    
    plt.xlim(-2, 92)
    plt.ylim(-2, 92)
    
    plt.grid(True, linestyle='--', alpha=0.5)


    plt.tight_layout()
    
    plt.savefig(filename, dpi=300)
    plt.show()
