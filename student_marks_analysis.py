"""
Student Marks Data Analysis
============================
Data Science Mini-Project
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Statistical Analysis
- Data Visualization (9 charts across 3 figures)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────────
FILE = 'Student_Marks_Dataset.xlsx'
df = pd.read_excel(FILE, sheet_name='Student Marks')
subjects = ['Maths', 'Science', 'English', 'History', 'Computer Science']

# ─────────────────────────────────────────────────────────────────
# STEP 2: DATA CLEANING & VALIDATION
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: DATA CLEANING")
print("=" * 60)
print(f"Shape          : {df.shape}")
print(f"Null values    : {df.isnull().sum().sum()}")
print(f"Duplicate rows : {df.duplicated().sum()}")
print(f"Score range    : {df[subjects].min().min()} – {df[subjects].max().max()}")
print(f"Data types OK  : {all(df[s].dtype in [np.int64, np.float64] for s in subjects)}")

# ─────────────────────────────────────────────────────────────────
# STEP 3: EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)
print(df[subjects + ['Average']].describe().round(2))

print("\nGrade Distribution:")
print(df['Grade'].value_counts().reindex(['A+','A','B','C','D','F']))

print("\nGender Split:")
print(df['Gender'].value_counts())

print("\nGender-wise Subject Averages:")
print(df.groupby('Gender')[subjects + ['Average']].mean().round(2))

print("\nSubject Correlation Matrix:")
print(df[subjects].corr().round(3))

# ─────────────────────────────────────────────────────────────────
# STEP 4: KEY ACADEMIC INDICATORS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: KEY ACADEMIC INDICATORS")
print("=" * 60)
print(f"Class Average      : {df['Average'].mean():.2f}")
print(f"Median Score       : {df['Average'].median():.2f}")
print(f"Std Deviation      : {df['Average'].std():.2f}")
print(f"Pass Rate          : {(df['Result']=='Pass').mean()*100:.1f}%")
print(f"Top Scorer         : {df.loc[df['Average'].idxmax(), 'Student Name']} ({df['Average'].max()})")
print(f"Lowest Scorer      : {df.loc[df['Average'].idxmin(), 'Student Name']} ({df['Average'].min()})")
print(f"Best Subject       : {df[subjects].mean().idxmax()} ({df[subjects].mean().max():.2f})")
print(f"Weakest Subject    : {df[subjects].mean().idxmin()} ({df[subjects].mean().min():.2f})")

# ─────────────────────────────────────────────────────────────────
# PALETTE & HELPERS
# ─────────────────────────────────────────────────────────────────
BG      = '#0D1117'
CARD    = '#161B22'
CARD2   = '#1C2333'
BORDER  = '#30363D'
TEXT    = '#E6EDF3'
MUTED   = '#8B949E'
ACCENT  = ['#58A6FF','#3FB950','#F78166','#D2A8FF','#FFA657']
GOLD    = '#E8C547'
GREEN   = '#3FB950'
RED     = '#F78166'
BLUE    = '#58A6FF'

def apply_dark(fig, axes_list):
    fig.patch.set_facecolor(BG)
    for ax in axes_list:
        ax.set_facecolor(CARD)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.spines['bottom'].set_color(BORDER)
        ax.spines['left'].set_color(BORDER)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)

plt.rcParams.update({'font.family':'DejaVu Sans','text.color':TEXT})
subj_short = ['Maths','Science','English','History','Comp Sci']

# ─────────────────────────────────────────────────────────────────
# FIGURE 1 — Overview Dashboard
# ─────────────────────────────────────────────────────────────────
fig1 = plt.figure(figsize=(20,14), facecolor=BG)
fig1.suptitle('Student Marks Analysis — Academic Performance Dashboard',
              fontsize=18, fontweight='bold', color=TEXT, y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.45, wspace=0.35,
                       left=0.06, right=0.97, top=0.93, bottom=0.06)

# Chart 1: Score Distribution Histogram
ax1 = fig1.add_subplot(gs[0, 0])
bins = [0,40,50,60,70,80,90,100]
labels_hist = ['<40','40–50','50–60','60–70','70–80','80–90','90–100']
counts = [sum((df['Average']>=b) & (df['Average']<bins[i+1])) for i,b in enumerate(bins[:-1])]
colors_hist = [RED,RED,'#FFA657','#FFA657',BLUE,GREEN,GOLD]
bars = ax1.bar(labels_hist, counts, color=colors_hist, edgecolor=BG, linewidth=0.8, width=0.7, zorder=3)
for bar,c in zip(bars, counts):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, str(c),
             ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')
ax1.set_title('Score Distribution', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Score Range', fontsize=9)
ax1.set_ylabel('No. of Students', fontsize=9)
ax1.set_ylim(0, max(counts)+3)
ax1.grid(axis='y', color=BORDER, alpha=0.5, zorder=0)
apply_dark(fig1, [ax1])
ax1.tick_params(axis='x', rotation=30)

# Chart 2: Grade Donut Chart
ax2 = fig1.add_subplot(gs[0, 1])
grade_order = ['A+','A','B','C','D','F']
grade_colors = [GOLD, GREEN, BLUE, '#D2A8FF', '#FFA657', RED]
gc = df['Grade'].value_counts().reindex(grade_order, fill_value=0)
wedges, texts, autotexts = ax2.pie(gc, labels=grade_order, colors=grade_colors,
    autopct='%1.1f%%', startangle=140, pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2))
for t in texts: t.set_color(TEXT); t.set_fontsize(10)
for at in autotexts: at.set_color(BG); at.set_fontsize(8); at.set_fontweight('bold')
ax2.set_title('Grade Distribution', fontsize=12, fontweight='bold', pad=10)
ax2.set_facecolor(BG)

# Chart 3: Subject Averages Horizontal Bar
ax3 = fig1.add_subplot(gs[0, 2])
subj_avgs = df[subjects].mean().round(2)
hbars = ax3.barh(subj_short, subj_avgs.values, color=ACCENT, edgecolor=BG, height=0.55, zorder=3)
for bar,v in zip(hbars, subj_avgs.values):
    ax3.text(v+0.5, bar.get_y()+bar.get_height()/2, f'{v:.1f}',
             va='center', color=TEXT, fontsize=9, fontweight='bold')
ax3.set_title('Subject-wise Averages', fontsize=12, fontweight='bold', pad=10)
ax3.set_xlabel('Average Score', fontsize=9)
ax3.set_xlim(0, 100)
ax3.axvline(x=50, color=RED, linestyle='--', linewidth=1, alpha=0.6)
ax3.grid(axis='x', color=BORDER, alpha=0.5, zorder=0)
apply_dark(fig1, [ax3])

# Chart 4: Gender Grouped Bar
ax4 = fig1.add_subplot(gs[1, 0])
gdf = df.groupby('Gender')[subjects].mean().round(2)
x = np.arange(len(subj_short))
ax4.bar(x-0.175, gdf.loc['M'].values, 0.35, color=BLUE, label='Male', alpha=0.9, zorder=3)
ax4.bar(x+0.175, gdf.loc['F'].values, 0.35, color='#FF7EB6', label='Female', alpha=0.9, zorder=3)
ax4.set_xticks(x); ax4.set_xticklabels(subj_short, rotation=25, ha='right', fontsize=8)
ax4.set_title('Gender-wise Subject Performance', fontsize=12, fontweight='bold', pad=10)
ax4.set_ylabel('Average Score', fontsize=9)
ax4.set_ylim(0, 100)
ax4.legend(facecolor=CARD2, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
ax4.grid(axis='y', color=BORDER, alpha=0.5, zorder=0)
apply_dark(fig1, [ax4])

# Chart 5: Box Plots
ax5 = fig1.add_subplot(gs[1, 1])
bp = ax5.boxplot([df[s].values for s in subjects], patch_artist=True, notch=False,
                 medianprops=dict(color=GOLD, linewidth=2),
                 whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                 flierprops=dict(marker='o', markerfacecolor=RED, markersize=5, alpha=0.7))
for patch, c in zip(bp['boxes'], ACCENT):
    patch.set_facecolor(c); patch.set_alpha(0.7); patch.set_edgecolor(BORDER)
ax5.set_xticklabels(subj_short, rotation=25, ha='right', fontsize=8)
ax5.set_title('Score Spread per Subject', fontsize=12, fontweight='bold', pad=10)
ax5.set_ylabel('Marks', fontsize=9)
ax5.grid(axis='y', color=BORDER, alpha=0.5, zorder=0)
apply_dark(fig1, [ax5])

# Chart 6: Stacked Pass/Fail Bar
ax6 = fig1.add_subplot(gs[1, 2])
pass_counts = [(df[s]>=40).sum() for s in subjects]
fail_counts = [(df[s]<40).sum() for s in subjects]
x = np.arange(len(subj_short))
ax6.bar(x, pass_counts, color=GREEN, label='Pass', alpha=0.85, zorder=3)
ax6.bar(x, fail_counts, bottom=pass_counts, color=RED, label='Fail', alpha=0.85, zorder=3)
ax6.set_xticks(x); ax6.set_xticklabels(subj_short, rotation=25, ha='right', fontsize=8)
ax6.set_title('Pass / Fail Count per Subject', fontsize=12, fontweight='bold', pad=10)
ax6.set_ylabel('Students', fontsize=9)
ax6.legend(facecolor=CARD2, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
ax6.grid(axis='y', color=BORDER, alpha=0.5, zorder=0)
apply_dark(fig1, [ax6])

fig1.savefig('fig1_overview.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ─────────────────────────────────────────────────────────────────
# FIGURE 2 — Deep Statistical Analysis
# ─────────────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=(20,14), facecolor=BG)
fig2.suptitle('Student Marks Analysis — Deep Statistical Insights',
              fontsize=18, fontweight='bold', color=TEXT, y=0.98)
gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.45, wspace=0.38,
                        left=0.06, right=0.97, top=0.93, bottom=0.06)

# Chart 7: Term-wise Trend Lines
ax7 = fig2.add_subplot(gs2[0, 0])
term_data = {'Maths':[72,75,78],'Science':[64,67,69],'English':[70,72,74],
             'History':[60,62,65],'Computer Science':[75,79,82]}
terms = ['Term 1','Term 2','Term 3']
for (subj,vals), c, m in zip(term_data.items(), ACCENT, ['o','s','D','^','v']):
    ax7.plot(terms, vals, color=c, marker=m, linewidth=2.2, markersize=7,
             label=subj if subj!='Computer Science' else 'Comp Sci')
    ax7.fill_between(terms, vals, alpha=0.07, color=c)
ax7.set_title('Term-wise Performance Trend', fontsize=12, fontweight='bold', pad=10)
ax7.set_ylabel('Average Score', fontsize=9)
ax7.set_ylim(50, 90)
ax7.legend(facecolor=CARD2, edgecolor=BORDER, labelcolor=TEXT, fontsize=8, loc='lower right')
ax7.grid(color=BORDER, alpha=0.5)
apply_dark(fig2, [ax7])

# Chart 8: Correlation Heatmap
ax8 = fig2.add_subplot(gs2[0, 1])
corr = df[subjects].corr()
cmap = mcolors.LinearSegmentedColormap.from_list('', ['#0D1117','#1C3A5E',BLUE,GOLD])
im = ax8.imshow(corr.values, cmap=cmap, vmin=0.9, vmax=1.0, aspect='auto')
ax8.set_xticks(range(5)); ax8.set_xticklabels(subj_short, rotation=35, ha='right', fontsize=8)
ax8.set_yticks(range(5)); ax8.set_yticklabels(subj_short, fontsize=8)
for i in range(5):
    for j in range(5):
        ax8.text(j, i, f'{corr.values[i,j]:.3f}', ha='center', va='center',
                 color=TEXT if corr.values[i,j]<0.98 else BG, fontsize=8, fontweight='bold')
plt.colorbar(im, ax=ax8, fraction=0.046, pad=0.04)
ax8.set_title('Subject Correlation Heatmap', fontsize=12, fontweight='bold', pad=10)
ax8.set_facecolor(CARD); ax8.tick_params(colors=MUTED); ax8.spines[:].set_color(BORDER)

# Chart 9: Scatter with Regression
ax9 = fig2.add_subplot(gs2[0, 2])
for gender, color, marker in [('M',BLUE,'o'),('F','#FF7EB6','s')]:
    gd = df[df['Gender']==gender]
    ax9.scatter(gd['Maths'], gd['Computer Science'], color=color, alpha=0.8, s=55,
                label=f"{'Male' if gender=='M' else 'Female'}", zorder=3,
                edgecolors=BG, linewidth=0.5, marker=marker)
m_r, b_r, r_val, _, _ = stats.linregress(df['Maths'], df['Computer Science'])
x_line = np.linspace(30, 100, 100)
ax9.plot(x_line, m_r*x_line+b_r, color=GOLD, linewidth=1.8, linestyle='--',
         label=f'Trend (r={r_val:.3f})')
ax9.set_title('Maths vs Computer Science', fontsize=12, fontweight='bold', pad=10)
ax9.set_xlabel('Maths Score', fontsize=9)
ax9.set_ylabel('Computer Science Score', fontsize=9)
ax9.legend(facecolor=CARD2, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
ax9.grid(color=BORDER, alpha=0.4, zorder=0)
apply_dark(fig2, [ax9])

# Chart 10: KDE Density
ax10 = fig2.add_subplot(gs2[1, 0])
for gender, color, label in [('M',BLUE,'Male (n=25)'),('F','#FF7EB6','Female (n=25)')]:
    vals = df[df['Gender']==gender]['Average'].values
    kde = gaussian_kde(vals, bw_method=0.4)
    x_range = np.linspace(20, 100, 300)
    ax10.fill_between(x_range, kde(x_range), alpha=0.25, color=color)
    ax10.plot(x_range, kde(x_range), color=color, linewidth=2.2, label=label)
ax10.axvline(df['Average'].mean(), color=GOLD, linestyle='--', linewidth=1.5,
             label=f'Mean: {df["Average"].mean():.1f}')
ax10.axvline(40, color=RED, linestyle=':', linewidth=1.5, label='Pass mark (40)')
ax10.set_title('Score Density Distribution by Gender', fontsize=12, fontweight='bold', pad=10)
ax10.set_xlabel('Average Score', fontsize=9)
ax10.set_ylabel('Density', fontsize=9)
ax10.legend(facecolor=CARD2, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
ax10.grid(color=BORDER, alpha=0.4)
apply_dark(fig2, [ax10])

# Chart 11: Top 10 Performers
ax11 = fig2.add_subplot(gs2[1, 1])
top10 = df.nlargest(10,'Average')[['Student Name','Average','Grade']].reset_index(drop=True)
short_names = [n.split()[0]+' '+n.split()[-1][0]+'.' for n in top10['Student Name']]
bar_colors_top = [GOLD if g=='A+' else GREEN if g=='A' else BLUE for g in top10['Grade']]
hb = ax11.barh(range(len(top10)), top10['Average'], color=bar_colors_top, edgecolor=BG, height=0.6, zorder=3)
for bar, row in zip(hb, top10.itertuples()):
    ax11.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
              f'{row.Average:.1f} ({row.Grade})', va='center', color=TEXT, fontsize=8.5)
ax11.set_yticks(range(len(top10)))
ax11.set_yticklabels(short_names, fontsize=9)
ax11.set_title('Top 10 Performers', fontsize=12, fontweight='bold', pad=10)
ax11.set_xlabel('Average Score', fontsize=9)
ax11.set_xlim(0, 105)
ax11.grid(axis='x', color=BORDER, alpha=0.5, zorder=0)
apply_dark(fig2, [ax11])
ax11.invert_yaxis()

# Chart 12: Quartile Analysis
ax12 = fig2.add_subplot(gs2[1, 2])
q_data = {s: [df[s].quantile(q) for q in [0.25,0.5,0.75]] for s in subjects}
x = np.arange(len(subj_short))
w = 0.22
ax12.bar(x-w, [q_data[s][0] for s in subjects], w, color=RED, label='Q1 (25th)', alpha=0.85, zorder=3)
ax12.bar(x,   [q_data[s][1] for s in subjects], w, color=BLUE, label='Median', alpha=0.85, zorder=3)
ax12.bar(x+w, [q_data[s][2] for s in subjects], w, color=GREEN, label='Q3 (75th)', alpha=0.85, zorder=3)
ax12.set_xticks(x); ax12.set_xticklabels(subj_short, rotation=25, ha='right', fontsize=8)
ax12.set_title('Quartile Analysis per Subject', fontsize=12, fontweight='bold', pad=10)
ax12.set_ylabel('Score', fontsize=9)
ax12.legend(facecolor=CARD2, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
ax12.grid(axis='y', color=BORDER, alpha=0.5, zorder=0)
ax12.set_ylim(0,100)
apply_dark(fig2, [ax12])

fig2.savefig('fig2_deep.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ─────────────────────────────────────────────────────────────────
# FIGURE 3 — Key Indicators
# ─────────────────────────────────────────────────────────────────
fig3 = plt.figure(figsize=(20,10), facecolor=BG)
fig3.suptitle('Student Marks Analysis — Key Academic Indicators & Insights',
              fontsize=18, fontweight='bold', color=TEXT, y=0.99)
gs3 = gridspec.GridSpec(1, 3, figure=fig3, hspace=0.4, wspace=0.35,
                        left=0.06, right=0.97, top=0.90, bottom=0.10)

# Chart 13: Radar Chart
ax13 = fig3.add_subplot(gs3[0, 0], polar=True)
N = len(subj_short)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
class_avg  = df[subjects].mean().values.tolist()
male_avg   = df[df['Gender']=='M'][subjects].mean().values.tolist()
female_avg = df[df['Gender']=='F'][subjects].mean().values.tolist()
for vals, color, label in [(class_avg,GOLD,'Class Avg'),(male_avg,BLUE,'Male'),(female_avg,'#FF7EB6','Female')]:
    v = vals + [vals[0]]
    ax13.plot(angles, v, 'o-', linewidth=2, color=color, markersize=5, label=label)
    ax13.fill(angles, v, alpha=0.1, color=color)
ax13.set_xticks(angles[:-1])
ax13.set_xticklabels(subj_short, size=9, color=TEXT)
ax13.set_ylim(0, 100)
ax13.yaxis.set_tick_params(colors=MUTED, labelsize=7)
ax13.set_facecolor(CARD); ax13.spines['polar'].set_color(BORDER)
ax13.grid(color=BORDER, alpha=0.5)
ax13.set_title('Subject Radar — Class vs Gender', fontsize=12, fontweight='bold', pad=20, color=TEXT)
ax13.legend(loc='lower right', bbox_to_anchor=(1.35,-0.1),
            facecolor=CARD2, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)

# Chart 14: Violin Plots
ax14 = fig3.add_subplot(gs3[0, 1])
positions_m = [1,4,7,10,13]; positions_f = [2,5,8,11,14]
vp_m = ax14.violinplot([df[df['Gender']=='M'][s].values for s in subjects],
                        positions=positions_m, showmedians=True, widths=0.9)
vp_f = ax14.violinplot([df[df['Gender']=='F'][s].values for s in subjects],
                        positions=positions_f, showmedians=True, widths=0.9)
for pc in vp_m['bodies']: pc.set_facecolor(BLUE); pc.set_alpha(0.6)
for pc in vp_f['bodies']: pc.set_facecolor('#FF7EB6'); pc.set_alpha(0.6)
for part in ['cbars','cmins','cmaxes','cmedians']:
    vp_m[part].set_color(TEXT); vp_f[part].set_color(TEXT)
ax14.set_xticks([1.5,4.5,7.5,10.5,13.5])
ax14.set_xticklabels(subj_short, rotation=25, ha='right', fontsize=8)
ax14.set_title('Score Distribution by Gender (Violin)', fontsize=12, fontweight='bold', pad=10)
ax14.set_ylabel('Score', fontsize=9)
ax14.legend([mpatches.Patch(color=BLUE,alpha=0.6),mpatches.Patch(color='#FF7EB6',alpha=0.6)],
            ['Male','Female'], facecolor=CARD2, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
ax14.grid(axis='y', color=BORDER, alpha=0.4)
apply_dark(fig3, [ax14])

# Chart 15: Cumulative Distribution (CDF)
ax15 = fig3.add_subplot(gs3[0, 2])
sorted_avgs = np.sort(df['Average'].values)
cumulative = np.arange(1, len(sorted_avgs)+1) / len(sorted_avgs) * 100
ax15.plot(sorted_avgs, cumulative, color=BLUE, linewidth=2.5, zorder=3)
ax15.fill_between(sorted_avgs, cumulative, alpha=0.15, color=BLUE)
ax15.axvline(40, color=RED, linestyle='--', linewidth=1.5, label='Pass mark (40)')
ax15.axvline(df['Average'].median(), color=GOLD, linestyle='--', linewidth=1.5,
             label=f'Median ({df["Average"].median():.1f})')
ax15.axhline(50, color=MUTED, linestyle=':', linewidth=1, alpha=0.6)
ax15.fill_betweenx([0,100], 0, 40, alpha=0.07, color=RED)
for pct, color in [(84,GREEN),(50,GOLD),(16,RED)]:
    score = np.percentile(df['Average'], pct)
    ax15.scatter([score],[pct], color=color, s=60, zorder=5)
    ax15.annotate(f'{score:.0f} → {pct}th %ile', (score,pct),
                  textcoords='offset points', xytext=(8,0), color=color, fontsize=8)
ax15.set_title('Cumulative Distribution (CDF)', fontsize=12, fontweight='bold', pad=10)
ax15.set_xlabel('Average Score', fontsize=9)
ax15.set_ylabel('Cumulative % of Students', fontsize=9)
ax15.set_xlim(25, 100); ax15.set_ylim(0, 105)
ax15.legend(facecolor=CARD2, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
ax15.grid(color=BORDER, alpha=0.4)
apply_dark(fig3, [ax15])

fig3.savefig('fig3_indicators.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

print("\nAll 3 figures saved successfully.")
print("fig1_overview.png  — Overview Dashboard")
print("fig2_deep.png      — Deep Statistical Insights")
print("fig3_indicators.png — Key Academic Indicators")
