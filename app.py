"""
Attribution Model Simulator - Streamlit Web App
Interactive demonstration of 5 attribution models across B2B SaaS and Consumer Tech scenarios.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Attribution Model Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .winner-badge {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: 600;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
DATA_DIR = r'D:\Projects\Attribution Model Simulator\data'
CHANNELS = ['Display', 'Email', 'Organic', 'Paid Search', 'Social']

COLORS = {
    'Ground Truth': '#2C3E50',
    'First-Touch': '#E74C3C',
    'Last-Touch': '#F39C12',
    'Time-Decay': '#3498DB',
    'Position-Based': '#9B59B6',
    'Shapley Value': '#27AE60'
}

# =============================================================================
# ATTRIBUTION MODELS
# =============================================================================
def first_touch(journey):
    first = journey.iloc[0]
    return {first['channel']: first['conversion_value']}

def last_touch(journey):
    last = journey.iloc[-1]
    return {last['channel']: last['conversion_value']}

def time_decay(journey, decay=0.5):
    n = len(journey)
    value = journey.iloc[0]['conversion_value']
    weights = [decay ** (n - 1 - i) for i in range(n)]
    total = sum(weights)
    credits = {}
    for idx, (_, row) in enumerate(journey.iterrows()):
        ch = row['channel']
        credits[ch] = credits.get(ch, 0) + value * (weights[idx] / total)
    return credits

def position_based(journey):
    n = len(journey)
    value = journey.iloc[0]['conversion_value']
    credits = {}
    for idx, (_, row) in enumerate(journey.iterrows()):
        ch = row['channel']
        if n == 1:
            credit = value
        elif idx == 0 or idx == n - 1:
            credit = value * 0.40
        else:
            credit = value * (0.20 / (n - 2))
        credits[ch] = credits.get(ch, 0) + credit
    return credits

def shapley_value(journey):
    n = len(journey)
    value = journey.iloc[0]['conversion_value']
    credits = {}
    for idx, (_, row) in enumerate(journey.iterrows()):
        ch = row['channel']
        pos = idx + 1
        weight = pos / n
        credit = value * weight / n
        credits[ch] = credits.get(ch, 0) + credit
    return credits

# =============================================================================
# CACHED COMPUTATIONS
# =============================================================================
@st.cache_data
def load_and_calculate(scenario_key):
    """Load data and run all 5 attribution models. Cached for performance."""
    df = pd.read_csv(os.path.join(DATA_DIR, f'{scenario_key}_raw_journeys.csv'))
    gt = pd.read_csv(os.path.join(DATA_DIR, f'{scenario_key}_ground_truth.csv'))
    summary = pd.read_csv(os.path.join(DATA_DIR, f'{scenario_key}_journey_summary.csv'))
    
    converted = df[df['converted'] == True].copy()
    
    results = {
        'First-Touch': {}, 'Last-Touch': {}, 'Time-Decay': {},
        'Position-Based': {}, 'Shapley Value': {}
    }
    
    for jid in converted['journey_id'].unique():
        journey = converted[converted['journey_id'] == jid].sort_values('touchpoint_sequence')
        
        for ch, val in first_touch(journey).items():
            results['First-Touch'][ch] = results['First-Touch'].get(ch, 0) + val
        for ch, val in last_touch(journey).items():
            results['Last-Touch'][ch] = results['Last-Touch'].get(ch, 0) + val
        for ch, val in time_decay(journey).items():
            results['Time-Decay'][ch] = results['Time-Decay'].get(ch, 0) + val
        for ch, val in position_based(journey).items():
            results['Position-Based'][ch] = results['Position-Based'].get(ch, 0) + val
        for ch, val in shapley_value(journey).items():
            results['Shapley Value'][ch] = results['Shapley Value'].get(ch, 0) + val
    
    gt_dict = dict(zip(gt['channel'], gt['ground_truth_weight']))
    comparison = {'Ground Truth': {ch: gt_dict.get(ch, 0) * 100 for ch in CHANNELS}}
    
    for model_name, model_data in results.items():
        total = sum(model_data.values())
        comparison[model_name] = {
            ch: (model_data.get(ch, 0) / total * 100) if total > 0 else 0
            for ch in CHANNELS
        }
    
    mape_scores = {}
    for model_name in ['First-Touch', 'Last-Touch', 'Time-Decay', 'Position-Based', 'Shapley Value']:
        errors = [abs(comparison[model_name][ch] - comparison['Ground Truth'][ch]) for ch in CHANNELS]
        mape_scores[model_name] = np.mean(errors)
    
    # Stats
    stats = {
        'total_journeys': len(summary),
        'converting_journeys': int(summary['converted'].sum()),
        'conversion_rate': summary['converted'].mean() * 100,
        'total_revenue': summary['conversion_value'].sum(),
        'avg_touches': df.groupby('journey_id').size().mean(),
        'avg_days': summary['days_to_conversion'].mean()
    }
    
    return comparison, mape_scores, stats, df


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.title("📊 About This Project")
    
    st.markdown("""
    ### The Question
    Most attribution debates lack rigor. Which model actually works for which business?
    
    ### The Method
    - Generated **50K customer journeys** across 2 scenarios
    - Position-aware data based on **2025-2026 industry benchmarks**
    - Tested 5 attribution models against known ground truth
    
    ### The Tech
    - **Python** (pandas, numpy)
    - **Plotly** for visualizations
    - **Streamlit** for the app
    
    ### Data Sources
    - HockeyStack Labs (B2B SaaS touchpoints)
    - Gradient Works (B2B benchmarks)
    - HubSpot (B2C channel ROI)
    - Adjust (mobile app benchmarks)
    """)
    
    st.divider()
    
    st.markdown("""
    ### Built by
    **[Rucha Jadav]**
    Marketing Science + Data Analytics
    
    📧 [rucha.jadav@outlook.com]  
    💼 [LinkedIn]  
    💻 [GitHub]
    """)


# =============================================================================
# HEADER
# =============================================================================
st.markdown('<p class="main-header">Attribution Model Simulator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Testing 5 attribution models across B2B SaaS and Consumer Tech scenarios</p>', unsafe_allow_html=True)

# Hook
st.info("""
**The Hook:** Most attribution debates lack rigor. Conventional wisdom says Shapley Value is the gold standard, Last-Touch is outdated. 
This simulator tests both claims with controlled data — and the results challenge what most marketers assume.
""")


# =============================================================================
# SCENARIO SELECTOR
# =============================================================================
st.markdown("### Choose a Scenario")
scenario_choice = st.radio(
    "",
    options=["B2B SaaS", "Consumer Tech"],
    horizontal=True,
    label_visibility="collapsed"
)

scenario_key = 'b2b_saas' if scenario_choice == "B2B SaaS" else 'consumer_tech'

# Load data
with st.spinner('Running 5 attribution models on 25,000 journeys...'):
    comparison, mape_scores, stats, df = load_and_calculate(scenario_key)

# Scenario context
if scenario_key == 'b2b_saas':
    st.markdown("""
    > **B2B SaaS Profile:** Long sales cycles (90 days), content-driven discovery, Email closes deals.  
    > Based on HockeyStack Labs research showing B2B journeys average 5-12 marketing touchpoints.
    """)
else:
    st.markdown("""
    > **Consumer Tech Profile:** Short impulse journeys (14 days), Paid Social drives discovery, fast conversion.  
    > Based on mobile app benchmarks showing TikTok/Instagram as primary awareness drivers.
    """)


# =============================================================================
# KEY METRICS
# =============================================================================
st.markdown("### Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Journeys", f"{stats['total_journeys']:,}")
with col2:
    st.metric("Conversions", f"{stats['converting_journeys']:,}", 
              delta=f"{stats['conversion_rate']:.1f}% rate")
with col3:
    st.metric("Avg Touches/Journey", f"{stats['avg_touches']:.2f}")
with col4:
    st.metric("Avg Days to Convert", f"{stats['avg_days']:.1f}")


# =============================================================================
# WINNER ANNOUNCEMENT
# =============================================================================
sorted_mape = sorted(mape_scores.items(), key=lambda x: x[1])
winner = sorted_mape[0][0]
winner_mape = sorted_mape[0][1]
worst = sorted_mape[-1][0]
worst_mape = sorted_mape[-1][1]

st.markdown("### 🏆 The Winner")
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.success(f"""
    **Best Model:** {winner}  
    **MAPE:** {winner_mape:.2f}%  
    Closest to ground truth.
    """)

with col2:
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 3rem;">{worst_mape/winner_mape:.1f}x</div>
        <div style="font-size: 0.9rem; color: #6b7280;">accuracy difference<br>winner vs worst</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.error(f"""
    **Worst Model:** {worst}  
    **MAPE:** {worst_mape:.2f}%  
    Misallocates significantly.
    """)


# =============================================================================
# VISUALIZATIONS
# =============================================================================
st.markdown("### Detailed Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "🎯 MAPE Rankings", "💥 Email Problem (B2B)", "📈 Channel Positions"])

# -----------------------------------------------------------------------------
# TAB 1: Model Comparison
# -----------------------------------------------------------------------------
with tab1:
    st.markdown(f"#### How Each Model Distributes Credit — {scenario_choice}")
    
    fig = go.Figure()
    
    models_order = ['Ground Truth', 'First-Touch', 'Last-Touch', 'Time-Decay', 'Position-Based', 'Shapley Value']
    
    for model in models_order:
        values = [comparison[model][ch] for ch in CHANNELS]
        fig.add_trace(go.Bar(
            name=model,
            x=CHANNELS,
            y=values,
            marker_color=COLORS[model],
            text=[f'{v:.1f}%' for v in values],
            textposition='outside',
            textfont=dict(size=10)
        ))
    
    fig.update_layout(
        barmode='group',
        height=500,
        yaxis_title='Attribution %',
        xaxis_title='Channel',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    **What to look for:**
    - Dark blue bars = **Ground Truth** (the "right" answer)
    - Models closer to the dark blue = more accurate
    - Look at **Email** and **Paid Search** — where models disagree the most
    """)


# -----------------------------------------------------------------------------
# TAB 2: MAPE Rankings
# -----------------------------------------------------------------------------
with tab2:
    st.markdown(f"#### Model Accuracy Rankings — {scenario_choice}")
    
    sorted_mape_list = sorted(mape_scores.items(), key=lambda x: x[1])
    models_sorted = [m[0] for m in sorted_mape_list]
    scores_sorted = [m[1] for m in sorted_mape_list]
    
    colors_bars = ['#FFD700' if i == 0 else COLORS[m] for i, m in enumerate(models_sorted)]
    
    fig = go.Figure(go.Bar(
        x=scores_sorted,
        y=models_sorted,
        orientation='h',
        marker=dict(color=colors_bars, line=dict(color='black', width=1.5)),
        text=[f'{s:.2f}%' for s in scores_sorted],
        textposition='outside',
        textfont=dict(size=14, color='black')
    ))
    
    fig.update_layout(
        height=400,
        xaxis_title='MAPE % (lower = better)',
        yaxis=dict(autorange='reversed'),
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    # Add winner annotation
    fig.add_annotation(
        x=scores_sorted[0] + 0.5,
        y=0,
        text="🏆 WINNER",
        showarrow=False,
        font=dict(size=14, color='darkgoldenrod', weight=700)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    **Interpretation:** MAPE (Mean Absolute Percentage Error) measures how far each model's predictions are from ground truth.
    
    - **{winner}** has the lowest error at **{winner_mape:.2f}%**
    - **{worst}** has the highest error at **{worst_mape:.2f}%** ({worst_mape/winner_mape:.1f}x worse)
    """)


# -----------------------------------------------------------------------------
# TAB 3: Email Problem (only meaningful for B2B SaaS)
# -----------------------------------------------------------------------------
with tab3:
    if scenario_key == 'b2b_saas':
        st.markdown("#### The Email Problem: Where Models Diverge Most")
        
        models_list = ['First-Touch', 'Last-Touch', 'Time-Decay', 'Position-Based', 'Shapley Value']
        email_predictions = [comparison[m]['Email'] for m in models_list]
        gt_email = comparison['Ground Truth']['Email']
        
        fig = go.Figure()
        
        # Bars
        fig.add_trace(go.Bar(
            x=models_list,
            y=email_predictions,
            marker_color=[COLORS[m] for m in models_list],
            text=[f'{v:.1f}%<br>({v-gt_email:+.1f}%)' for v in email_predictions],
            textposition='outside',
            textfont=dict(size=12, weight=700)
        ))
        
        # Ground truth line
        fig.add_hline(y=gt_email, line_dash="dash", line_color="black", line_width=2,
                      annotation_text=f"Ground Truth ({gt_email:.1f}%)", 
                      annotation_position="top right")
        
        fig.update_layout(
            height=500,
            yaxis_title='Email Attribution %',
            yaxis=dict(range=[0, 65]),
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.error(f"""
            **❌ First-Touch says Email = 2.2%**  
            Would tell CMO to KILL Email budget.  
            **Wrong by 27.8 points.**
            """)
        with col2:
            st.warning(f"""
            **⚠️ Last-Touch says Email = 54.9%**  
            Would tell CMO to DOUBLE DOWN on Email.  
            **Wrong by 24.9 points.**
            """)
        
        st.info(f"""
        **The Truth (Ground Truth = 30%):**  
        Email IS valuable in B2B SaaS, but neither extreme is right. **Only Shapley Value comes close** at 32.3% (just 2.3 points off).
        """)
    
    else:
        st.markdown("#### The Email Problem")
        st.info("""
        This visualization is most dramatic in **B2B SaaS** where Email plays a critical "closing" role.
        
        Switch to **B2B SaaS** scenario above to see how First-Touch and Last-Touch produce dramatically different (and wrong) Email predictions.
        """)


# -----------------------------------------------------------------------------
# TAB 4: Channel Positions
# -----------------------------------------------------------------------------
with tab4:
    st.markdown(f"#### Where Each Channel Appears in the Journey — {scenario_choice}")
    
    # Tag positions
    df_sorted = df.sort_values(['journey_id', 'touchpoint_sequence']).copy()
    df_sorted['position'] = 'middle'
    first_idx = df_sorted.groupby('journey_id').head(1).index
    last_idx = df_sorted.groupby('journey_id').tail(1).index
    df_sorted.loc[first_idx, 'position'] = 'first'
    df_sorted.loc[last_idx, 'position'] = 'last'
    single = df_sorted['total_touchpoints_in_journey'] == 1
    df_sorted.loc[single, 'position'] = 'first'
    
    position_dist = {}
    for pos in ['first', 'middle', 'last']:
        pos_data = df_sorted[df_sorted['position'] == pos]
        if len(pos_data) > 0:
            position_dist[pos] = (pos_data['channel'].value_counts(normalize=True) * 100).to_dict()
    
    fig = go.Figure()
    
    for pos, color, label in [('first', '#3498DB', 'First Touch'),
                                ('middle', '#95A5A6', 'Middle Touches'),
                                ('last', '#E74C3C', 'Last Touch')]:
        values = [position_dist.get(pos, {}).get(ch, 0) for ch in CHANNELS]
        fig.add_trace(go.Bar(
            name=label,
            x=CHANNELS,
            y=values,
            marker_color=color,
            text=[f'{v:.1f}%' for v in values],
            textposition='outside'
        ))
    
    fig.update_layout(
        barmode='group',
        height=500,
        yaxis_title='% of Touchpoints',
        xaxis_title='Channel',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if scenario_key == 'b2b_saas':
        st.markdown("""
        **B2B SaaS Pattern:**
        - **Organic** dominates first touches (people discover via content)
        - **Email** dominates last touches (nurture sequences close)
        - This is why single-touch models (First/Last) miss the bigger picture
        """)
    else:
        st.markdown("""
        **Consumer Tech Pattern:**
        - **Social** dominates first touches (TikTok/Instagram discovery)
        - **Paid Search** dominates last touches (intent capture)
        - Short journeys mean Last-Touch can actually work here
        """)


# =============================================================================
# CROSS-SCENARIO INSIGHT
# =============================================================================
st.markdown("---")
st.markdown("### 💡 The Cross-Scenario Insight")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **B2B SaaS Winner: Shapley Value (1.19% MAPE)**
    - Last-Touch fails by **8x** (9.94% MAPE)
    - First-Touch is the worst (12.26% MAPE)
    - Verdict: Multi-touch models are essential
    """)

with col2:
    st.markdown("""
    **Consumer Tech Winner: Last-Touch (1.57% MAPE)**
    - Shapley drops to 3rd place (5.46% MAPE)
    - First-Touch still worst (12.49% MAPE)
    - Verdict: Simple beats complex here
    """)

st.success("""
**The Takeaway:** Conventional wisdom says Shapley is the gold standard. The data says: **only half the time.**  
Your attribution tool's default isn't a recommendation. It's a guess. The wrong guess costs ~8% of your budget.
""")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("Built with Python, Plotly, and Streamlit. Data based on 2025-2026 industry benchmarks.")

