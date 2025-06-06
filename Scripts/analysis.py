import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM

PLOTS_DIR = Path('Plots')
PLOTS_DIR.mkdir(exist_ok=True)


def dprime_calc(hits, fas, miss, cr):
    hr = (hits + 0.5) / (hits + miss + 1)
    far = (fas + 0.5) / (fas + cr + 1)
    return norm.ppf(hr) - norm.ppf(far)


def load_data(data_dir):
    files = list(Path(data_dir).glob('*.csv'))
    if not files:
        raise RuntimeError(f'No CSV files found in {data_dir}')
    frames = [pd.read_csv(f, encoding='utf-8', engine='python') for f in files]
    data = pd.concat(frames, ignore_index=True)
    data = data.rename(columns=lambda c: c.strip())
    return data


def summarise_trials(trial_data):
    trial_data['correct'] = trial_data['resp_shape'] == trial_data['true_shape']
    trial_data['expect'] = pd.Categorical(trial_data['expect_level'],
                                          categories=['low', 'high'])
    trial_data['bias'] = pd.Categorical(trial_data['angle_bias'])
    results = []
    for (pid, expect, bias), df in trial_data.groupby(['participant', 'expect', 'bias']):
        hits = ((df['true_shape'] == 'square') & (df['resp_shape'] == 'square')).sum()
        fas = ((df['true_shape'] == 'dot') & (df['resp_shape'] == 'square')).sum()
        miss = ((df['true_shape'] == 'square') & (df['resp_shape'] == 'dot')).sum()
        cr = ((df['true_shape'] == 'dot') & (df['resp_shape'] == 'dot')).sum()
        results.append({
            'participant': pid,
            'expect': expect,
            'bias': bias,
            'accuracy': df['correct'].mean(),
            'dprime': dprime_calc(hits, fas, miss, cr),
            'conf': df['conf_level'].astype(float).mean(),
            'agency': df['agency_rating'].astype(float).mean()
        })
    return pd.DataFrame(results)


def run_anovas(summary):
    print("--- Running Repeated Measures ANOVAs ---")
    try:
        a_conf = AnovaRM(summary, 'conf', 'participant', within=['expect', 'bias']).fit()
        print("\n--- ANOVA Results for Confidence ---")
        print(a_conf.summary())
    except Exception as e:
        print(f"\nCould not run ANOVA for Confidence. Error: {e}")

    try:
        a_agency = AnovaRM(summary, 'agency', 'participant', within=['expect', 'bias']).fit()
        print("\n--- ANOVA Results for Agency ---")
        print(a_agency.summary())
    except Exception as e:
        print(f"\nCould not run ANOVA for Agency. Error: {e}")

    try:
        a_dprime = AnovaRM(summary, 'dprime', 'participant', within=['expect', 'bias']).fit()
        print("\n--- ANOVA Results for d' ---")
        print(a_dprime.summary())
    except Exception as e:
        print(f"\nCould not run ANOVA for d'. Error: {e}")


def expectation_effects(trial_data, summary):
    eff = trial_data.groupby(['participant', 'bias', 'expect']).agg({
        'conf_level': 'mean',
        'agency_rating': 'mean'
    }).reset_index()

    if eff.empty:
        print("\nWarning: Not enough data to calculate expectation effects.")
        return summary, None
    
    eff_piv = eff.pivot_table(index=['participant', 'bias'],
                              columns='expect',
                              values=['conf_level', 'agency_rating'])
    
    eff_piv.columns = ['_'.join(map(str, col)).strip() for col in eff_piv.columns.values]
    eff_piv = eff_piv.rename(columns={
        'conf_level_low': 'conf_low', 'conf_level_high': 'conf_high',
        'agency_rating_low': 'agency_low', 'agency_rating_high': 'agency_high'
    })
    eff_piv = eff_piv.reset_index()

    eff_piv['conf_effect'] = eff_piv['conf_high'] - eff_piv['conf_low']
    eff_piv['agency_effect'] = eff_piv['agency_high'] - eff_piv['agency_low']

    baseline = summary[summary['expect'] == 'low'].groupby('participant')['dprime'].mean().reset_index()
    baseline = baseline.rename(columns={'dprime': 'baseline_dprime'})
    effects = eff_piv.merge(baseline, on='participant', how='left')

    if effects.empty or 'conf_effect' not in effects.columns or effects['conf_effect'].isnull().all():
         print("\nWarning: Cannot run Mixed Linear Models due to insufficient data for expectation effects.")
         return summary, effects

    print("\n--- Mixed Linear Model for Confidence Expectation Effect ---")
    conf_model = sm.MixedLM.from_formula('conf_effect ~ baseline_dprime * bias',
                                         groups='participant', data=effects).fit()
    print(conf_model.summary())

    print("\n--- Mixed Linear Model for Agency Expectation Effect ---")
    agency_model = sm.MixedLM.from_formula('agency_effect ~ baseline_dprime * bias',
                                           groups='participant', data=effects).fit()
    print(agency_model.summary())
    return summary, effects


def make_plots(summary, effects):
    sns.set(style='whitegrid', context='talk')
    plot_data = summary.groupby(['expect', 'bias']).agg(
        conf_m=('conf', 'mean'), conf_se=('conf', 'sem'),
        agency_m=('agency', 'mean'), agency_se=('agency', 'sem'),
        dprime_m=('dprime', 'mean'), dprime_se=('dprime', 'sem')
    ).reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_data, x='expect', y='conf_m', hue='bias')
    plt.ylabel('Confidence')
    plt.xlabel('Expectation')
    plt.title('Confidence by Expectation and Bias')
    plt.savefig(PLOTS_DIR / 'conf_plot.png', dpi=300, bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_data, x='expect', y='agency_m', hue='bias')
    plt.ylabel('Agency Rating')
    plt.xlabel('Expectation')
    plt.title('Agency Rating by Expectation and Bias')
    plt.savefig(PLOTS_DIR / 'agency_plot.png', dpi=300, bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_data, x='expect', y='dprime_m', hue='bias')
    plt.ylabel("d'")
    plt.xlabel('Expectation')
    plt.title("d' by Expectation and Bias")
    plt.savefig(PLOTS_DIR / 'dprime_plot.png', dpi=300, bbox_inches='tight')
    plt.clf()

    if effects is not None and not effects.empty:
        plt.figure(figsize=(10, 6))
        sns.lmplot(data=effects, x='baseline_dprime', y='conf_effect', hue='bias', height=6)
        plt.ylabel('High - Low Confidence (Expectation Effect)')
        plt.xlabel("Baseline d'")
        plt.suptitle("Confidence Expectation Effect vs Baseline d'", y=1.02)
        plt.savefig(PLOTS_DIR / 'conf_effect_plot.png', dpi=300, bbox_inches='tight')
        plt.clf()

        plt.figure(figsize=(10, 6))
        sns.lmplot(data=effects, x='baseline_dprime', y='agency_effect', hue='bias', height=6)
        plt.ylabel('High - Low Agency (Expectation Effect)')
        plt.xlabel("Baseline d'")
        plt.suptitle("Agency Expectation Effect vs Baseline d'", y=1.02)
        plt.savefig(PLOTS_DIR / 'agency_effect_plot.png', dpi=300, bbox_inches='tight')
        plt.clf()
    
    plt.close('all')
    print(f"\nPlots saved to '{PLOTS_DIR.resolve()}'")


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'Main Experiment' / 'data'

    try:
        data = load_data(data_dir)
        # Convert all participant IDs to strings to prevent type errors
        data['participant'] = data['participant'].astype(str)
        print(f"Successfully loaded {len(data['participant'].unique())} participant(s) from {data_dir.resolve()}")
    except RuntimeError as e:
        print(f"Error: {e}")
        print(f"Looked for data in: {data_dir.resolve()}")
        return

    trial_data = data[data['phase'].str.strip().str.lower() == 'test'].copy()
    if trial_data.empty:
        print('No test trials found in the data files.')
        return
        
    summary = summarise_trials(trial_data)
    if summary.empty:
        print("Could not generate summary data, skipping analysis.")
        return

    run_anovas(summary)
    summary, effects = expectation_effects(trial_data, summary)
    make_plots(summary, effects)


if __name__ == '__main__':
    main()