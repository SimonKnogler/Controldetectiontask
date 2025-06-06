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
            'agency': df['control_rating'].astype(float).mean()
        })
    return pd.DataFrame(results)


def run_anovas(summary):
    a_conf = AnovaRM(summary, 'conf', 'participant', within=['expect', 'bias']).fit()
    a_agency = AnovaRM(summary, 'agency', 'participant', within=['expect', 'bias']).fit()
    a_dprime = AnovaRM(summary, 'dprime', 'participant', within=['expect', 'bias']).fit()
    print(a_conf.summary())
    print(a_agency.summary())
    print(a_dprime.summary())


def expectation_effects(trial_data, summary):
    medium = trial_data[np.isclose(trial_data['prop_used'], 0.5)]
    eff = medium.groupby(['participant', 'bias', 'expect']).agg({
        'conf_level': 'mean',
        'control_rating': 'mean'
    }).reset_index()
    eff_piv = eff.pivot_table(index=['participant', 'bias'],
                              columns='expect',
                              values=['conf_level', 'control_rating'])
    eff_piv.columns = ['conf_low', 'conf_high', 'agency_low', 'agency_high']
    eff_piv = eff_piv.reset_index()
    eff_piv['conf_effect'] = eff_piv['conf_high'] - eff_piv['conf_low']
    eff_piv['agency_effect'] = eff_piv['agency_high'] - eff_piv['agency_low']

    baseline = summary[summary['expect'] == 'low'].groupby('participant')['dprime'].mean().reset_index()
    baseline = baseline.rename(columns={'dprime': 'baseline_dprime'})
    effects = eff_piv.merge(baseline, on='participant', how='left')

    conf_model = sm.MixedLM.from_formula('conf_effect ~ baseline_dprime * bias',
                                         groups='participant', data=effects).fit()
    agency_model = sm.MixedLM.from_formula('agency_effect ~ baseline_dprime * bias',
                                           groups='participant', data=effects).fit()
    print(conf_model.summary())
    print(agency_model.summary())
    return summary, effects


def make_plots(summary, effects):
    sns.set(style='whitegrid')
    plot_data = summary.groupby(['expect', 'bias']).agg(
        conf_m=('conf', 'mean'), conf_se=('conf', 'sem'),
        agency_m=('agency', 'mean'), agency_se=('agency', 'sem'),
        dprime_m=('dprime', 'mean'), dprime_se=('dprime', 'sem')
    ).reset_index()

    ax = sns.barplot(data=plot_data, x='expect', y='conf_m', hue='bias', capsize=.2)
    ax.set(ylabel='Confidence', title='Confidence by expectation and bias')
    ax.figure.savefig(PLOTS_DIR / 'conf_plot.png', dpi=300)
    plt.clf()

    ax = sns.barplot(data=plot_data, x='expect', y='agency_m', hue='bias', capsize=.2)
    ax.set(ylabel='Agency rating', title='Agency rating by expectation and bias')
    ax.figure.savefig(PLOTS_DIR / 'agency_plot.png', dpi=300)
    plt.clf()

    ax = sns.barplot(data=plot_data, x='expect', y='dprime_m', hue='bias', capsize=.2)
    ax.set(ylabel="d'", title="d' by expectation and bias")
    ax.figure.savefig(PLOTS_DIR / 'dprime_plot.png', dpi=300)
    plt.clf()

    ax = sns.scatterplot(data=effects, x='baseline_dprime', y='conf_effect', hue='bias')
    sns.regplot(data=effects, x='baseline_dprime', y='conf_effect', scatter=False, ax=ax)
    ax.set(ylabel='High - Low confidence', title="Confidence expectation effect vs baseline d'")
    ax.figure.savefig(PLOTS_DIR / 'conf_effect_plot.png', dpi=300)
    plt.clf()

    ax = sns.scatterplot(data=effects, x='baseline_dprime', y='agency_effect', hue='bias')
    sns.regplot(data=effects, x='baseline_dprime', y='agency_effect', scatter=False, ax=ax)
    ax.set(ylabel='High - Low agency', title="Agency expectation effect vs baseline d'")
    ax.figure.savefig(PLOTS_DIR / 'agency_effect_plot.png', dpi=300)
    plt.clf()


def main():
    data = load_data(Path('Main Experiment') / 'data')
    trial_data = data[data['phase'].str.strip().str.lower() == 'test'].copy()
    if trial_data.empty:
        print('No test trials found in data files.')
        return
    summary = summarise_trials(trial_data)
    run_anovas(summary)
    summary, effects = expectation_effects(trial_data, summary)
    make_plots(summary, effects)


if __name__ == '__main__':
    main()
