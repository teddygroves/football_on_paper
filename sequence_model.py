import pandas as pd
import cmdstanpy
import arviz
from scipy.special import expit, logit
from matplotlib import pyplot as plt
from matplotlib import cm

XCOLS = ['period_id', 'start_second', 'is_home_num', 'start_distance']
LIKELIHOOD = 1
plt.style.use('sparse.mplstyle')

def get_input_data(sequences_in, likelihood, xcols):
    seq = sequences_in.copy()
    seq['is_home_num'] = seq['is_home'].astype(int)
    team_map = codify(seq['team'])
    team_ix = seq['team'].map(team_map.get)
    team_opp_ix = seq['team_opp'].map(team_map.get)
    return {
        'N': len(seq),
        'K': len(xcols),
        'T': len(team_map),
        'team': team_ix.values,
        'team_opp': team_opp_ix.values,
        'x': seq[xcols].values,
        'logit_y': seq['logit_best_xg'].values,
        'likelihood': likelihood
    }, team_map


def codify(l):
    return dict(zip(list(set(l)), range(1, len(set(l)) + 1)))


def run_model(sequences_in, likelihood, xcols):
    input_data, team_map = get_input_data(sequences_in, likelihood, xcols)
    model = cmdstanpy.CmdStanModel('sequence_model.stan')
    model.compile()
    mcmcfit = model.sample(data=input_data)
    return mcmcfit, team_map


def run_sequence_model():
    sequences_raw = pd.read_csv('sequences.csv')
    deep_block_sequences = sequences_raw.query('n_pass >= 7').copy()
    mcmcfit, team_map = run_model(deep_block_sequences, LIKELIHOOD, XCOLS)
    mcmcfit.diagnose()
    draws = {}
    for p, cols in zip(
        ['a_team', 'a_team_opp', 'logit_y_rep', 'b'],
        [team_map.keys(), team_map.keys(), deep_block_sequences.index, XCOLS]
    ):
        draw_df = mcmcfit.get_drawset(params=[p])
        draw_df.columns = cols
        draws[p] = draw_df
    deep_block_sequences['logit_y_rep_low'] = draws['logit_y_rep'].quantile(0.1)
    deep_block_sequences['logit_y_rep_median'] = draws['logit_y_rep'].quantile(0.5)
    deep_block_sequences['logit_y_rep_high'] = draws['logit_y_rep'].quantile(0.9)
    return mcmcfit, draws, deep_block_sequences




if __name__ == "__main__":
    infd, mcmcfit, deep_block_sequences = run_sequence_model()
