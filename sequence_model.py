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


def main():
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


def plot_by_num_and_group(sequences_in, numcol, groupcols):
    seq = sequences_in.copy()
    g = seq.groupby(groupcols)
    n_col = 4
    n_row = len(g) // n_col + len(g) % n_col
    f, axes = plt.subplots(n_row, n_col, figsize=[20, 15], sharex=True, sharey=True,
                           constrained_layout=True)
    axes = axes.ravel()
    for (n, df), (i, ax) in zip(g, enumerate(axes)):
        s = ax.scatter(df[numcol], df['logit_best_xg'],
                       c=df['n_pass'],
                       cmap='viridis')
        v = ax.vlines(df[numcol], df['logit_y_rep_low'],
                      df['logit_y_rep_high'], color='grey', zorder=0)
        ax.set_title(n, y=0.77)
        if i % n_col == 0:
            ax.set_ylabel('Best xg (logit scale)')
        if i >= n_row * n_col - n_col:
            ax.set_xlabel(numcol)

    f.colorbar(s, ax=axes, label='number of passes in sequence', shrink=0.35,
               location='top')
    f.suptitle(f'Max xg in long sequences\nfor different {numcol}s and {groupcols}s',
               fontsize=20, x=0.01, y=0.99, ha='left', fontweight='bold')
    f.legend([v], ['Model predictions: 10%-90% credible interval'], frameon=False,
             loc='upper right', prop={'size': 14})
    plt.savefig(f"plots/yrep_{numcol}_and_{groupcols}.png")
    plt.close('all')


def plot_player_effect(sequences_in, numcol, player, team):
    team_seqs = sequences_in.query(f"team == '{team}'").copy()
    player_seqs = team_seqs.loc[lambda df: df['players'].str.contains(player)]
    f, ax = plt.subplots(figsize=[10, 6])
    cmap = cm.get_cmap('viridis')
    ax.vlines(team_seqs[numcol], team_seqs['logit_y_rep_low'],
              team_seqs['logit_y_rep_high'], color='grey', zorder=0,
              label="Model predictions: 10%-90% credible interval")
    ax.scatter(team_seqs[numcol], team_seqs['logit_best_xg'],
                   color=cmap(0.5), label=f'Moves without {player}')
    ax.scatter(player_seqs[numcol], player_seqs['logit_best_xg'], color='red',
               label=f'Moves with {player}')
    ax.legend(frameon=False, loc='upper left')
    ax.set_title(f'{team} with and without {player}')
    ax.set_xlabel(numcol)
    ax.set_ylabel("Best xg (logit scale)")
    plt.savefig(f"plots/{player}.png")
    plt.close('all')


if __name__ == "__main__":
    infd, mcmcfit, deep_block_sequences = main()
