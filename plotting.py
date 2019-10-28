from matplotlib import pyplot as plt
from matplotlib import cm

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
    return f, axes


def plot_player_effect(sequences_in, numcol, player, team, ax):
    team_seqs = sequences_in.query(f"team == '{team}'").copy()
    player_seqs = team_seqs.loc[lambda df: df['players'].str.contains(player)]
    cmap = cm.get_cmap('viridis')
    ax.vlines(team_seqs[numcol], team_seqs['logit_y_rep_low'],
              team_seqs['logit_y_rep_high'], color='grey', zorder=0,
              label="Model predictions: 10%-90% credible interval")
    ax.scatter(team_seqs[numcol], team_seqs['logit_best_xg'],
                   color=cmap(0.5), label=f'Sequences without {player}')
    ax.scatter(player_seqs[numcol], player_seqs['logit_best_xg'], color='red',
               label=f'Sequences with {player}')
    ax.legend(frameon=False, loc='upper left')
    ax.set_title(f'{team} with and without {player}')
    ax.set_xlabel(numcol)
    ax.set_ylabel("Best xg (logit scale)")
    return ax
