import matplotlib
from matplotlib import pyplot as plt
from plotting import plot_by_num_and_group, plot_player_effect
from sequence_model import run_sequence_model





def main():
    _, _, dbs = run_sequence_model()

    plt.style.use('sparse.mplstyle')
    matplotlib.rcParams['figure.subplot.top'] = 0.5

    numcol = 'start_distance'
    groupcols = 'team'
    f, axes = plot_by_num_and_group(dbs, numcol, groupcols)
    plt.savefig(f"plots/yrep_{numcol}_and_{groupcols}.png")
    plt.close('all')

    comparison_1 = [('Özil', 'Arsenal'), ('Xhaka', 'Arsenal')]
    comparison_2 = [('Hart', 'West Ham United'), ('Ederson', 'Manchester City')]

    f, axes = plt.subplots(1, 2, figsize=[15, 5], sharey=True, sharex=True)
    for ax, (player, team) in zip(axes, comparison_1):
        plot_player_effect(dbs, 'start_distance', player, team, ax)
    f.suptitle("Özil was crucial to Arsenal's deep block attacks",
               x=0.01, y=1.03, ha='left', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"plots/Özil_vs_Xhaka.png", facecolor=f.get_facecolor(), transparent=True, bbox_inches="tight")
    plt.close('all')
    
    f, axes = plt.subplots(1, 2, figsize=[15, 5], sharey=True, sharex=True)
    for ax, (player, team) in zip(axes, comparison_2):
        plot_player_effect(dbs, 'start_distance', player, team, ax)
    f.suptitle("West Ham United's deep block attacks went wrong when Hart was involved",
               x=0.01, y=1.03, ha='left', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"plots/Hart_vs_Ederson.png", facecolor=f.get_facecolor(), transparent=True, bbox_inches="tight")
    plt.close('all')

if __name__ == "__main__":
    main()
