import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.special import logit

def get_sequences(events):
    events = events.query('period_id in (1, 2)').copy()
    events['goal_distance'] = get_distance_from_goal(events)
    events['is_pass'] = is_pass(events)
    events['is_goal'] = is_goal(events)
    events['is_shot'] = is_shot(events)
    events['period_second'] = get_start_second(events)
    groupcols = ['game_id', 'sequence_id']
    g = events.groupby(groupcols)
    return pd.DataFrame({
        'period_id': g['period_id'].first(),
        'start_second': g['period_second'].first(),
        'team_id': g['team_id'].first(),
        'team': g['team'].first(),
        'team_id_opp': g['team_id_opp'].first(),
        'team_opp': g['team_opp'].first(),
        'game_id': g['game_id'].first(),
        'is_home': g['is_home'].first(),
        'first_event_type': g['event_type'].first(),
        'start_distance': g['goal_distance'].first(),
        'first_xg': g['xg'].first(),
        'best_xg': g['xg'].max(),
        'logit_best_xg': logit(g['xg'].max()),
        'n_pass': g['is_pass'].sum(),
        'goal': g['is_goal'].any(),
        'players': g['player'].apply(get_player_sequence)
    }).reset_index(drop=True)


def get_player_sequence(s):
    non_null = [p.split(' ')[-1] for p in s.dropna()]
    out = [non_null[0]]
    for p in non_null:
        if p != out[-1]:
            out.append(p)
    return '-'.join(out)
    


def get_start_second(events):
    return (
        60 * events['min']
        + events['sec']
        - 45 * 60 * (events['period_id'] - 1)
    )
    

def is_pass(events):
    return events['event_type'] == 'Pass'


def is_goal(events):
    return events['event_type'] == 'Goal'


def is_shot(events):
    shot_event_types = ['Miss', 'Post' 'Goal', 'Attempt Saved']
    return events['event_type'].isin(shot_event_types)


def get_distance_from_goal(events):
    goal_x, goal_y = 100, 50
    dist_x = goal_x - events['x']
    dist_y = goal_y - events['y']
    return np.sqrt(dist_x ** 2 + dist_y ** 2)


if __name__ == "__main__":
    events = pd.read_csv('events.csv').apply(pd.to_numeric, errors='ignore')
    sequences = get_sequences(events)
    sequences.to_csv('sequences.csv')
