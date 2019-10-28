import cmdstanpy
import pandas as pd
import xml.etree.ElementTree as et
from typing import List, Dict
from qualifier_ids import EVENT_IDS, QUALIFIER_IDS
import json
import os

DATA_DIR = 'example_data'
EVENT_TYPES_TO_NOT_SAVE = [
    'Deleted event',
    'Formation change',
    "Team set up",
    "Start",
    "Coach Setup"
]


def get_player_id_map(filenames: List[str]) -> Dict[int, str]:
    player_id_map = {}
    for filename in filenames:
        if filename[:4] != 'srml':
            continue
        xt = et.parse(os.path.join(DATA_DIR, filename))
        result = xt.getroot()[0]
        teams = [e for e in result if e.tag == 'Team']
        for team in teams:
            players = [e for e in team if e.tag == 'Player']
            for player in players:
                player_id = int(player.attrib['uID'].replace('p', ''))
                known = [e for e in player[0] if e.tag == 'Known']
                if len(known) > 0:
                    player_name = known[0].text
                else:
                    player_name = player[0][0].text + ' ' + player[0][1].text
                player_id_map[player_id] = player_name
    return player_id_map


def get_events_for_filename(filename: str):
    game_events = et.parse(os.path.join(DATA_DIR, filename)).getroot()[0]
    return game_events.attrib, game_events


def get_events(filenames: List[str], player_id_map: Dict[int, str]):
    out = []
    for filename in filenames:
        if filename[:3] != 'f73':
            continue
        game_info, game_events = get_events_for_filename(filename)
        game_info['game_id'] = game_info.pop('id')
        for e in game_events:
            event_type = EVENT_IDS[int(e.attrib['type_id'])]
            if event_type not in EVENT_TYPES_TO_NOT_SAVE:
                qualifiers = {
                    QUALIFIER_IDS[int(q.attrib['qualifier_id'])]: q.attrib['value']
                    if 'value' in q.attrib.keys() else '1'
                    for q in e if int(q.attrib['qualifier_id']) in QUALIFIER_IDS.keys()
                }
                event_dict = {
                    **game_info, 'event_type': event_type, **e.attrib, **qualifiers
                }
                if 'player_id' in event_dict.keys():
                    event_dict['player'] = player_id_map[int(event_dict['player_id'])]
                if 'shot_xg' in event_dict.keys():
                    event_dict['xg'] = event_dict['shot_xg']
                elif 'pass_xg' in event_dict.keys():
                    event_dict['xg'] = event_dict['pass_xg']
                event_dict['team_id_opp'] = (
                    event_dict['away_team_id']
                    if event_dict['team_id'] == event_dict['home_team_id']
                    else event_dict['home_team_id']
                )
                event_dict['is_home'] = (
                    event_dict['team_id'] == event_dict['home_team_id']
                )
                out.append(event_dict)
    return out


def get_team_id_map(filenames):
    game_info_df = pd.DataFrame([
        get_events_for_filename(f)[0] for f in filenames if f[:3] == 'f73'
    ]).apply(pd.to_numeric, errors='ignore')
    return {
        **game_info_df.groupby('home_team_id')['home_team_name'].first().to_dict(),
        **game_info_df.groupby('away_team_id')['away_team_name'].first().to_dict()
    }



if __name__ == "__main__":
    _, _, filenames = next(os.walk(DATA_DIR))
    player_id_map = get_player_id_map(filenames)
    team_id_map = get_team_id_map(filenames)
    events = get_events(filenames, player_id_map)
    event_df = pd.DataFrame(events).apply(pd.to_numeric, errors='ignore')
    event_df['team'] = event_df['team_id'].map(team_id_map.get)
    event_df['team_opp'] = event_df['team_id_opp'].map(team_id_map.get)
    event_df.to_csv('events.csv')
