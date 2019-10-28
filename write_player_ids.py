import json

def get_player_id_map(filenames: List[str]):
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

if __name__ == '__main__':
    filenames = 
