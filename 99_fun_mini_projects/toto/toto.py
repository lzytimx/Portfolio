import numpy as np

def main():

    earnings_array = []
    player_chosen_nos = [2, 8, 9, 28, 36, 40]
    winnings = 0

    for j in range(100000):
        toto_win_array, toto_bonus_no = generate_winning_numbers()
        no_of_wins, no_of_bonus = check_winning_numbers(player_chosen_nos, toto_win_array, toto_bonus_no)
        winnings += counting_wins(no_of_wins, no_of_bonus)

        # print(toto_win_array, toto_bonus_no)
        # print(no_of_wins, no_of_bonus)
        # print(winnings)

    print(winnings)


def generate_winning_numbers():
    """
    The idea is to generate seven numbers and store it in an array. Next,
    a number from 0 to the length of is randomly selected and this
    is used to identify the bonus number (via the index).
    Both the winning array and the bonus number are returned/
    """
    prize_nos = np.random.choice(a=np.arange(1,50), size=7, replace=False)
    bonus_index = np.random.choice(a=np.arange(len(prize_nos)), size=1)
    bonus_no = prize_nos[bonus_index]
    prize_nos = np.array(sorted(prize_nos[prize_nos != bonus_no]))
    return prize_nos, bonus_no

def check_winning_numbers(player, winning_arr, bonus_no):
    no_wins = 0
    no_bonus = 0

    for i in player:
        if i in winning_arr:
            no_wins += 1
        elif i in bonus_no:
            no_bonus += 1

    assert no_wins < 7, "bug"
    assert no_bonus <= 1, "bug"

    return no_wins, no_bonus

def counting_wins(player_wins, player_bonus):
    """
    This function counts the wins for each round of toto.
    The payout is referenced from the toto website and is appended
    in the comments section for reference:

    Group 1 - 6 winning numbers             ; $1,000,000 
    Group 2 - 5 winning numbers + add number; $80,000 8% prize pool     -> these prize pools are wrong
    Group 3 - 5 Winning numbers             ; $55,000 5.5% prize pool   -> these prize pools are wrong
    Group 4 - 4 Winning numbers + add number; $30,000 3% prize pool     -> these prize pools are wrong
    Group 5 - 4 Winning numbers             ; $50
    Group 4 - 3 Winning numbers + add number; $25
    Group 5 - 3 Winning numbers             ; $10
    """
    if player_wins == 6:
        return 1000000 - 1
    elif player_wins == 5 and player_bonus == 1:
        return 200000 - 1
    elif player_wins == 5:
        return 1800 - 1
    elif player_wins == 4 and player_bonus == 1:
        return 400 - 1
    elif player_wins == 4:
        return 50 - 1
    elif player_wins == 3 and player_bonus == 1:
        return 25 - 1
    elif player_wins == 3:
        return 10 - 1
    else:
        return -1
