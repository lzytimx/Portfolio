import random

sci_pap_stone = ['scissors', 'paper', 'stone']


def computer():
    random_no = random.randint(0, 2)
    return sci_pap_stone[random_no]


def player():
    player_choice = input("Choose scissors, paper, or stone : ").lower()
    while player_choice not in sci_pap_stone:
        player_choice = input(
            "You have entered a wrong value. Please enter scissors, paper, or stone")
    return player_choice


def sps_rules(player, computer):
    if player == 'scissors' and computer == 'scissors':
        return 3
    elif player == 'scissors' and computer == 'paper':
        return 1
    elif player == 'scissors' and computer == 'stone':
        return 2
    elif player == 'paper' and computer == 'scissors':
        return 2
    elif player == 'paper' and computer == 'paper':
        return 3
    elif player == 'paper' and computer == 'stone':
        return 1
    elif player == 'stone' and computer == 'scissors':
        return 1
    elif player == 'stone' and computer == 'paper':
        return 2
    elif player == 'stone' and computer == 'stone':
        return 3


def main():
    print("Let's play a game of scissors, paper, stone!")
    win_count, loss_count, tie_count = 0, 0, 0
    flag = True

    while flag:
        computer_choice = computer()
        player_choice = player()
        win_condition = sps_rules(player_choice, computer_choice)

        print(f'Computer chose {computer_choice}')
        print(f'You chose {player_choice}')

        if win_condition == 1:
            win_count += 1
            print('You won!')
        elif win_condition == 2:
            loss_count += 1
            print('You lost!')
        else:
            tie_count += 1
            print('You tied!')

        print(f'Wins: {win_count}, Losses: {loss_count}, Ties: {tie_count}')

        replay = input(
            "Type 'Y' to play again, or 'N' to exit the game: ").upper()
        while replay not in ['Y', 'N']:
            replay = input(
                "Please try again. Type 'Y' to play again, or 'N' to exit the game: ").upper()
        if replay == 'N':
            flag = False
        else:
            pass

    if flag == False:
        print('Goodbye!')


main()
