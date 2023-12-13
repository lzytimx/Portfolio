import random

intro = """
I am thinking of a 3-digit number. Try to guess what it is.
Here are some clues:
When I say:     That means:
    Pico        One digit is correct but in the wrong position
    Fermi       One digit is correct and in the right position
    Bagels      No digits are correct.

I have thought of a number.
    You have 10 guesses to get it.
"""


def main():

    # Hyperparameters
    length_of_answer = 3
    tries = 1000
    no_of_try = 1
    response = None
    hint = None

    # Code
    print(intro)

    correct_answer = generate_answer(length_of_answer)
    print(correct_answer)

    while tries > 0:

        flag = False
        while flag != True:
            response = get_user_input(response, no_of_try)
            flag = validate_input(response)
            if flag != True:
                print(flag)
        no_of_try += 1

        if response != correct_answer:
            hint = get_hint(response, correct_answer)
            print("Oops you did not get it right. Here are some hints: ")
            print(hint)
            tries -= 1
        else:
            print(f'Nice! You got it in {no_of_try} tries!')
            break

    if tries == 0:
        print("Aww. you used up all 10 tries.")
        print("Please try again next time!")


def generate_answer(len):
    # Generates an answer

    number_bank = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    answer = None
    for i in range(len):
        lucky_no = random.randint(0, 9)
        if answer is None:
            answer = number_bank[lucky_no]
        else:
            answer += number_bank[lucky_no]

    return answer


def get_user_input(user_input, tries):
    # gets an input from the user

    msg1 = f'Guess {tries}: '
    return input(msg1)


def validate_input(user_input):

    error_msg1 = 'You did not provide a number. Please try again.'
    error_msg2 = 'You did not provide a valid 3-digit number. Please try again'

    try:
        input_value = int(user_input)
    except:
        return error_msg1
    else:
        if len(user_input) != 3:
            return error_msg2
        else:
            return True


def get_hint(res, ans):
    response = list(res)
    answer = list(ans)
    answer_set = set(ans)

    hint = []

    for i in answer_set:
        locals()[f'no{i}'] = answer.count(str(i))

    for i in range(len(response)):
        if response[i] == answer[i]:
            hint.append('Fermi')
            locals()[f'no{response[i]}'] -= 1
        elif response[i] in answer and locals()[f'no{response[i]}'] > 0:
            hint.append('Pico')
            locals()[f'no{response[i]}'] -= 1
    if len(hint) == 0:
        hint.append('Bagel')

    return hint


if __name__ == '__main__':
    main()
