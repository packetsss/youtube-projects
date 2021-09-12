# Create by Packetsss
# Personal use is allowed
# Commercial use is prohibited

import random
import string
from words import words


def valid(words_list):
    word = random.choice(words_list)

    while "-" in word or "_" in word or " " in word:
        word = random.choice(words_list)
    return word

def hangman():
    word = valid(words)
    word_letters = set(word)

    alphabet = set(string.ascii_lowercase)

    used_letters = set()

    wrong_guesses = 12

    while wrong_guesses > 0 and len(word_letters) > 0:
        print(f"\nYou have {wrong_guesses} lives, and you have used [{', '.join(used_letters).upper()}]")

        word_lst = [letter if letter in used_letters else "_" for letter in word]

        print(f"Word:    {''.join(word_lst).upper()}")

        user_letter = input("Take a guess: ")

        if user_letter in alphabet - used_letters:
            used_letters.add(user_letter)

            if user_letter in word_letters:
                word_letters.remove(user_letter)
            
            else:
                wrong_guesses -= 1
        
        else:
            print("\nYou've guessed this letter already")
        

    if wrong_guesses == 0:
        print(f"Sorry you lost, the correct word is: {word.upper()}")
    else:
        print(f"Congrets! The correct word is: {word.upper()}")
        
    
if __name__ =='__main__':
    hangman()



        
