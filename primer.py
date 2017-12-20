import argparse
import cv2
import dict_utils
import letter_utils
import random
import time
import os
from os import system

parser = argparse.ArgumentParser(description="Alphabet Letter Primer")
parser.add_argument('--cv_input', default=1, type=int)
parser.add_argument('--min_len', default=3, type=int)
parser.add_argument('--train', default=0, type=int)
args = parser.parse_args()
print(args)

if args.train == 1:
    letter_utils.train()
    exit()

cap = cv2.VideoCapture(args.cv_input)
accolades = ["Correct!", "Yup!", "Yep!", "You got it!", "Nice work.", "Keep going!", "Excellent!", "Wonderful!", "Spectacular!", "Amazing!", "Smart!", "Cool!"]
word_accolades = ["Great job!", "My friend you are superb.", "You really know how to spell.", "Congratulations!", "That is a good word", "It is absolutely perfect", "Tiny human being, you spelled it right!", "You are my favorite speller", "I like the way you spell!"]
letter_criticisms = ["Oh oh, that is not right.", "That is not good man.", "Nope, wrong.", "Try again", "Incorrect", "Bad bad bad!", "No silly, that is not right", "You need to learn how to spell my friend."]
thinking = ["Hmmmm, ok let me think of another word.", "Coming up with another word.", "Pulling out my dictionary", "New word coming up...", "Let me find another good one!", "You are making me tired with all these words."]

words = dict_utils.getWords(args.min_len)
word_to_spell, sentence = dict_utils.getWordToSpell(words)

directory = os.path.dirname(os.path.realpath(__file__))
iterations = 0
spell_count = 0
last_letters = []
last_letter = ''
expected_letter = ''
correct_letter_count = 0
correct_word_count = 0
frames = 0
flag_waiting = False
flag_capture_frames = False
flag_uhoh = False
t = time.time()
total_points = 0


FRAMES_TO_CAPTURE = 10

def say(msg):
    system('say ' + msg)

say('Hello! I am Veena. I can help you practice your spelling.')

while True:
    ret, image_np = cap.read()
    image_np = cv2.flip(image_np, -1)
    cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    res = cv2.waitKey(50)

    detected_letter, img = letter_utils.detect(image_np, expected_letter, correct_letter_count)

    if correct_letter_count < len(word_to_spell):
        expected_letter = word_to_spell[correct_letter_count].upper()

    if ''.join(last_letters[-len(word_to_spell):]) == word_to_spell.upper():

        # Scoring:
        elapsed = str(int(time.time() - t))
        print(elapsed + " seconds elapsed")
        t = time.time()
        say(random.choice(word_accolades))
        say(str(len(word_to_spell)) + " points.")
        total_points += len(word_to_spell)
        min_len = args.min_len + int(total_points / 10)
        words = dict_utils.getWords(min_len)

        flag_waiting = False
        last_letters = []
        iterations = 0
        correct_word_count += 1
        correct_letter_count = 0
        say(random.choice(thinking))
        word_to_spell, sentence = dict_utils.getWordToSpell(words)
        continue

    if last_letter != detected_letter and detected_letter != '':

        if detected_letter == expected_letter:
            say(detected_letter.lower())
            say(random.choice(accolades))
            correct_letter_count += 1
            last_letters.append(detected_letter)
            iterations = 0
            flag_uhoh = False
            flag_capture_frames = False
            frames = 0
            print('Looking for ' + expected_letter + '...')
        elif flag_uhoh == False and iterations > 0:
            say(detected_letter.lower())
            say(random.choice(letter_criticisms))
            flag_uhoh = True

    iterations = iterations + 1

    if res == 13 or flag_capture_frames == True: # Enter
        flag_capture_frames = True
        frames += 1
        millis = int(round(time.time() * 1000))
        img.save(directory + '/images/' + expected_letter + '/' + expected_letter + '_' + str(millis) + '.png','png')

    if frames == FRAMES_TO_CAPTURE:
        msg = 'Captured ' + str(frames) + ' frames for ' + expected_letter
        print(msg)
        frames = 0
        flag_capture_frames = False

    if flag_waiting == False:
        say('Can you spell ' + word_to_spell + '? As in: "' + sentence + '"')
        flag_waiting = True

    if iterations % 50 == 0 and iterations != 0:
        say('The word is ' + word_to_spell)

    if iterations == 149:
        say('Ok, we will spell ' + word_to_spell + ' some other time. You have ' + str(total_points) + ' points.')
        flag_waiting = False
        last_letters = []
        correct_letter_count = 0
        word_to_spell, sentence = dict_utils.getWordToSpell(words)

    if len(last_letters) > 0:
        last_letter = last_letters[-1]

    if res == 27: # Escape
        cv2.destroyAllWindows()
        break
