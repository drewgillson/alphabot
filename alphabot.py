""" _    _       _           _           _
   / \  | |_ __ | |__   __ _| |__   ___ | |_
  / _ \ | | '_ \| '_ \ / _` | '_ \ / _ \| __|
 / ___ \| | |_) | | | | (_| | |_) | (_) | |_
/_/   \_\_| .__/|_| |_|\__,_|_.__/ \___/ \__|
          |_|
A screen-less interactive spelling primer powered by computer vision

Copyright (C) 2018  Drew Gillson <drew.gillson@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import sys
import argparse
import cv2
import letter_utils
import random
import time
import os
import detector_utils
from detector_utils import WebcamVideoStream
import tensorflow as tf
from multiprocessing import Queue, Pool
import subprocess
from collections import deque
import numpy as np
import csv
import PIL.Image as Image
from traceback import print_exc
import trie


class RingBuffer(deque):
    def __init__(self, size):
        deque.__init__(self)
        self.size = size

    def full_append(self, item):
        deque.append(self, item)
        # full, pop the oldest item, left most item
        self.popleft()

    def append(self, item):
        deque.append(self, item)
        # max size reached, append becomes full_append
        if len(self) == self.size:
            self.append = self.full_append

    def get(self):
        return list(self)


def say(msg):
    if isinstance(msg, list):
        msg = msg[0]

    if sys.platform != "darwin" and args.tts == "say":
        raise Exception("If your operating system is not Mac OS you will need to use a different command line utility"
                        "to render text to speech. espeak is one option: http://espeak.sourceforge.net/download.html")
    subprocess.run([args.tts, msg])


def worker_hands(input_q, output_q):
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    while True:
        frame = input_q.get()
        if frame is not None:
            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)
            output_q.put((boxes, scores))
        else:
            output_q.put((boxes, scores))
    sess.close()


def worker_letters(input_q, output_q):
    nn = letter_utils.letter_net()

    while True:
        frame, min_certainty = input_q.get()
        letter = ''
        if frame is not None and frame.shape == (28, 28):
            for_pred = frame.reshape(1, 28, 28, 1).astype('float32') / 255
            y_pred = nn.predict(for_pred, batch_size=1)
            certainty = np.amax(y_pred, 1)
            guess = chr(np.argmax(y_pred, 1) + ord('A'))
            if args.debug:
                print(str(certainty) + ': ' + guess)
            if certainty > min_certainty:
                letter = guess

        output_q.put(letter)


def read_from_csv(file):
    with open('corpus/' + file + '.csv', 'r') as f:
        reader = csv.reader(f)
        contents = [list(filter(None, c)) for c in list(reader)]
        return contents


def get_word_to_spell(min_length=3):
    global words, last_letters, last_letter, correct_letter_count
    time.sleep(2)

    last_letters = []
    last_letter = ''
    correct_letter_count = 0

    while True:
        word = random.choice(words)
        if len(word[0]) >= min_length and len(word[0]) < (min_length + 2):
            sentence = word[1]
            say('Can you spell ' + word[0] + '? As in: "' + sentence + '"')
            return word[0].upper(), sentence


def get_touched_letter():
    global frames

    # Submit the frame to the hand detector queue
    worker_hand_input.put(image_np)
    boxes, scores = worker_hand_output.get()

    score = scores[0]
    if score > cap_params['hand_score_thresh']:
        left, right, top, bottom = detector_utils.get_box_coords(boxes, 0, cap_params)
        ring.append(left + top)
        hand_movement = np.std(np.array(ring.get()))
        if hand_movement < cap_params['hand_movement_thresh']:
            crops = detector_utils.get_touched_letter(cap_params, scores, boxes, image_np, args)

            if crops is not None:
                for crop in crops:
                    if flag_capture_frames:  # Enter
                        frames += 1
                        millis = int(round(time.time() * 1000))
                        img = Image.fromarray(np.uint8(crop))
                        img.save(directory + '/images/' + expected_letter + '/' + expected_letter + '_' + str(
                            millis) + '.png', 'png')

                    # Submit the cropped area near a hand to the letter detector queue
                    worker_letter_input.put((crop, 0.999))
                    out = worker_letter_output.get()

                    if out != '' and out != last_letter:
                        return out


def get_another_word(success=True):
    global t, word_to_spell, total_points, last_letters, iterations, correct_word_count, correct_letter_count
    elapsed = str(int(time.time() - t))

    if success:
        say(random.choice(word_accolades))
        say(str(len(word_to_spell)) + " points.")
        total_points += len(word_to_spell)
        correct_word_count += 1

    if args.debug:
        print(elapsed + " seconds elapsed. You have " + str(total_points) + " points.")

    last_letters = []
    iterations = 0
    correct_letter_count = 0
    say(random.choice(thinking))
    t = time.time()

    min_length = args.length + int(correct_word_count / 3)
    return get_word_to_spell(min_length)


def get_random_letter():
    expected_letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return expected_letter


def terminate(e=False):
    if e:
        print(e.__class__.__name__)
        print_exc()

    pool_worker_letters.terminate()
    pool_worker_hands.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()


def handle_keys():
    global frames, flag_capture_frames

    if frames == 50:  # frames of training data to capture when you press Enter
        print('Captured ' + str(frames) + ' frames for ' + expected_letter)
        frames = 0
        flag_capture_frames = False

    if res == 13:  # Enter
        flag_capture_frames = True
    elif res == 27:  # Escape
        terminate()
        return True

    return False


def handle_correct_letter():
    global correct_letter_count, last_letter, iterations, flag_uhoh, flag_capture_frames, frames
    say(detected_letter.lower())
    say(random.choice(accolades))
    correct_letter_count += 1
    last_letters.append(detected_letter)
    last_letter = detected_letter
    iterations = 0
    flag_uhoh = False
    flag_capture_frames = False
    frames = 0


def handle_wrong_letter():
    global flag_uhoh
    say(detected_letter.lower())
    say(random.choice(letter_criticisms))
    flag_uhoh = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alphabot - a screen-less interactive spelling primer powered by computer vision. " +
                                                 "This program comes with ABSOLUTELY NO WARRANTY. This is free software, " +
                                                 "and you are welcome to redistribute it under certain conditions. The suggested " +
                                                 "alphabet letter set is \"Melissa & Doug 52 Wooden Alphabet Magnets in a Box\". " +
                                                 "Copyright (C) 2018  Drew Gillson <drew.gillson@gmail.com>")
    parser.add_argument('--input', default=0, type=int, help="OpenCV device id for your top-mounted camera, default is 0")
    parser.add_argument('--width', default=800, type=int, help="Video width, default is 800")
    parser.add_argument('--height', default=600, type=int, help="Video height, default is 600")
    parser.add_argument('--length', default=3, type=int, help="Default starting character length for words, default is 3")
    parser.add_argument('--debug', default=True, type=bool, help="Print debug information and display input to the letter detector CNN using imshow, default is True")
    parser.add_argument('--corpus', default="brown_words", type=str, help="The CSV file in corpus/ containing words you will be prompted to spell, default is corpus/brown_words.csv")
    parser.add_argument('--tts', default="say", type=str, help="Text-to-speech binary, default is Mac OS \"say\" command")
    parser.add_argument('--mode', default="words", type=str, help="Options are \"words\", \"letters\", or \"train\", default is \"words\"")
    args = parser.parse_args()
    parser.print_help()

    try:
        video_capture = WebcamVideoStream(src=args.input,
                                          width=args.width,
                                          height=args.height).start()
        if not video_capture:
            raise Exception("Camera with device id " + args.input + "could not be initialized")

        cap_params = {}
        cap_params['im_width'], cap_params['im_height'] = video_capture.size()
        cap_params['num_hands_detect'] = 1
        cap_params['hand_score_thresh'] = 0.38
        cap_params['hand_movement_thresh'] = 40

        if args.mode == "train":
            # If you find that certain letters are not being recognized, or if you are using a different letter set
            # than I am, you'll have to capture your own training data and retrain the CNN. You can capture training
            # data frames by pressing Enter at any time and frames will be written to images/[A-Z]/
            letter_utils.train()
        else:
            words = read_from_csv(args.corpus)
            predictor = trie.TrieNode('*')
            for word in words:
                trie.add(predictor, word[0])
            # print(trie.find_prefix(predictor, 'prefix'))

            # You can modify the default accolades and criticisms by modifying the CSV files in corpus/
            accolades = read_from_csv('accolades')
            word_accolades = read_from_csv('word_accolades')
            letter_criticisms = read_from_csv('letter_criticisms')
            thinking = read_from_csv('thinking')

            directory = os.path.dirname(os.path.realpath(__file__))
            iterations = 0
            last_letters = []
            last_letter = ''
            expected_letter = ''
            detected_letter = ''
            correct_letter_count = 0
            correct_word_count = 0
            frames = 0
            flag_capture_frames = False
            flag_uhoh = False
            t = time.time()
            total_points = 0
            crop = np.ones((28, 28), dtype=np.uint8)
            ring = RingBuffer(10)
            worker_hand_input = Queue(maxsize=5)
            worker_hand_output = Queue(maxsize=5)
            pool_worker_hands = Pool(4, worker_hands, (worker_hand_input, worker_hand_output))
            worker_letter_input = Queue(maxsize=5)
            worker_letter_output = Queue(maxsize=5)
            pool_worker_letters = Pool(2, worker_letters, (worker_letter_input, worker_letter_output))

            while True:
                iterations += 1
                res = cv2.waitKey(80)
                image_np = cv2.flip(video_capture.read(), -1)
                cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

                detected_letter = get_touched_letter() or ''

                if args.mode == "words":
                    if 'word_to_spell' not in locals():
                        word_to_spell, sentence = get_word_to_spell(args.length)

                    # letter buffer = the last N touched letters where N is the length of the word to spell:
                    letter_buffer = ''.join(last_letters[-len(word_to_spell):])
                    if letter_buffer == word_to_spell:
                        word_to_spell, sentence = get_another_word()
                        continue
                    else:
                        expected_letter = word_to_spell[correct_letter_count]

                        if last_letter != detected_letter and detected_letter == expected_letter:
                            handle_correct_letter()
                        elif detected_letter != '' and not flag_uhoh and iterations > 0:
                            handle_wrong_letter()

                        if iterations % 50 == 0 and iterations != 0:
                            say('The word is ' + word_to_spell)
                        elif iterations == 299:
                            say('Ok, we will spell ' + word_to_spell + ' some other time. You have ' + str(
                                total_points) + ' points.')
                            word_to_spell, sentence = get_another_word(success=False)
                elif args.mode == "letters":

                    # if you spell PLAY then go to words mode

                    if not flag_capture_frames and iterations == 1:
                        expected_letter = get_random_letter()
                        say('Can you find the ' + expected_letter + '?')

                    say(detected_letter.lower())
                    if expected_letter == detected_letter:

                        # give random short example word and alliterative sentence

                        say(random.choice(accolades))
                        iterations = 0

                if handle_keys():
                    break
    except Exception as e:
        terminate(e)