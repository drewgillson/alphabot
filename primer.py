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
        return np.array(self)


def say(msg):
    if isinstance(msg, list):
        msg = msg[0]

    subprocess.run(["say", msg])


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
    nn = letter_utils.letterNet()

    while True:
        frame, min_certainty = input_q.get()
        letter = ''
        if frame is not None and frame.shape == (28, 28):
            for_pred = frame.reshape(1, 28, 28, 1).astype('float32') / 255
            y_pred = nn.predict(for_pred, batch_size=1)
            certainty = np.amax(y_pred, 1)
            guess = chr(np.argmax(y_pred, 1) + ord('A'))
            print(str(certainty) + ': ' + guess)
            if certainty > min_certainty:
                letter = guess

        output_q.put(letter)


def read_from_csv(file):
    with open('corpus/' + file + '.csv', 'r') as f:
        reader = csv.reader(f)
        contents = [list(filter(None, c)) for c in list(reader)]
        return contents

def getWordToSpell(max_length = 4):
    global words
    time.sleep(2)
    while True:
        word = random.choice(words)
        if len(word[0]) <= max_length:
            return word[0], word[1]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Alphabet Letter Primer")
    parser.add_argument('--input', default=1, type=int)
    parser.add_argument('--width', default=800, type=int) #1280
    parser.add_argument('--height', default=600, type=int) #960
    parser.add_argument('--min_len', default=3, type=int)
    parser.add_argument('--mode', default="spell", type=str)
    args = parser.parse_args()

    video_capture = WebcamVideoStream(src=args.input,
                                      width=args.width,
                                      height=args.height).start()

    cap_params = {}
    cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    cap_params['score_thresh'] = 0.4
    cap_params['num_hands_detect'] = 1

    if args.mode == "train":
        letter_utils.train()

    elif args.mode == "spell":

        accolades = read_from_csv('accolades')
        word_accolades = read_from_csv('word_accolades')
        letter_criticisms = read_from_csv('letter_criticisms')
        thinking = read_from_csv('thinking')
        words = read_from_csv('words')
        word_to_spell, sentence = getWordToSpell()

        directory = os.path.dirname(os.path.realpath(__file__))
        iterations = 0
        last_letters = []
        last_letter = ''
        expected_letter = ''
        detected_letter = ''
        correct_letter_count = 0
        correct_word_count = 0
        frames = 0
        flag_waiting = False
        flag_capture_frames = False
        flag_uhoh = False
        t = time.time()
        total_points = 0
        crop = np.ones((28, 28), dtype=np.uint8)
        FRAMES_TO_CAPTURE = 10

        ring = RingBuffer(30)

        worker_hand_input = Queue(maxsize=5)
        worker_hand_output = Queue(maxsize=5)
        pool_worker_hands = Pool(4, worker_hands, (worker_hand_input, worker_hand_output))

        worker_letter_input = Queue(maxsize=5)
        worker_letter_output = Queue(maxsize=5)
        pool_worker_letters = Pool(1, worker_letters, (worker_letter_input, worker_letter_output))

        while True:
            res = cv2.waitKey(80)
            image_np = video_capture.read()
            image_np = cv2.flip(image_np, -1)
            cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            ring.append(np.mean(image_np))
            samples = ring.get()

            print("     " +str(np.std(samples)))
            if np.std(samples) < 0.015:

                worker_hand_input.put(image_np)
                boxes, scores = worker_hand_output.get()

                crops = detector_utils.get_touched_letter(cap_params['num_hands_detect'], cap_params["score_thresh"], scores, boxes, cap_params['im_width'], cap_params['im_height'], image_np)

                if crops is not None:
                    for crop in crops:
                        worker_letter_input.put((crop, 0.9998))
                        out = worker_letter_output.get()

                        if out != '' and out != last_letter:
                            detected_letter = out
                            break

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

                flag_waiting = False
                last_letters = []
                iterations = 0
                correct_word_count += 1
                correct_letter_count = 0
                say(random.choice(thinking))
                word_to_spell, sentence = getWordToSpell()
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
                crop.save(directory + '/images/' + expected_letter + '/' + expected_letter + '_' + str(millis) + '.png','png')

            if frames == FRAMES_TO_CAPTURE:
                msg = 'Captured ' + str(frames) + ' frames for ' + expected_letter
                print(msg)
                frames = 0
                flag_capture_frames = False

            if flag_waiting == False:
                say('Can you spell ' + word_to_spell + '? As in: "' + sentence + '"')
                flag_waiting = True

            if iterations % 800 == 0 and iterations != 0:
                say('The word is ' + word_to_spell)

            if iterations == 2000:
                say('Ok, we will spell ' + word_to_spell + ' some other time. You have ' + str(
                    total_points) + ' points.')
                flag_waiting = False
                last_letters = []
                correct_letter_count = 0
                word_to_spell, sentence = getWordToSpell()

            if len(last_letters) > 0:
                last_letter = last_letters[-1]

            if res == 27:  # Escape
                pool_worker_letters.terminate()
                video_capture.stop()
                cv2.destroyAllWindows()
                break
