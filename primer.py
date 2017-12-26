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
    nn = letter_utils.letter_net()

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


def get_word_to_spell(max_length=3):
    global words, last_letters, last_letter, correct_letter_count
    time.sleep(2)

    last_letters = []
    last_letter = ''
    correct_letter_count = 0

    while True:
        word = random.choice(words)
        if len(word[0]) <= max_length:
            say('Can you spell ' + word[0] + '? As in: "' + word[1] + '"')
            return word[0].upper(), word[1]


def get_touched_letter():
    global frames

    # Submit the frame to the hand detector queue
    worker_hand_input.put(image_np)
    boxes, scores = worker_hand_output.get()

    score = scores[0]
    #print(str(score) + ': [' + str(left) + ', ' + str(top) + ']')
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
                    worker_letter_input.put((crop, 0.9998))
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

    print(elapsed + " seconds elapsed. You have " + str(total_points) + " points.")
    last_letters = []
    iterations = 0
    correct_letter_count = 0
    say(random.choice(thinking))
    t = time.time()

    max_length = args.length + int(correct_word_count / 3)
    return get_word_to_spell(max_length)


def get_random_letter():
    expected_letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return expected_letter

def handle_keys():
    global frames, flag_capture_frames

    if frames == 50:  # frames of training data to capture when you press Enter
        print('Captured ' + str(frames) + ' frames for ' + expected_letter)
        frames = 0
        flag_capture_frames = False

    if res == 13:  # Enter
        flag_capture_frames = True
    elif res == 27:  # Escape
        pool_worker_letters.terminate()
        video_capture.stop()
        cv2.destroyAllWindows()
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
    parser = argparse.ArgumentParser(description="Alphabet Letter Primer")
    parser.add_argument('--input', default=1, type=int)
    parser.add_argument('--width', default=800, type=int) #1280
    parser.add_argument('--height', default=600, type=int) #960
    parser.add_argument('--length', default=6, type=int)
    parser.add_argument('--show', default=False, type=bool)
    parser.add_argument('--mode', default="letters", type=str)
    args = parser.parse_args()

    video_capture = WebcamVideoStream(src=args.input,
                                      width=args.width,
                                      height=args.height).start()

    cap_params = {}
    cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    cap_params['num_hands_detect'] = 1
    cap_params['hand_score_thresh'] = 0.4
    cap_params['hand_movement_thresh'] = 40

    if args.mode == "train":
        letter_utils.train()
    else:
        words = read_from_csv('words')
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
                    elif iterations == 200:
                        say('Ok, we will spell ' + word_to_spell + ' some other time. You have ' + str(
                            total_points) + ' points.')
                        word_to_spell, sentence = get_another_word(success=False)
            elif args.mode == "letters":
                if not flag_capture_frames and iterations == 1:
                    expected_letter = get_random_letter()
                    say('Can you find the ' + expected_letter + '?')

                say(detected_letter.lower())
                if expected_letter == detected_letter:
                    say(random.choice(accolades))
                    iterations = 0

            if handle_keys():
                break
