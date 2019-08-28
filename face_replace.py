import face_recognition
from PIL import Image, ImageDraw
import cv2
import numpy as np
import ffmpeg
import math
import os


# some variables
INPUT_FILE = 'example/horns.mp4'  # input video file
SEARCH_FILE = 'example/search.jpg'  # image of the face to recognise and replace
REPLACE_FILE = 'example/replace.png'  # transparent png to use as the face replacement
OUTPUT_FILE = 'output.mp4'  # output video file

DETECTION_THRESH = 0.68  # threshold for face detection - 0.6 is the library default
UPSAMPLES = 2  # how many times to upscale the source image by - larger number finds smaller faces
SCALE_FACTOR = 1.5  # how much to scale the replacement face by


def draw_image(src, dest, src_landmarks, dest_landmarks, factor=1.0):
    transformed = src.copy()

    src_center = src_landmarks['nose_tip'][0]
    dest_center = dest_landmarks['nose_tip'][0]

    src_size = math.hypot(src_landmarks['right_eye'][1][0] - src_landmarks['left_eye'][0][0], src_landmarks['right_eye'][1][1] - src_landmarks['left_eye'][0][1])
    dest_size = math.hypot(dest_landmarks['right_eye'][1][0] - dest_landmarks['left_eye'][0][0], dest_landmarks['right_eye'][1][1] - dest_landmarks['left_eye'][0][1])
    scale = float(dest_size) / float(src_size)
    scale *= factor

    transformed = transformed.resize((int(src.width * scale), int(src.height * scale)), Image.LANCZOS)
    box = (
        dest_center[0] - int(src_center[0] * scale),
        dest_center[1] - int(src_center[1] * scale),
        transformed.width + dest_center[0] - int(src_center[0] * scale),
        transformed.height + dest_center[1] - int(src_center[1] * scale),
    )
    dest.paste(transformed, box, transformed)


search_picture = face_recognition.load_image_file(SEARCH_FILE)
search_encodings = face_recognition.face_encodings(search_picture)
if len(search_encodings) != 1:
    print("Search image should contain 1 face. Found {}.".format(len(search_encodings)))
    exit(2)

search_encoding = search_encodings[0]

replace_picture = face_recognition.load_image_file(REPLACE_FILE, mode='RGB')
replace_locations = face_recognition.face_locations(replace_picture, model="cnn")
if len(replace_locations) != 1:
    print("Replace image should contain 1 face. Found {}.".format(len(replace_locations)))
    exit(3)

replace_image = Image.open(REPLACE_FILE).convert('RGBA')

replace_landmarks = face_recognition.face_landmarks(replace_picture, [replace_locations[0]], model='small')[0]

input_movie = cv2.VideoCapture(INPUT_FILE)
frame_count = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_movie = cv2.VideoWriter('temp.mp4', fourcc, input_movie.get(cv2.CAP_PROP_FPS), (int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH)), int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))))


for index in range(frame_count):
    success, frame = input_movie.read()
    if not success:
        print("DEATH CAME TOO SOON! :'(")
        exit(1)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    d = ImageDraw.Draw(image)

    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=UPSAMPLES, model="cnn")
    face_landmarks = face_recognition.face_landmarks(frame, face_locations, model='small')

    if len(face_locations) > 0:
        encodings = face_recognition.face_encodings(frame, known_face_locations=face_locations, num_jitters=10)
        if len(encodings) != len(face_locations):
            print("That was unexpected.")
            exit(4)

        distances = []

        for face_index in range(len(face_locations)):
            distances.append(face_recognition.face_distance([search_encoding], encodings[face_index])[0])

        best_index = np.argmin(distances)
        if distances[best_index] < DETECTION_THRESH:
            draw_image(replace_image, image, replace_landmarks, face_landmarks[best_index], SCALE_FACTOR)

        print("I found {}/{} face(s) in frame {}/{}.".format((1 if distances[best_index] < DETECTION_THRESH else 0), len(face_locations), index, frame_count))
    else:
        print("I found 0 faces in frame {}/{}".format(index, frame_count))

    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    output_movie.write(frame)

input_movie.release()
output_movie.release()

audio_input = ffmpeg.input(INPUT_FILE)
video_input = ffmpeg.input('temp.mp4')

(
    ffmpeg
    .output(video_input.video, audio_input.audio, 'output.mp4', codec='copy')
    .overwrite_output()
    .run()
)

os.remove('temp.mp4')
