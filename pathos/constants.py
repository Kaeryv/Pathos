from collections import OrderedDict

CASC_PATH = './haarcascade_files/haarcascade_frontalface_default.xml'
INPUT_SIZE = 48
SENSOR_SIZE = 48
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
SAVE_DIRECTORY = './db/'
MODEL_ARCHIVE_NAME = "test1"
DATASET_CSV_FILENAME = 'fer2013.csv'
SAVE_DATASET_IMAGES_FILENAME = 'affectnet.pictures.training.npy'
SAVE_DATASET_LABELS_FILENAME = 'affectnet.labels.training.npy'

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])

EMOTIONS = ['ANG', 'DIS', 'FEA', 'HAP', 'SAD', 'SUR', 'NTR']
COLORS = [(255, 0, 0), (150, 0, 150), (200, 200, 250), (255, 255, 0), (50, 50, 50), (0, 255, 255), (0, 255, 255)]
