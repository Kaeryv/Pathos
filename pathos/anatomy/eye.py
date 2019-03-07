from scipy.spatial import distance as dist

class Eye(object):
    """description of class"""

    def __init__(self, landmarks, *args, **kwargs):
        self._landmarks = landmarks
        return super().__init__(*args, **kwargs)

    def aspect_ratio(self):
        vertical_dist_1 = dist.euclidean(self._landmarks[1], self._landmarks[5])
        vertical_dist_2 = dist.euclidean(self._landmarks[2], self._landmarks[4])
        horizontal_dist = dist.euclidean(self._landmarks[0], self._landmarks[3])

        ear = (vertical_dist_1 + vertical_dist_2) / 2.0 / horizontal_dist

        return ear



    def landmark(self, index):
        return self._landmarks[index]