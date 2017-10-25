from exceptions import KeyError


class Bucketizer(object):
    def __init__(self):
        pass

    def _transform(self, feature):
        left = 0
        right = len(self.splits) - 1
        while left < right:
            mid = left + (right - left)/2
            if self.splits[mid] < feature:
                left = mid + 1
            else:
                right = mid
        if self.splits[left] > feature:
            return left - 1
        else:
            return left
