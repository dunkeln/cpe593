import unittest

from ..lib.astarfibon import getNeighbors

class testUtils(unittest.TestCase):
    
    def testneighbors(self):
        f_neighbors = getNeighbors(100, 100)
        self.assertListEqual(f_neighbors(0, 0), [(1, 1)], "Incorrect neighbors listed")

if __name__ == "__main__":
    unittest.main()
