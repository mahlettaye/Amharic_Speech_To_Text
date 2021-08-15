import unittest

import sys
sys.path.insert(0, '/home/mahlet/10ac/Amharic_Speech_To_Text/scripts/')

from data_processor. AudioGenerator import load_test_data

class TestSum(unittest.TestCase):
    def test_list_int(self):
        """
        Test that it can sum a list of integers
        """
        
        result = sum(3,3)
        self.assertEqual(result, 6)

if __name__ == '__main__':
    # to run test terminal python3 -m unittest discover -v -s . -p "*Test_*.py"
    unittest.main()
