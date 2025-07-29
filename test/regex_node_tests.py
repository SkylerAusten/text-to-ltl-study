import unittest
from regex_node import RegexNode


class TestRegexNode(unittest.TestCase):

    def test_accept(self):
        r = RegexNode("a(b|c)*")
        self.assertTrue(r.accepts("a"))
        self.assertTrue(r.accepts("ab"))
        self.assertTrue(r.accepts("acbc"))

        self.assertFalse(r.accepts("b"))
        self.assertFalse(r.accepts("abcabcabcabz"))

    def test_equivalence(self):
        r1 = RegexNode("(foot)|(feet)")
        r2 = RegexNode("f(oo|ee)t")

        self.assertTrue(r1 == r2)

    def test_union_disjoint(self):
        r1 = RegexNode("a+")
        r2 = RegexNode("b+")

        result = r1.union(r2)

        self.assertTrue(result.accepts("aaa"))
        self.assertTrue(result.accepts("bbb"))

        self.assertFalse(result.accepts("ab"))

    def test_union_overlap(self):
        r1 = RegexNode("ab*")
        r2 = RegexNode("a+")
        result = r1.union(r2)

        self.assertTrue(result.accepts("a"))
        self.assertTrue(result.accepts("abbb"))
        self.assertTrue(result.accepts("aaaaa"))
        self.assertFalse(result.accepts("b"))

    def test_difference(self):
        r1 = RegexNode("a*")
        r2 = RegexNode("aaa")

        diff = r1.difference(r2)

        self.assertTrue(diff.accepts(""))
        self.assertTrue(diff.accepts("a"))
        self.assertTrue(diff.accepts("aa"))
        self.assertFalse(diff.accepts("aaa"))
        self.assertTrue(diff.accepts("aaaa"))

    def test_intersect_overlap(self):
        r1 = RegexNode("a+")
        r2 = RegexNode("(aa)*")

        result = r1.intersect(r2)

        self.assertTrue(result.accepts("aa"))
        self.assertTrue(result.accepts("aaaa"))
        self.assertFalse(result.accepts("a"))
        self.assertFalse(result.accepts("aaa"))

    def test_intersect_disjoint(self):
        r1 = RegexNode("a+")
        r2 = RegexNode("b+")

        result = r1.intersect(r2)

        self.assertFalse(result.accepts("a"))
        self.assertFalse(result.accepts("b"))
        self.assertFalse(result.accepts("ab"))
        self.assertFalse(result.accepts(""))

    def test_difference_disjoint(self):
        r1 = RegexNode("a+")
        r2 = RegexNode("b+")

        diff = r1.difference(r2)

        self.assertTrue(diff.accepts("a"))
        self.assertTrue(diff.accepts("aaa"))
        self.assertFalse(diff.accepts("b"))

    def test_minimum_word_length(self):
        r = RegexNode("abc|de")
        self.assertEqual(r.minimum_word_length(), 2)  # "de" is shortest

    def test_maximum_word_length(self):
        r = RegexNode("abc|de")
        self.assertEqual(r.maximum_word_length(), 3)  # "abc" is longest

    def test_maximum_word_length_infinite(self):
        r = RegexNode("a+")
        self.assertIsNone(r.maximum_word_length())  # Repetition → infinite

    # def test_generate_sequential_words(self):
    #     r = RegexNode("a{1,3}")
    #     words = r.generate_sequential_words(num_items=5)
    #     self.assertIn("a", words)
    #     self.assertIn("aa", words)
    #     self.assertIn("aaa", words)
    #     self.assertNotIn("", words)
    #     self.assertEqual(len(words), 3)

    # def test_generate_random_words(self):
    #     r = RegexNode("a{2,4}")
    #     words = r.generate_random_words(num_words=10)
    #     for w in words:
    #         self.assertTrue(r.accepts(w))
    #         self.assertGreaterEqual(len(w), 2)
    #         self.assertLessEqual(len(w), 4)
    #     self.assertEqual(len(set(words)), len(words))  # All unique

    # def test_generate_random_word(self):
    #     r = RegexNode("ab|cd")
    #     word = r.generate_random_word(2, seed=123)
    #     self.assertIn(word, ["ab", "cd"])

    def test_reduce(self):
        r1 = RegexNode("(a|a)")
        r2 = r1.reduce()
        self.assertTrue(r2.accepts("a"))
        self.assertEqual(r2.minimum_word_length(), 1)
        self.assertTrue(r1 == r2)  # Language is the same

    def test_study_examples(self):
        mtime_regex = RegexNode("((0?\d)|(1\d)|(2[0-3])):([0-5]\d)")
        stime_regex = RegexNode("((0?[1-9])|(1[0-2])):([0-5]\d)(\s?((A|a|P|p)(M|m)))?")
        time_regex = mtime_regex.union(stime_regex)

        # Test wildly wrong examples.
        self.assertFalse(time_regex.accepts("abcdxyz"))
        self.assertFalse(time_regex.accepts("12312"))
        self.assertFalse(time_regex.accepts("aaaaammm"))
        self.assertFalse(time_regex.accepts("1?:40"))
        self.assertFalse(time_regex.accepts("*"))
        self.assertFalse(time_regex.accepts(" "))
        self.assertFalse(time_regex.accepts("."))

        # Test valid symbols in an invalid order.
        self.assertFalse(time_regex.accepts("1230"))
        self.assertFalse(time_regex.accepts("1:300"))
        self.assertFalse(time_regex.accepts("AM"))
        self.assertFalse(time_regex.accepts("am"))
        self.assertFalse(time_regex.accepts("PM"))
        self.assertFalse(time_regex.accepts("pm"))
        self.assertFalse(time_regex.accepts(":1230"))
        self.assertFalse(time_regex.accepts("am12:00"))
        self.assertFalse(time_regex.accepts("P4:00M"))
        self.assertFalse(time_regex.accepts("220:00"))
        self.assertFalse(time_regex.accepts("111:00pm"))
        self.assertFalse(time_regex.accepts("111:00pm"))
        self.assertFalse(time_regex.accepts("01"))
        self.assertFalse(time_regex.accepts("13"))
        self.assertFalse(time_regex.accepts("22"))
        self.assertFalse(time_regex.accepts("8pm"))
        self.assertFalse(time_regex.accepts("12AM"))

        # Test close-but-invalid examples.
        self.assertFalse(time_regex.accepts("13:00pm"))
        self.assertFalse(time_regex.accepts("08:71PM"))
        self.assertFalse(time_regex.accepts("72:51aM"))
        self.assertFalse(time_regex.accepts("3:60"))
        self.assertFalse(time_regex.accepts("03:72"))
        self.assertFalse(time_regex.accepts("30:15"))
        self.assertFalse(time_regex.accepts("24:00"))
        self.assertFalse(time_regex.accepts("22::00"))
        self.assertFalse(time_regex.accepts("001:00Pm"))
        self.assertFalse(time_regex.accepts("1:00ppm"))
        self.assertFalse(time_regex.accepts("3:01 "))
        self.assertFalse(time_regex.accepts(" 3:01 "))
        self.assertFalse(time_regex.accepts("013:00"))

        # Test valid examples.
        self.assertTrue(time_regex.accepts("12:00Pm"))
        self.assertTrue(time_regex.accepts("12:00am"))
        self.assertTrue(time_regex.accepts("1:00pm"))
        self.assertTrue(time_regex.accepts("01:00pM"))
        self.assertTrue(time_regex.accepts("8:57aM"))
        self.assertTrue(time_regex.accepts("2:46am"))
        self.assertTrue(time_regex.accepts("10:10AM"))
        self.assertTrue(time_regex.accepts("11:11PM"))
        self.assertTrue(time_regex.accepts("1:02 pm"))
        self.assertTrue(time_regex.accepts("2:13 am"))
        self.assertTrue(time_regex.accepts("3:24 Am"))
        self.assertTrue(time_regex.accepts("4:35 aM"))
        self.assertTrue(time_regex.accepts("5:46 Pm"))
        self.assertTrue(time_regex.accepts("6:57 pM"))
        self.assertTrue(time_regex.accepts("7:08 AM"))
        self.assertTrue(time_regex.accepts("8:19 PM"))
        self.assertTrue(time_regex.accepts("1:10"))
        self.assertTrue(time_regex.accepts("2:03"))
        self.assertTrue(time_regex.accepts("3:26"))
        self.assertTrue(time_regex.accepts("4:47"))
        self.assertTrue(time_regex.accepts("7:51"))
        self.assertTrue(time_regex.accepts("00:00"))
        self.assertTrue(time_regex.accepts("00:01"))
        self.assertTrue(time_regex.accepts("10:01"))
        self.assertTrue(time_regex.accepts("03:00"))
        self.assertTrue(time_regex.accepts("03:46"))
        self.assertTrue(time_regex.accepts("13:00"))
        self.assertTrue(time_regex.accepts("22:00"))


if __name__ == "__main__":
    unittest.main()
