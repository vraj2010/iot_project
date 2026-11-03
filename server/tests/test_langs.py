import unittest

import _path  # noqa: F401
import langs
from brain import SentenceChunker, clean_for_speech
from stt import clean_transcript
from tts import split_first_sentence


class LangTests(unittest.TestCase):
    def setUp(self):
        langs.set_turn("en")

    def test_voice_follows_script(self):
        self.assertIn("hi-IN", langs.voice_for("पनीर की सब्ज़ी बनाने के लिए प्याज़ काटिए।"))
        self.assertIn("gu-IN", langs.voice_for("સ્ટેશન ક્યાં છે?"))
        self.assertNotIn("-IN-", langs.voice_for("Where is the station?"))

    def test_marathi_shares_devanagari_with_hindi(self):
        self.assertIn("hi-IN", langs.voice_for("स्टेशन कुठे आहे"))      # no turn language: Hindi
        langs.set_turn("mr")
        self.assertIn("mr-IN", langs.voice_for("स्टेशन कुठे आहे"))

    def test_few_foreign_words_stay_in_language(self):
        self.assertNotIn("-IN-", langs.voice_for("Open the Wi-Fi settings and tap the कुंजी"))
        self.assertIn("hi-IN", langs.voice_for("पहले onion को काटिए और फिर पनीर डालिए।"))

    def test_set_reply_language(self):
        self.assertTrue(langs.set_reply_language("Marathi")["ok"])
        self.assertEqual(langs.turn(), "mr")
        self.assertIn("error", langs.set_reply_language("Klingon"))

    def test_transcript_in_indian_script_is_not_dropped(self):
        self.assertEqual(clean_transcript("मुझे पनीर की सब्ज़ी की रेसिपी दो"), "मुझे पनीर की सब्ज़ी की रेसिपी दो")

    def test_danda_splits_sentences(self):
        c = SentenceChunker()
        self.assertEqual(c.feed("पहले प्याज़ काटिए। फिर पनीर डालिए। "), ["पहले प्याज़ काटिए।", "फिर पनीर डालिए।"])
        first, rest = split_first_sentence("पहले प्याज़ काटिए। फिर पनीर डालिए।")
        self.assertEqual(first, "पहले प्याज़ काटिए।")
        self.assertEqual(rest, "फिर पनीर डालिए।")
        self.assertEqual(clean_for_speech("**पनीर** डालिए।"), "पनीर डालिए।")


if __name__ == "__main__":
    unittest.main()
