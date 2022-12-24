import string
import re
import camel_tools.utils.dediac as camel_utils_dediac
import camel_tools.utils.normalize as camel_utils_normalize
import camel_tools.tokenizers.word as camel_utils_tokenizer
from camel_tools.disambig.mle import MLEDisambiguator
from utils import UNICODE_EMO

# from camel_tools.morphology.analyzer import Analyzer
# from camel_tools.morphology.database import MorphologyDB

# from farasa.segmenter import FarasaSegmenter
# from farasa.stemmer import FarasaStemmer

import nltk
import arabicstopwords.arabicstopwords as stp

# SPECIAL TOKENS:
# <LINK> <NUM> <Mt> <LF> for links, numbers, mentions, line feed
# ENGLISH Text is kept as is
# HASHTAGS are repeated 3 times
# TODO: Emojis

class Preprocess:

    # farasa_seg = FarasaSegmenter(interactive=True)
    # farasa_stm = FarasaStemmer(interactive=True)

    nltk_arb_stopwords = set(nltk.corpus.stopwords.words("arabic"))
    arabicstopwords = set(stp.stopwords_list())

    def __init__(self, INCLUDE_EMOJIS=True, HASH_FREQ=2, LEMMATIZOR='camel') -> None:
        print("Emojis: ", INCLUDE_EMOJIS)
        print("Lemmatizor: ", LEMMATIZOR)
        self.INCLUDE_EMOJIS = INCLUDE_EMOJIS
        self.LEMMATIZOR = LEMMATIZOR
        self.HASH_FREQ = HASH_FREQ
        self.mle = MLEDisambiguator.pretrained()
        # Empty tokens as stop words
        self.STOPWORDS = self.arabicstopwords.union(['', ' '])

    # string -> dediacritized string
    # print(dediac("ميسبت"));
    def dediac(self, text):
        # Can also do using regex
        return camel_utils_dediac.dediac_ar(text)

    # replace:
        # 1. links with <LINK>
        # 2. numbers with <NUM>
        # 3. Mentions (@USER) with <Mt>
    def tokens(self, text):
        ret = text
        # all links are https://t.co/...
        # add extra space to avoid joining words
        ret = re.sub(r'http\S+', ' <LINK> ', ret)
        ret = re.sub(r'\d+', ' <NUM> ', ret)
        ret = re.sub(r'\S*@\S*', ' <Mt> ', ret)
        ret = re.sub('<LF>', ' <LF> ', ret)
        return ret

    def convert_emojis_to_meaning(self, text):
        # emojis unicode are '\U000[0-9A-F]{5}'
        # emojis = re.findall(r'\\U000[0-9A-F]{5}', text)
        #
        # map the emoji unicode into its english word
        # remove : , from the english word
        # the emoji word is represented as a single token even if it consists of multiple words
        # i.e 🌹 "red_flower" not "red flower"
        # add space before and after the emoji word, since usually emojis are written close to each other without spaces.
        # i.e. want 🔥🔥🔥 -> "fire fire fire" not "firefirefire"
        return "".join([' <'+UNICODE_EMO[c]+'> ' if c in UNICODE_EMO else c for c in text])

    # string -> remove punctuation
    def remove_punctuation(self, text):
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~''' + '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ«»'''
        punc += '''⁉'''

        # keep HashTags 
        hashtags = re.findall(r'#\S+', text)
        text = re.sub(r'#\S+', '', text)
        # keep tokens: <...>
        tokens = re.findall(r'<\S+>', text)
        text = re.sub(r'<\S+>', '', text)

        # replace punc with space
        text = text.translate(str.maketrans(punc, ' ' * len(punc)))
        # re-add tokens
        return text +' '.join(tokens) +' '.join(hashtags)

    # replacing: أ إ آ with ا
    # replacing: ة with ه
    # replacing: ي with ى
    # Maybe add ئ ؤ ?
    def normalize(self, text):
        ret = text
        ret = camel_utils_normalize.normalize_alef_ar(ret)
        ret = camel_utils_normalize.normalize_teh_marbuta_ar(ret)
        ret = camel_utils_normalize.normalize_alef_maksura_ar(ret)
        # replace تبسم with ت ب س م ?
        # ret = camel_utils_normalize.normalize_unicode(ret)
        return ret

    # English words?         --> kept intact
    # <LINK> <NUM> <Mt> ?    --> made into 3 tokens
    def tokenizer(self, text):
        # TODO: better handling of <LF> ??
        # TODO: check camel_tools.tokenizers.morphological.MorphologicalTokenizer

        # extract < .. >
        e1 = re.findall(r'<\S+>', text)
        # remove < .. >
        text = re.sub(r'<\S+>', '', text)
        # extact # .. _ .. _ ..
        e2 = re.findall(r'#(\S+)', text)
        # split on _
        e2 = [tg.split('_') for tg in e2]
        # flatten
        e2 = [item for sublist in e2 for item in sublist if item != '' and item != ' ']
        # remove # .. _ .. _ ..
        text = re.sub(r'#\S+', '', text)

        ret = camel_utils_tokenizer.simple_word_tokenize(text)
        # add extracted!
        ret += e1
        # repeat hashtags for heavier weight!
        for tg in e2:
            ret += [tg] *self.HASH_FREQ

        return ret

    def camel_lemmatize(self, tokenized_text):
        disambig = self.mle.disambiguate(tokenized_text)
        lemmas = [d.analyses[0].analysis['lex'] for d in disambig]
        return lemmas

    # Farasapy is bad
    # This doesn't give same result as their API!
    def farasapy_lemmatize(self, text):
        ret = self.farasa_stm.stem(text)
        return self.tokenizer(ret)

    def lemmatize(self, text, method='camel'):
        if method == 'camel':
            return self.camel_lemmatize(text)
        elif method == 'farasapy':
            return self.farasapy_lemmatize(text)
        else:
            raise Exception('Invalid method')

    # After lemmas?
    def remove_stopwords(self, tokenized_text):
        ret = [tk for tk in tokenized_text if self.dediac(tk) not in self.STOPWORDS]
        return ret

    # do:
    #    1. dediac
    #    2. replace links, numbers, mentions
    #    3. convert emojis to meaning
    #    4. remove punctuation
    #    5. normalize
    #    6. tokenize & lemmatize
    #    7. remove stopwords
    def do_all(self, text):
        if self.INCLUDE_EMOJIS:
            ret = self.normalize(self.remove_punctuation(self.convert_emojis_to_meaning(self.tokens(self.dediac(text)))))
        else:
            ret = self.normalize(self.remove_punctuation(self.tokens(self.dediac(text))))
        #
        # print('tokens:', ret)
        if self.LEMMATIZOR == 'camel':
            return self.remove_stopwords(self.camel_lemmatize(self.tokenizer(ret)))
        elif self.LEMMATIZOR == 'farasapy':
            return self.remove_stopwords(self.farasapy_lemmatize(ret))
        else:
            raise Exception('Invalid lemmatizor')

# open file

# p = Preprocess()
# x = '''
# لقاح #فايزر/بيونتيك https://t.co/LHlDwaLhby,info_news,1

# train   خبراء صحة صينيون يدعون إلى تعليق استخدام لقاح فايزر/بيونتيك"" و""موديرنا""",info_news,1

# خبر جيد عن فعاليه لقاح فايزر/بيونتيك ضد التحولات الجينيه الجديده التي حدثت لفيروس كورونا وجعلته اكثر قدره على الانتشار وخصوصا الطفره الجديده التي تم رصدها والتي تغير اجزاء من البروتين الشوكي. <LF> https://t.co/cYcEAfcNp5,info_news,1

# بعد وفيات النرويج.. خبراء صحة صينيون يدعون إلى تعليق استخدام لقاح “فايزر/بيونتيك” و”موديرنا” https://t.co/9whK0H6dTQ,info_news,-1

# ثاني لقاح يحصل على ترخيص من وكالة الأدوية الأوروبية، بعد السماح باستخدام لقاح فايزر/بيونتيك في دول الاتحاد الـ27 https://t.co/EID7q81aMx #العربية,info_news,1"""
# '''
# x = 'تساؤلات آخر ليل <LF>اذا ما كفانا لقاح الكورونا ...بزيدولو مي ؟؟ 😅<LF>#هبل'
# print(p.do_all(x))

# f = open("output.txt", "w")
# print("NATIVE:", p.farasapy_lemmatize('يُشار إلى أن اللغة العربية يتحدثها أكثر من 422 مليون نسمة ويتوزع متحدثوها في المنطقة المعروفة باسم الوطن العربي بالإضافة إلى العديد من المناطق الأخرى المجاورة مثل الأهواز وتركيا وتشاد والسنغال وإريتريا وغيرها. وهي اللغة الرابعة من لغات منظمة الأمم المتحدة الرسمية الست.'), file=f)
# print("ALL: ", p.do_all('يُشار إلى أن اللغة العربية يتحدثها أكثر من 422 مليون نسمة ويتوزع متحدثوها في المنطقة المعروفة باسم الوطن العربي بالإضافة إلى العديد من المناطق الأخرى المجاورة مثل الأهواز وتركيا وتشاد والسنغال وإريتريا وغيرها. وهي اللغة الرابعة من لغات منظمة الأمم المتحدة الرسمية الست.'), file=f)

# print(camel_utils_normalize.normalize_unicode('ميسبت'))

# print(p.dediac('asd'))

# print(p.do_all('''بيل غيتس يتلقى لقاح #كوفيد19 من غير تصوير الابرة و لا السيرنجة و لا الدواء و لابس بولو صيفي في عز الشتاء و يقول ان إحدى مزايا عمر ال 65 عامًا هي انه مؤهل للحصول على اللقاح ... يعنى ما كان يحتاج اللقاح لو كان عمره اصغر من 65 🤔 https://t.co/QQKFFUNwBn,celebrity,1وزير الصحة لحد اليوم وتحديدا هلأ بمؤتمروا الصحفي كان ما عندي مشكلة معوا.<LF>بس انوا بفايزر هو اللقاح الوحيد الآمن بالعالم لأ حبيبي هاي فيها نفاق واضح.<LF>عفوا يعني.<LF>مش إنوا مجبورين فيه وما قادر تجبر الدولة المارقة تدفع مصاري تجيب لقاح تاني.<LF>يلا باي,info_news,1قولكن  رح يكونو اد المسؤولية ب لبنان لما يوصل اللقاح؟<LF>ألفي جرعة من لقاح #كورونا للتلف في مستشفى بوسطن،لأن عامل النظافة وقف عن طريق الخطأ عمل الثلاجة التي حفظت بها.<LF>#لقاح_كورونا,info_news,1'''))

# print(p.do_all('''مات 23 شخص في النرويج بعد تلقي لقاح  #COVID19..<LF>نيويورك بوست 🤦 https://t.co/g51cmWeoVL,info_news,-1''', lemmatizor='farasapy'))
# print(p.do_all('''مات 23 شخص في النرويج بعد تلقي لقاح  #COVID19..<LF>نيويورك بوست 🤦 https://t.co/g51cmWeoVL,info_news,-1''', lemmatizor='camel'))

# print(p.convert_emojis_to_meaning("😂😶😶🤔❤👍🏻😁"));
# print("empty test");

