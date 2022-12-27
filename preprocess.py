import string
import re
import camel_tools.utils.dediac as camel_utils_dediac
import camel_tools.utils.normalize as camel_utils_normalize
import camel_tools.tokenizers.word as camel_utils_tokenizer
from camel_tools.disambig.mle import MLEDisambiguator
from utils import UNICODE_EMO

# from camel_tools.morphology.analyzer import Analyzer
# from camel_tools.morphology.database import MorphologyDB

from farasa.segmenter import FarasaSegmenter
from farasa.stemmer import FarasaStemmer

from nltk.stem.isri import ISRIStemmer

import nltk
import arabicstopwords.arabicstopwords as stp

# SPECIAL TOKENS:
# <LINK> <NUM> <Mt> <LF> for links, numbers, mentions, line feed
# ENGLISH Text is kept as is
# HASHTAGS are repeated 3 times

class Preprocess:

    farasa_seg = FarasaSegmenter(interactive=True)
    farasa_stm = FarasaStemmer(interactive=True)

    nltk_arb_stopwords = set(nltk.corpus.stopwords.words("arabic"))
    arabicstopwords = set(stp.stopwords_list())

    def __init__(self, EMOJIS='raw', HASH_FREQ=2, BERT=False) -> None:
        print("Bert MODE", BERT)
        print("Emojis: ", EMOJIS)
        self.EMOJIS = EMOJIS
        self.HASH_FREQ = HASH_FREQ
        self.BERT = BERT
        #
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
        if self.EMOJIS == 'text':
            return "".join([' <'+UNICODE_EMO[c]+'> ' if c in UNICODE_EMO else c for c in text])
        elif self.EMOJIS == 'none':
            return "".join([c if c not in UNICODE_EMO else ' ' for c in text])
        else: # raw
            return "".join([c if c not in UNICODE_EMO else ' '+c+' ' for c in text])

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

    # lowercase all latin latters
    # replacing: أ إ آ with ا
    # replacing: ة with ه
    # replacing: ي with ى
    # Maybe add ئ ؤ ?
    def normalize(self, text):
        ret = text
        ret = camel_utils_normalize.normalize_alef_ar(ret)
        ret = camel_utils_normalize.normalize_teh_marbuta_ar(ret)
        ret = camel_utils_normalize.normalize_alef_maksura_ar(ret)
        # lowercase latins
        ret = ret.lower()
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
        #
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
        if self.BERT:
            return ' '.join([self.dediac(l) for l in lemmas])
        else:
            return [self.dediac(l) for l in lemmas]

    def nltk_lemmatize(self, tokenized_text):
        si = ISRIStemmer()
        return [si.stem(t) for t in tokenized_text]

    # Farasapy is bad
    # This doesn't give same result as their API!
    def farasapy_lemmatize(self, text):
        # extract hashtags
        hashtags = re.findall(r'#(\S+)', text)
        text = re.sub(r'#\S+', '', text)
        # extract < .. >
        tokens = re.findall(r'<\S+>', text)
        text = re.sub(r'<\S+>', '', text)
        #
        if self.BERT:
            ret = self.farasa_stm.stem(text)
            return (ret + ' '.join(tokens) + ' '.join(hashtags))
        else:
            ret = self.tokenizer(text)
            return ret + tokens + hashtags

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
    def do_all(self, text, debug=False):
        #
        ret = self.normalize(self.remove_punctuation(self.convert_emojis_to_meaning(self.tokens(self.dediac(text)))))
        if debug:
            print('0', ret)
        #
        if self.BERT:
            tot = ret
            tot += self.camel_lemmatize(self.tokenizer(ret))
            tot += self.farasapy_lemmatize(ret)
            return tot
        else:
            tot = self.tokenizer(ret)
            if debug:
                print('1', tot)
            tot += self.camel_lemmatize(self.tokenizer(ret))
            if debug:
                print('2', self.camel_lemmatize(self.tokenizer(ret)))
            tot += self.farasapy_lemmatize(ret)
            if debug:
                print('3', self.farasapy_lemmatize(ret))
            return self.remove_stopwords(tot)

# open file

# p = Preprocess(EMOJIS='raw')
# # p = Preprocess(EMOJIS='text')
# p = Preprocess(EMOJIS='none')
# x = 'تساؤلات آخر ليل <LF>اذا ما كفانا لقاح الكورونا ...بزيدولو مي ؟؟ 😅<LF>#هبل'
# print('done', p.do_all(x))

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


