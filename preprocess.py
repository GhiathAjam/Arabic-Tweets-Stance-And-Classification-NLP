import string
import re
import camel_tools.utils.dediac as camel_utils_dediac
import camel_tools.utils.normalize as camel_utils_normalize
import camel_tools.tokenizers.word as camel_utils_tokenizer
from camel_tools.disambig.mle import MLEDisambiguator
# from camel_tools.morphology.analyzer import Analyzer
# from camel_tools.morphology.database import MorphologyDB

import nltk
import arabicstopwords.arabicstopwords as stp

# SPECIAL TOKENS:
# <LINK> <NUM> <Mt> <LF> for links, numbers, mentions, line feed
# ENGLISH Text is kept as is
# HASHTAGS are repeated 3 times
# TODO: Emojis

class Preprocess:

    HASH_FREQ = 3
    mle = MLEDisambiguator.pretrained()

    nltk_arb_stopwords = set(nltk.corpus.stopwords.words("arabic"))
    arabicstopwords = set(stp.stopwords_list())

    STOPWORDS = arabicstopwords

    # def __init__(self, HASH_FREQ = 3) -> None:
    #     self.HASH_FREQ = HASH_FREQ
    #     self.mle = MLEDisambiguator.pretrained()
    #     self.STOPWORDS = self.arabicstopwords
    #     pass

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
        ret = re.sub(r'@\S+', ' <Mt> ', ret)
        ret = re.sub('<LF>', ' <LF> ', ret)
        return ret

    # string -> remove punctuation
    def remove_punctuation(self, text):
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~''' + '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ«»'''
        # keep HashTags ?
        punc = punc.replace('#', '')
        punc = punc.replace('_', '')
        # keep tokens: <...>
        punc = punc.replace('<', '')
        punc = punc.replace('>', '')
        # replace punc with space
        return text.translate(str.maketrans(punc, ' ' * len(punc)))

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
        # disambig = self.mle.disambiguate(tokenized_text)
        # lemmas = [d.analyses[0].analysis['lex'] for d in disambig]
        # return lemmas
        return tokenized_text

    def frasa_lemmatize(self, tokenized_text):
        return tokenized_text

    # Lemmatization
    def lemmatize(self, tokenized_text):
        # Camel
        cl = self.camel_lemmatize(tokenized_text)
        # Frasa
        fr = self.frasa_lemmatize(tokenized_text)
        
        # print(tokenized_text)
        # print("Camel: ", cl)
        # print("Frasa: ", fr)
        return tokenized_text

    # After lemmas?
    def remove_stopwords(self, tokenized_text):
        ret = [tk for tk in tokenized_text if tk not in self.STOPWORDS]
        return ret

    def do_all(self, text):
        return self.remove_stopwords(self.lemmatize(self.tokenizer(self.normalize(self.remove_punctuation(self.tokens(self.dediac(text)))))))


p = Preprocess()

# print(camel_utils_normalize.normalize_unicode('ميسبت'))

# print(p.dediac('asd'))

# print(p.do_all('''بيل غيتس يتلقى لقاح #كوفيد19 من غير تصوير الابرة و لا السيرنجة و لا الدواء و لابس بولو صيفي في عز الشتاء و يقول ان إحدى مزايا عمر ال 65 عامًا هي انه مؤهل للحصول على اللقاح ... يعنى ما كان يحتاج اللقاح لو كان عمره اصغر من 65 🤔 https://t.co/QQKFFUNwBn,celebrity,1وزير الصحة لحد اليوم وتحديدا هلأ بمؤتمروا الصحفي كان ما عندي مشكلة معوا.<LF>بس انوا بفايزر هو اللقاح الوحيد الآمن بالعالم لأ حبيبي هاي فيها نفاق واضح.<LF>عفوا يعني.<LF>مش إنوا مجبورين فيه وما قادر تجبر الدولة المارقة تدفع مصاري تجيب لقاح تاني.<LF>يلا باي,info_news,1قولكن  رح يكونو اد المسؤولية ب لبنان لما يوصل اللقاح؟<LF>ألفي جرعة من لقاح #كورونا للتلف في مستشفى بوسطن،لأن عامل النظافة وقف عن طريق الخطأ عمل الثلاجة التي حفظت بها.<LF>#لقاح_كورونا,info_news,1'''))

print(p.do_all('''مات 23 شخص في النرويج بعد تلقي لقاح  #COVID19..<LF>نيويورك بوست 🤦 https://t.co/g51cmWeoVL,info_news,-1'''))