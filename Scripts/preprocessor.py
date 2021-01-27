import nltk
from nltk.corpus import stopwords
from nltk import re
from wordsegment import load, segment
from autocorrect import Speller
import demoji
import string
from nltk.tokenize import RegexpTokenizer


MIN_YEAR = 1900
MAX_YEAR = 2100

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@.""-,`'

contractions_dict = {
    "ain't": "am not / are not / is not / has not / have not",
    "aint": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "arent": "are not / am not",
    "can't": "cannot",
    "cant": "cannot",
    "can't've": "cannot have",
    "cant've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldve": "could have",
    "couldnt": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "didnt": "did not",
    "doesnt": "does not",
    "dont": "do not",
    "hadnt": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "hasnt": "has not",
    "havent": "have not",
    "hed": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "hes": "he has / he is",
    "howd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "hows": "how has / how is / how does",
    "I'd": "I had / I would",
    "I'd've": "I would have",
    "I'll": "I shall / I will",
    "Ill": "I shall / I will",
    "I'll've": "I shall have / I will have",
    "I'm": "I am",
    "Im": "I am",
    "I've": "I have",
    "Ive": "I have",
    "isn't": "is not",
    "isnt": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "itll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "lets": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightve": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustve": "must have",
    "mustnt": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "neednt": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "shed": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "shes": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "theyll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "theyre": "they are",
    "they've": "they have",
    # "theyve": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "werent": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "whats": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "wheres": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "wont": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "yall": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "youd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "youll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "youre": "you are",
    "you've": "you have",
    "youve": "you have"
}


def get_url_patern():
    return re.compile(
        r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))'
        r'[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})')


def get_emojis_pattern():
    try:
        # UCS-4
        emojis_pattern = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    except re.error:
        # UCS-2
        emojis_pattern = re.compile(
            u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
    return emojis_pattern


def get_hashtags_pattern():
    return re.compile(r'#\w*')


def get_single_letter_words_pattern():
    return re.compile(r'(?<![\w\-])\w(?![\w\-])')


def get_blank_spaces_pattern():
    return re.compile(r'\s{2,}|\t')


def get_twitter_reserved_words_pattern():
    return re.compile(r'( RT| rt| FAV| fav| VIA| via)')


def get_mentions_pattern():
    return re.compile(r'@\w*')


def is_year(text):
    if (len(text) == 3 or len(text) == 4) and (MIN_YEAR < len(text) < MAX_YEAR):
        return True
    else:
        return False


class TwitterPreprocessor:

    def __init__(self, text: str):
        self.text = text
        self.print = False

    def ml_preprocess(self):
        return self \
            .remove_urls() \
            .remove_users() \
            .remove_mentions() \
            .remove_hashtags() \
            .demojify() \
            .filter_non_english() \
            .remove_punct_nltk() \
            .remove_numbers() \
            .remove_twitter_reserved_words() \
            .remove_single_letter_words() \
            .remove_blank_spaces() \
            .remove_stopwords() \
            .lowercase() \
            .expand_contractions()

    def se_preprocess(self):
        return self \
            .remove_urls() \
            .remove_hashtags() \
            .demojify() \
            .filter_non_english() \
            .remove_punct_nltk() \
            .remove_twitter_reserved_words() \
            .remove_single_letter_words() \
            .remove_blank_spaces() \
            .expand_contractions()

    def remove_urls(self):
        if self.print:
            print('Remove URL')
        self.text = re.sub(pattern=get_url_patern(), repl='', string=self.text)
        return self

    def remove_punct_emoj(self):
        if self.print:
            print('Remove Punct EMoji')
        self.text = ''.join(re.sub("(@[A-Za-z0–9]+)|([0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", self.text).split())
        return self

    def demojify(self):
        if self.print:
            print('Demojify')
        self.text = demoji.replace(self.text)
        return self

    def filter_non_english(self):
        if self.print:
            print('filter non english')
        printable = set(string.printable)
        self.text = ''.join(filter(lambda x: x in printable, self.text))
        return self

    def segment_text(self):
        if self.print:
            print('Segment text')
        load()
        self.text = ' '.join(segment(self.text))
        return self

    def expand_contractions(self):
        if self.print:
            print('Expand COntractions')
        contractions_re = re.compile(' (%s)' % '|'.join(contractions_dict.keys()))

        def replace(match):
            try:
                return contractions_dict[match.group(0)]
            except KeyError:
                pass

        self.text = contractions_re.sub(replace, self.text)
        return self

    def autocorrect_text(self):
        if self.print:
            print('Autocorrect text')
        self.text = Speller().autocorrect_sentence(self.text)
        return self

    #     def process_contractions(self):
    #         cont = Contractions(api_key="glove-twitter-100")
    #         self.text = cont.expand_texts([self.text])[0]
    #         return self

    def remove_punctuation(self):
        if self.print:
            print('Remove punctuation')
        self.text = self.text.translate(str.maketrans(' ', ' ', string.punctuation))
        self.text = re.sub('[' + my_punctuation + ']+', '', self.text)  # strip punctuation
        return self

    def remove_punct_nltk(self):
        if self.print:
            print('Remove ounct with nltk')
        tokenizer = RegexpTokenizer(r'\w+')
        tokenized = tokenizer.tokenize(self.text)
        self.text = ' '.join(tokenized)
        return self

    def remove_mentions(self):
        if self.print:
            print('Remove mentions')
        self.text = re.sub(pattern=get_mentions_pattern(), repl='', string=self.text)
        return self

    def remove_hash_sign(self):
        if self.print:
            print('Remove # sign')
        self.text = self.text.replace("#", "")
        return self

    def remove_at_sign(self):
        if self.print:
            print('Remove @ sign')
        self.text = self.text.replace("@", "")
        return self

    def remove_hashtags(self):
        if self.print:
            print('Remove hashtags')
        self.text = re.sub(pattern=get_hashtags_pattern(), repl='', string=self.text)
        return self

    def remove_twitter_reserved_words(self):
        if self.print:
            print('Remove reserved words')
        self.text = re.sub(pattern=get_twitter_reserved_words_pattern(), repl='', string=self.text)
        return self

    def remove_single_letter_words(self):
        if self.print:
            print('Remove single letter words')
        self.text = re.sub(pattern=get_single_letter_words_pattern(), repl='', string=self.text)
        return self

    def remove_blank_spaces(self):
        if self.print:
            print('Remove blank spaces')
        self.text = re.sub(pattern=get_blank_spaces_pattern(), repl=' ', string=self.text)
        self.text = re.sub('\s+', ' ', self.text)
        return self

    def remove_new_line(self):
        if self.print:
            print('Remove new line')
        self.text = re.sub('\s+', ' ', self.text)
        return self

    def remove_single_quotes(self):
        if self.print:
            print('Remove single quittes')
        self.text = re.sub("\'", "", self.text)
        return self

    def remove_emoji(self):
        if self.print:
            print('Remove emoji')
        '''Takes a string and removes the emoji'''
        self.text = emoji_pattern.sub(r'', self.text)  # remove emoji
        return self

    def remove_links(self):
        if self.print:
            print('Remove links')
        '''Takes a string and removes web links from it'''
        self.text = re.sub(r'http\S+', '', self.text)  # remove http links
        self.text = re.sub(r'bit.ly/\S+', '', self.text)  # rempve bitly links
        self.text = self.text.strip('[link]')  # remove [links]
        return self

    def remove_users(self):
        if self.print:
            print('Remove users')
        '''Takes a string and removes retweet and @user information'''
        self.text = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', self.text)  # remove retweet
        self.text = re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)', '', self.text)  # remove tweeted at
        return self

    def remove_stopwords(self, extra_stopwords=None):
        if self.print:
            print('Remove stopwords')
        if extra_stopwords is None:
            extra_stopwords = []
        text = nltk.word_tokenize(self.text)
        stop_words = set(stopwords.words('english'))

        new_sentence = []
        for w in text:
            if w not in stop_words and w not in extra_stopwords:
                new_sentence.append(w)
        self.text = ' '.join(new_sentence)
        return self

    def remove_numbers(self, preserve_years=False):
        if self.print:
            print('Remove numbers')
        text_list = self.text.split(' ')
        for text in text_list:
            if text.isnumeric():
                if preserve_years:
                    if not is_year(text):
                        text_list.remove(text)
                else:
                    text_list.remove(text)

        self.text = ' '.join(text_list)
        self.text = re.sub('([0-9]+)', '', self.text)
        return self

    def lowercase(self):
        self.text = self.text.lower()
        return self
