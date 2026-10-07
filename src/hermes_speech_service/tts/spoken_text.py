"""Verbalize technical tokens before model punctuation normalization."""
import re

# Match ASCII technical tokens without using Unicode-sensitive word boundaries.
_TECHNICAL = re.compile(
    r'(?P<path>(?<![A-Za-z0-9_])[A-Za-z]:[\\/][A-Za-z0-9_./\\-]+)'
    r'|(?P<escape>\\[bBwWdDsS](?![A-Za-z0-9_]))'
    r'|(?P<code>(?<![A-Za-z0-9_])(?:'
    r'(?:[A-Za-z_][A-Za-z0-9_]*\.)+[A-Za-z_][A-Za-z0-9_]*(?:\([^()\n]*\))?'
    r'|[A-Za-z_][A-Za-z0-9_]*\([^()\n]*\)'
    r'|[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+))'
)
_SYMBOLS = {'.': ' 点 ', '_': ' 下划线 ', '=': ' 等于 ',
            '(': ' 左括号 ', ')': ' 右括号 ', '\\': ' 反斜杠 ',
            '/': ' 斜杠 ', '-': ' 连字符 ', ':': ' 冒号 '}

# Hermes inserts spaces after periods. Restrict recovery to lowercase dotted
# configuration keys immediately followed by a colon, preserving normal prose.
_CONFIG_KEY = re.compile(
    r'(?<![A-Za-z0-9_.])(?P<key>[a-z_][a-z0-9_]*(?:\.[ \t]*[a-z_][a-z0-9_]*)+)'
    r'[ \t]*[:：][ \t]*(?=[^\s])'
)
_ORDERED_ITEM = re.compile(r'(?m)^[ \t]*(\d{1,2})[.)][ \t]+(?=\S)')


def _ordinal(match):
    number = int(match[1])
    if not 1 <= number <= 99:
        return match[0]
    digits = '零一二三四五六七八九'
    tens, ones = divmod(number, 10)
    spoken = ((digits[tens] if tens > 1 else '') + '十' if tens else '')
    spoken += digits[ones] if ones else ''
    return '第' + spoken + '，'


def _speak_token(value):
    value = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', value)
    value = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', value)
    value = ''.join(_SYMBOLS.get(char, char) for char in value)
    value = re.sub(r'(?<![A-Za-z0-9])[A-Z]{2,6}(?![A-Za-z0-9])',
                   lambda match: ' '.join(match[0]), value)
    return re.sub(r' +', ' ', value).strip()


def normalize_technical_text(text):
    def config_key(match):
        key = re.sub(r'\.[ \t]*', '.', match['key'])
        # English compound segmentation keeps both syllables audible in mixed
        # Chinese/English configuration keys without changing their meaning.
        spoken = _speak_token(key)
        spoken = re.sub(r'(?<![A-Za-z])backend(?![A-Za-z])', 'back end', spoken)
        return spoken + '，冒号，'

    text = _CONFIG_KEY.sub(config_key, text)
    # Only actual surviving line-start list markers are verbalized. Never
    # infer a deleted list number from the words that follow it.
    if re.search(r'[\u4e00-\u9fff]', text):
        text = _ORDERED_ITEM.sub(_ordinal, text)
        # CosyVoice replace_corner_mark otherwise drops the Chinese dash into
        # whitespace. Preserve its clause boundary before that frontend runs.
        text = re.sub(r'(?<=[\u4e00-\u9fff])[ \t]*—{1,2}[ \t]*(?=[\u4e00-\u9fff])', '；', text)
        # Narrow prosody hint for the observed noun/verb misgrouping. Keep the
        # complete term intact and leave all other occurrences unchanged.
        text = text.replace('热词表喂', '热词表，喂')
        # A colon following Chinese prose is a prosodic boundary; do not
        # alter timestamps, drive letters or English key-value syntax.
        text = re.sub(r'(?<=[\u4e00-\u9fff])[：:][ \t]*(?:[.。][ \t]*)?', '。', text)

    def replace(match):
        value = match[0]
        if match.lastgroup == 'escape':
            letter = value[1]
            return '反斜杠' + ('大写 ' if letter.isupper() else '小写 ') + letter.upper() + ' '
        if match.lastgroup == 'path':
            # Keep sentence punctuation outside the path, including English prose.
            suffix = '.' if value.endswith('.') else ''
            value = value[:-1] if suffix else value
            return value[0].upper() + ' 盘 ' + _speak_token(value[2:]) + suffix
        return _speak_token(value)

    return _TECHNICAL.sub(replace, text).replace('`', '')

