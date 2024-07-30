import re

date_regex = re.compile(r'''^\d{1,4}[-\/\.\s]\S+[-\/\.\s]\S+''')

def is_date(value):
    return date_regex.match(value)

