from datetime import datetime, time

import numpy as np

import re

import locale

print(locale.getdefaultlocale())
locale.setlocale(locale.LC_TIME, "spanish")


def parse_dates(dates):
    if type(dates) is "str":
        date = dates
        return parse_date(date)
    else:
        parsed_dates = np.empty(len(dates), dtype="datetime64[s]")
        for i in range(len(dates)):
            parsed_dates[i] = parse_date(dates[i])
        return parsed_dates


def parse_date(date_string):
    # 06-oct-2015 21:57:03
    locale_date_string = re.sub("(.+-)(.+)(-.+)", "\\1\\2.\\3", date_string)
    return datetime.strptime(locale_date_string, "%d-%b-%Y %H:%M:%S")
