from datetime import datetime, time
import numpy as np
import re
import locale
import seaborn as sns

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


def plot_covariance_heatmaps(covariance_X, covariance_inverse_X):
    # sns.plt.clf()
    f, (ax1, ax2) = sns.plt.subplots(1, 2, figsize=(12, 5))

    # sns.plt.figure()
    ax1.set_title("covariance_X")
    sns.heatmap(covariance_X, annot=False, fmt=".2f", robust=True, ax=ax1, cmap="Paired")

    # sns.plt.figure()
    ax2.set_title("covariance_inverse_X")
    sns.heatmap(covariance_inverse_X, annot=False, fmt=".2f", robust=True, ax=ax2, cmap="Paired")

    sns.plt.show()

# plt.figure()
# plt.imshow(covariance_X, cmap='Reds', interpolation='nearest')
#
# plt.figure()
# plt.imshow(covariance_inverse_X, cmap='Reds', interpolation='nearest')
#
# plt.show()