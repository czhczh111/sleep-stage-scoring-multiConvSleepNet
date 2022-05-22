import datetime
import time


def sec_dif_within_day(start_time, current_time):
    #24小时内的两个时间的秒数之差
    start_sec = start_time.second
    start_min = start_time.minute
    start_hour = start_time.hour
    start_day = start_time.day
    start_month = start_time.month
    start_year = start_time.year

    current_sec = current_time.second
    current_min = current_time.minute
    current_hour = current_time.hour
    current_day = current_time.day
    current_month = current_time.month
    current_year = current_time.year
    if current_year > start_year or current_month > start_month or current_day > start_day:
        current_hour += 24
    diff = (current_hour * 3600 + current_min * 60 + current_sec) - (start_hour * 3600 + start_min * 60 + start_sec)
    return diff

