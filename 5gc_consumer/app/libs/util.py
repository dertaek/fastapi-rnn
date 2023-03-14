from datetime import datetime, timedelta


def get_timetuple_by_time(dt):
    '''
    获取各类需要的时间或时间字符串，常用的有前后5分钟，前后1天，前40天。
    @dt: datetime时间
    '''
    dt_next5min = dt + timedelta(minutes=5)
    dt_last5min = dt + timedelta(minutes=-5)
    dt_last10min = dt + timedelta(minutes=-10)
    dt_lastday = dt + timedelta(days=-1)
    dt_nextday = dt + timedelta(days=1)
    dt_lastmonth = (dt.replace(day=1) + timedelta(days=-1)).replace(day=dt.day)
    dt_last2month = (dt_lastmonth.replace(day=1) + timedelta(days=-1)).replace(day=dt.day)

    return {
        'dt': dt,
        'dt_next5min': dt_next5min,
        'dt_last5min': dt_last5min,
        'dt_last10min': dt_last10min,
        'dt_lastday': dt_lastday,
        'dt_nextday': dt_nextday,
        'dt_lastmonth': dt_lastmonth,
        'dt_last2month': dt_last2month
    }


def get_timetuple_by_timestr(time_str, format_str="%Y-%m-%d %H:%M:%S"):
    '''
    通过时间字符串获取各类需要的时间或时间字符串。
    @time_str: 时间字符串
    @format_str: 时间格式化字符串
    '''
    dt = datetime.strptime(time_str, format_str)
    return get_timetuple_by_time(dt)


def format_timestr(dt, format_type=0) -> str:
    '''
    格式化时间为字符串。
    @dt: 时间戳
    @format_type: 格式化类型
    @format_type = 0 => '%Y-%m-%dT%X.000Z'
    @format_type = 1 => '%Y-%m-%d %X'
    @format_type = 2 => '%Y%m%d%H%M'
    @format_type = 3 => '%Y%m%d'
    @format_type = 4 => '%Y%m'
    @format_type = 5 => '%Y-%m-%d %H:%M'
    '''
    formats = {
        0: '%Y-%m-%dT%X.000Z',
        1: '%Y-%m-%d %X',
        2: '%Y%m%d%H%M',
        3: '%Y%m%d',
        4: '%Y%m',
        5: '%Y-%m-%d %H:%M'
    }
    if format_type > len(formats) or format_type < 0:
        format_type = 0
    return dt.strftime(formats[format_type])