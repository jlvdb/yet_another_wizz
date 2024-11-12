import sys
from logging import DEBUG, LogRecord, getLogger

from pytest import mark

from yaw import logging


@mark.parametrize(
    "value,expect",
    [
        (-123.0, "-123"),
        (1234.0, "1.23K"),
        (1235.0, "1.24K"),
        (12345, "12.3K"),
        (123456, "123K"),
        (1234567, "1.23M"),
    ],
)
def test_long_num_format(value, expect):
    assert logging.long_num_format(value) == expect


@mark.parametrize(
    "value,expect",
    [
        (1.9, "0m01.90s"),
        (59.99, "0m59.99s"),
        (61.9, "1m01.90s"),
        (3600, "60m00.00s"),
    ],
)
def test_format_time(value, expect):
    assert logging.format_time(value) == expect


def test_Indicator(capsys):
    iterator = range(10)
    stream = sys.stdout
    prefix = "INDICATOR"
    logging.set_indicator_prefix(prefix)
    assert logging.INDICATOR_PREFIX == prefix

    indicator = logging.Indicator(iterator, min_interval=0.0, stream=stream)
    assert indicator.printer.stream is stream
    assert indicator.num_items == len(iterator)

    for _ in indicator:
        pass
    captured = capsys.readouterr()

    assert captured.out.count("\r") == indicator.num_items + 2
    assert captured.out.count(prefix) == indicator.num_items + 2
    assert captured.out.endswith("\n")


@mark.parametrize(
    "name,filtered",
    [("yaw", True), ("yaw.utils", True), ("mod", False), ("mod.yaw", False)],
)
def test_OnlyYAWFilter(name, filtered):
    record = LogRecord(name, DEBUG, "file.py", 1, "log msg", None, None)
    assert logging.OnlyYAWFilter().filter(record) is filtered


def send_logs(log_msg):
    logger_yaw = getLogger("yaw.test")
    logger_filtered = getLogger("any.module")
    logger_filtered.debug(log_msg)
    logger_yaw.debug(log_msg)


@mark.parametrize("pretty", [False, True])
def test_get_default_logger(caplog, pretty):
    caplog.set_level(DEBUG, logger="yaw")
    n_logs = 0
    log_msg = "test msg"

    logger_filtered = getLogger("any.module")
    logger_yaw = getLogger("yaw.test")

    logger = logging.get_default_logger("debug", pretty=pretty)
    logger.level == DEBUG
    n_logs += 2 - int(pretty)
    assert len(caplog.records) == n_logs

    logger_filtered.debug(log_msg)
    assert len(caplog.records) == n_logs

    logger_yaw.debug(log_msg)
    n_logs += 1
    assert len(caplog.records) == n_logs
    assert log_msg in caplog.records[-1].message
