#!/bin/env python
# Copyright (c) 2012-2014 The Khronos Group Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and/or associated documentation files (the
# "Materials"), to deal in the Materials without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Materials, and to
# permit persons to whom the Materials are furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Materials.
#
# THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

import os
import sys
import traceback

import datetime
import time

import subprocess
import threading

import re

from pprint import pprint

class BackgroundProcess(threading.Thread):
    def __init__(self, **args):
        self.args = args
        self.stdout = None
        self.stderr = None
        self.process = None
        threading.Thread.__init__(self)

    def run(self):
        self.process = subprocess.Popen(
            stderr=subprocess.PIPE,
            **self.args)
        self.stdout, self.stderr = self.process.communicate()

report_re = re.compile(
        r'#REPORT:'
        r' (?P<timestamp>.{14})'
        r' (?P<testid>.*)'
        r' (?P<total>\d+)'
        r' (?P<disabled>\d+)'
        r' (?P<started>\d+)'
        r' (?P<completed>\d+)'
        r' (?P<passed>\d+)'
        r' (?P<failed>\d+)'
        r' (?P<version>[^$]+)'
)

class TestRunner(object):

    tests = []

    timeout = float(os.environ.get('VX_TEST_TIMEOUT', '65'))  # seconds

    testid = '?'
    total_tests = 0
    total_disabled_tests = 0
    total_started_tests = 0
    total_completed_tests = 0
    total_passed_tests = 0
    total_failed_tests = 0
    tests_version = 'unknown'

    def get_test_list(self):
        p = subprocess.Popen(
                             args=self.launch_args + ['--quiet', '--list_tests', '--run_disabled'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            raise Exception("Can't get list of tests")

        lines = stdout.replace('\r\n', '\n').split('\n');
        for line in lines:
            if line == "":
                continue
            self.tests.append(line)

        m = re.search(report_re, stderr)
        if m:
            self.testid = m.group('testid')

    def run_test(self, test):
        self.total_started_tests += 1

        bp = BackgroundProcess(
                args=self.launch_args + ['--quiet', '--filter=%s' % test.replace(':', '*')]
                )
        bp.start()

        bp.join(self.timeout)
        if bp.is_alive():
            print('#TIMEOUT on test "%s". TERMINATING' % test)
            sys.stdout.flush()
            try:
                bp.process.terminate()
            except:
                pass
            bp.join()

        m = re.search(report_re, bp.stderr)
        if m:
            timestamp = m.group('timestamp')
            testid = m.group('testid')
            total = m.group('total')
            disabled = m.group('disabled')
            started = m.group('started')
            completed = m.group('completed')
            passed = m.group('passed')
            failed = m.group('failed')
            if m.group('version'):
                self.tests_version = m.group('version')
            if str(started) == '0' and str(disabled) == '0':
                print("#CHECK FILTER: %s" % test)
                self.total_failed_tests += 1
            if str(disabled) != '0':
                 self.total_disabled_tests += 1
                 self.total_started_tests -= 1
            if str(completed) != '0':
                 self.total_completed_tests += 1
            if str(passed) != '0':
                 self.total_passed_tests += 1
            if str(failed) != '0':
                 self.total_failed_tests += 1
        else:
            self.total_failed_tests += 1
            sys.stdout.write(bp.stderr)
            if bp.process.returncode != 0:
                print('Process exit code: %d' % bp.process.returncode)

    def printUsage(self):
        print('''\
Usage:
    run_tests.py <vx_test_conformance executable> <filter and other parameters>

Environment variables:
    VX_TEST_DATA_PATH - path to test_data directory (used by vx_test_conformance)
    VX_TEST_TIMEOUT - single test timeout (in seconds)

Example:
    run_tests.py ./bin/vx_test_conformance
    run_tests.py ./bin/vx_test_conformance --filter=*Canny*\
''')

    def run(self):
        try:
            if len(sys.argv) < 2:
                print("Missed executable path")
                self.printUsage()
                return 2
            if sys.argv[1] in ['-h', '--help', '/?']:
                self.printUsage()
                return 0

            self.launch_args = sys.argv[1:]

            self.get_test_list()

            self.launch_args = [a for a in self.launch_args if not a.startswith('--filter=')]

            self.total_tests = len(self.tests)

            print('#FOUND %d tests' % self.total_tests)
            print('Test timeout=%s' % self.timeout)
            print('')
            sys.stdout.flush()

            prev = 0
            i = 0
            for t in self.tests:
                if (self.total_tests >= 500):
                    next = i * 100 / self.total_tests
                    if int(next) != prev:
                        print('# %02d%%' % next)
                        prev = next
                i += 1
                sys.stdout.flush()
                sys.stderr.flush()
                try:
                    self.run_test(t)
                except KeyboardInterrupt:
                    break
                except:
                    print traceback.format_exc()

            print('')
            print('ALL DONE')

            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

            print('')
            print('#REPORT: %s %s %d %d %d %d %d %d %s' % (
                    timestamp, self.testid,
                    self.total_tests,
                    self.total_disabled_tests,
                    self.total_started_tests,
                    self.total_completed_tests,
                    self.total_passed_tests,
                    self.total_failed_tests,
                    self.tests_version))

            return 0 if (self.total_tests == (self.total_started_tests + self.total_disabled_tests) and self.total_failed_tests == 0) else 1
        except:
            print traceback.format_exc()

if __name__ == "__main__":
    sys.exit(TestRunner().run())
