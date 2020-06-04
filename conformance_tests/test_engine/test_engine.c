/*
 * Copyright (c) 2012-2014 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>

#ifdef _MSC_VER
#include <direct.h>
#elif defined __linux__
#include <unistd.h>
#endif


#define HAVE_TIME_H

#ifdef HAVE_TIME_H
#include <time.h>
#endif

#include "test.h"

char CT_EXTENDED_ARG_BEGIN[] = {'\0'};
char CT_EXTENDED_ARG_END[] = {'\0'};

#ifdef _MSC_VER
#  define snprintf(buf, sz, pattern, ...) _snprintf_s(buf, sz, _TRUNCATE, pattern, __VA_ARGS__)
#  undef setenv
#  define setenv(name, value, overwrite) _putenv_s(name, value)
#elif defined __linux__
int setenv(const char* name, const char* value, int overwite);
#else
#  define setenv(...)
#endif

#ifdef CT_TEST_TIME

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h> // gettimeofday
#endif
#include <time.h>
#if defined __MACH__ && defined __APPLE__
#include <mach/mach_time.h>
#endif

#if defined WIN32 || defined _WIN32 || defined WINCE
#include <windows.h> // QueryPerformanceFrequency / QueryPerformanceCounter
#endif

static int64_t CT_getTickCount(void)
{
#if defined WIN32 || defined _WIN32 || defined WINCE
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (int64_t)counter.QuadPart;
#elif defined __linux || defined __linux__
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return (int64_t)tp.tv_sec * 1e9 + tp.tv_nsec;
#elif defined __MACH__ && defined __APPLE__
    return (int64_t)mach_absolute_time();
#else
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return (int64_t)tv.tv_sec * 1e6 + tv.tv_usec;
#endif
}

static double CT_getTickFrequency(void)
{
#if defined WIN32 || defined _WIN32 || defined WINCE
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return (double)freq.QuadPart;
#elif defined __linux || defined __linux__
    return 1e9;
#elif defined __MACH__ && defined __APPLE__
    static double freq = 0;
    if(freq == 0)
    {
        mach_timebase_info_data_t sTimebaseInfo;
        mach_timebase_info(&sTimebaseInfo);
        freq = sTimebaseInfo.denom * 1e9 / sTimebaseInfo.numer;
    }
    return freq;
#else
    return 1e6;
#endif
}

static double g_tickFreq = 0;

static int g_timeShow =
#ifdef CT_TIME_SHOW
        1
#else
        0
#endif
        ;
#endif

#define CT_LOGF(...)            \
    do {                        \
        printf(__VA_ARGS__);    \
        fflush(stdout);         \
    } while(0)

struct CT_GC_Node
{
    CT_GCType           type_;
    void*               object_;
    CT_ObjectDestructor destructor_;
    struct CT_GC_Node*  next_;
};

struct CT_FailedTestEntry
{
    struct CT_TestCaseEntry*   testcase_;
    struct CT_TestEntry*       test_;
    int                        param_idx_;
    struct CT_FailedTestEntry* next_;
};

struct CT_GlobalContextBlackBox
{
    int g_quiet;
    int g_list_tests;

    int num_test_errors_;
    int g_num_failed_tests_;
    int g_num_disabled_tests_;
    struct CT_GC_Node* gc_chain_;

    struct CT_FailedTestEntry* g_failed_tests_;
    struct CT_FailedTestEntry* g_failed_tests_end_;
};

// testing context, it might be converted to be thread local some day
static struct CT_GlobalContextBlackBox g_context_internals = { 0 };
static struct CT_GlobalContext g_context = { 0, 0, 0, 0, &g_context_internals };
static int g_has_running_test = 0;
static int g_option_run_disabled_tests = 0;

struct CT_GlobalContext* CT() { return g_has_running_test ? & g_context : NULL; }

void CT_RecordFailure()
{
    if (!CT()->internal_->num_test_errors_)
        CT()->internal_->g_num_failed_tests_++;
    CT()->internal_->num_test_errors_++;
}

void CT_RecordFailureAt(const char* message, const char* func, const char* file, const int line)
{
    CT_RecordFailure();
    printf("\nFAILED at %20s:%d\n\t%s\n\n", file, line, message);
    fflush(stdout);
}

void CT_RecordFailureAtFormat(const char* message, const char* func, const char* file, const int line, ...)
{
    va_list args;

    CT_RecordFailure();

    printf("\nFAILED at %20s:%d\n\t", file, line);
    fflush(stdout); // just in case of mailformed "message"

    va_start(args, line);
    vprintf(message, args);
    va_end(args);

    printf("\n\n");
    fflush(stdout);
}

void CT_DumpMessage(const char* message, ...)
{
    va_list args;

    va_start(args, message);
    vprintf(message, args);
    va_end(args);

    printf("\n\n");
    fflush(stdout);
}

int CT_HasFailure()
{
    return CT()->internal_->num_test_errors_;
}

void CT_RegisterForGarbageCollection(void *object, CT_ObjectDestructor collector, CT_GCType type)
{
    struct CT_GlobalContextBlackBox* bb = CT()->internal_;
    struct CT_GC_Node* node = (struct CT_GC_Node*)malloc(sizeof(struct CT_GC_Node));

    if (!node)
    {
        CT_RecordFailure();
        return;
    }

    node->type_ = type > 0 ? type : CT_GC_OBJECT;
    node->object_ = object;
    node->destructor_ = collector;
    node->next_ = bb->gc_chain_;
    bb->gc_chain_ = node;
}

void CT_CollectGarbage(int type)
{
    struct CT_GlobalContextBlackBox* bb = CT()->internal_;
    struct CT_GC_Node* node = bb->gc_chain_;
    struct CT_GC_Node stub, *prev = &stub;
    stub.next_ = node;

    while(node)
    {
        struct CT_GC_Node* killme = node;
        node = node->next_;

        if(type == CT_GC_ALL || (CT_GCType)type == killme->type_)
        {
            killme->destructor_(&killme->object_);
            free(killme);
            prev->next_ = node;
        }
        else
        {
            prev = killme;
        }
    }

    bb->gc_chain_ = stub.next_;
}

#ifdef HAVE_VCS_VERSION_INC
# include "vcs_version.inc"
#endif
#ifndef VCS_VERSION_STR
# define VCS_VERSION_STR "unknown"
#endif

static void print_version(const char* version_str)
{
    printf("VxTests version: %s\n", version_str);
    printf("VCS version: " VCS_VERSION_STR "\n");
    printf("Build config: "
#ifdef DEBUG
        "Debug"
#elif NDEBUG
        "Release"
#else
        "unknown"
#endif
        "\n\n");
    fflush(stdout);
}


static int isNameMatches(const char* str, const char *pattern)
{
    switch (*pattern)
    {
    case '?':
        return (*str != '\0' && isNameMatches(str + 1, pattern + 1) != 0) ? 1 : 0;
    case '*':
        return ((*str != '\0' && isNameMatches(str + 1, pattern) != 0) ||
                isNameMatches(str, pattern + 1) != 0) ? 1 : 0;
    case '\0': case ':': case '\n':
        return (*str == '\0' || *str == '/') ? 1 : 0;
    default:
        return (*pattern == *str &&
                isNameMatches(str + 1, pattern + 1) != 0) ? 1 : 0;
    }
}


// accepts filters like gtest with some changes for "negative" tests
static int filterTestName(const char* test_name, const char* fullFilter)
{
    const char *cur_pattern = fullFilter;
    int result = 0;
    if (fullFilter == NULL)
    {
        result = 1; // no filter
    }
    else
    {
        for (;;)
        {
            int negative = 0;
            if (*cur_pattern == '-')
            {
                negative = 1;
                cur_pattern++;
            }
            if (result == 0 || negative == 1)
            {
                if (isNameMatches(test_name, cur_pattern) != 0)
                {
                    if (negative)
                        return 0;
                    result = 1;
                }
            }

            cur_pattern = strchr(cur_pattern, ':');

            if (cur_pattern == NULL)
                break;
            cur_pattern++;
        }
    }

    if (result && !g_option_run_disabled_tests && strstr(test_name, "DISABLED") != NULL)
    {
        g_context.internal_->g_num_disabled_tests_++;
        result = 0;
    }

    return result;
}

static uint64_t fnv1a(const char *str)
{
    uint64_t hval = 0xCBF29CE484222325ULL;
    const unsigned char *s = (const unsigned char *)str;

    while (*s)
    {
        hval ^= (uint64_t)*s++;
        hval *= 0x100000001B3ULL;
    }

    return hval;
}

extern CT_RegisterTestCaseFN g_testcase_register_fns[];
static struct CT_TestCaseEntry* g_firstTestCase = NULL;
static const char* g_test_filter = NULL;

static void* get_test_params(struct CT_TestEntry* test, int param_idx)
{
    return test->args_ ? (void*)(((uint8_t*)test->args_) + test->arg_size_ * param_idx) : NULL;
}

static void get_test_name(char* buf, int bufsz, struct CT_TestCaseEntry* testcase, struct CT_TestEntry* test, void* parg, int param_idx)
{
    const char* test_name = test->name_;
    if (strncmp(test_name, "test_", 5) == 0)
        test_name += 5;
    else if (strncmp(test_name, "test", 4) == 0)
        test_name += 4;
    if (parg)
        snprintf(buf, bufsz, "%s.%s/%d/%s", testcase->name_, test_name, param_idx, *(const char**)parg);
    else
        snprintf(buf, bufsz, "%s.%s", testcase->name_, test_name);
}

static int update_extended_flag(void* parg, int* extended_flag)
{
    if (parg)
    {
        if (*(const char**)parg == CT_EXTENDED_ARG_BEGIN)
        {
            *extended_flag = 1;
            return 1;
        }
        else if (*(const char**)parg == CT_EXTENDED_ARG_END)
        {
            *extended_flag = 0;
            return 1;
        }
    }
    return 0;
}

static int run_test(struct CT_TestCaseEntry* testcase, struct CT_TestEntry* test, int param_idx, int run_tests, int* extended_flag)
{
    char test_name[1024];
    void *parg = get_test_params(test, param_idx);
    get_test_name(test_name, sizeof(test_name), testcase, test, parg, param_idx);

    if (update_extended_flag(parg, extended_flag))
        return 0;

    if (*extended_flag && !ct_check_any_size())
        return 0;

    if (filterTestName(test_name, g_test_filter))
    {
        if (g_context.internal_->g_list_tests)
        {
            CT_LOGF("%s\n", test_name);
            return 0;
        }
        else
        {
            char timestr[256] = {0};
#ifdef CT_TEST_TIME
            int64_t timestart;
#endif

            if (run_tests == 0 && !g_context.internal_->g_quiet)
                CT_LOGF("[ -------- ] tests from %s\n", testcase->name_);

            // setup global test execution context
            g_context.testname_     = test_name;
            g_context.seed_         = fnv1a(test_name);
            g_context.arg_          = parg;
            g_context.user_context_ = NULL;
            g_context.internal_->num_test_errors_ = 0;

            if (g_context.internal_->g_quiet)
            {
                CT_LOGF("[ RUN      ] %s ...\n", test_name);
            }
            else
            {
                CT_LOGF("[ RUN %04d ] %s ...\n", run_tests+1, test_name);
            }

            g_has_running_test = 1; /* GO! */

#ifdef CT_TEST_TIME
            timestart = CT_getTickCount();
#endif

            // test setup
            g_context.user_context_ = (testcase->setupFn_) ? testcase->setupFn_() : NULL;

            if (!CT_HasFailure()) /* no errors during test setup */
            {
                // test body
                test->test_fn_(g_context.user_context_, parg);

                //test teardown
                if (testcase->teardownFn_) testcase->teardownFn_(g_context.user_context_);
            }
            else
            {
                /* do not call teardown if setup is failed*/
                CT_LOGF("[ !FAILED! ] Test setup\n");
            }

            // release automatic resources
            CT_CollectGarbage(CT_GC_ALL);

            g_has_running_test = 0; /* FIN! */

#ifdef CT_TEST_TIME
            if (g_timeShow)
                snprintf(timestr, sizeof(timestr), " (%.1f ms)", (CT_getTickCount() - timestart) * 1000. / g_tickFreq);
#endif

            CT_LOGF("[ %s ] %s%s\n",
                (g_context.internal_->num_test_errors_) ? "!FAILED!" : "    DONE", test_name, timestr);

            if (g_context.internal_->num_test_errors_)
            {
                struct CT_FailedTestEntry* f = (struct CT_FailedTestEntry*)(malloc(sizeof(*f)));
                f->testcase_ = testcase;
                f->test_ = test;
                f->param_idx_ = param_idx;
                f->next_ = NULL;

                if (g_context.internal_->g_failed_tests_end_)
                    g_context.internal_->g_failed_tests_end_->next_ = f;
                else
                    g_context.internal_->g_failed_tests_ = f;
                g_context.internal_->g_failed_tests_end_ = f;
            }
            return 1; // test was executed
        }
    }
    return 0; // test is skipped
}

int CT_main(int argc, char* argv[], const char* version_str)
{
    const char* testid_str = 0;
    int arg;
    int total_tests = 0;
    int total_testcases = 0;
    int total_run_tests = 0;
    int total_run_testcases = 0;

    int use_global_context = 0;

#ifdef CT_TEST_TIME
    int64_t timestart_all;
#endif

    struct CT_TestCaseEntry* testcase = 0;

    for (arg = 1; arg < argc; arg++)
    {
        const char* argStr = argv[arg];
        if (memcmp(argStr, "--filter=", 9) == 0)
        {
            if (g_test_filter)
            {
                // TODO add message
                return 1;
            }
            g_test_filter = argStr + 9;
        }
        else if (strcmp(argStr, "--verbose") == 0)
        {
            setenv("VX_ZONE_LIST", "0,1", 1);
        }
        else if (memcmp(argStr, "--global_context=", 17) == 0)
        {
            use_global_context = atoi(argStr + 17);
        }
        else if (memcmp(argStr, "--run_disabled", 14) == 0)
        {
            g_option_run_disabled_tests = 1;
        }
        else if (memcmp(argStr, "--check_any_size=", 17) == 0)
        {
            ct_set_check_any_size(atoi(argStr + 17) != 0);
        }
        else if (memcmp(argStr, "--testid=", 9) == 0)
        {
            testid_str = argStr + 9;
        }
        else if (memcmp(argStr, "--list_tests", 9) == 0)
        {
            g_context.internal_->g_list_tests = 1;
        }
        else if (memcmp(argStr, "--quiet", 8) == 0)
        {
            g_context.internal_->g_quiet = 1;
        }
        else if (memcmp(argStr, "--show_test_duration=", 21) == 0)
        {
#ifdef CT_TEST_TIME
            g_timeShow = (atoi(argStr + 21) != 0);
#else
            // nothing, ignore option
#endif
        }
        else if (memcmp(argStr, "--help", 7) == 0)
        {
            print_version(version_str);
            printf("Usage:\n");
            printf("    %s [--filter=<filter>] [--run_disabled] [--global_context=0|1] [--check_any_size=0|1] [--show_test_duration=0|1] [--verbose] [--testid=<testid>] [--list_tests] [--quiet]\n", argv[0]);
            printf("\n");
            printf("   <filter> - is GTest like filter, list of patterns separated by colon ':'.\n");
            printf("              Filter-out tests with '-' pattern's prefix.\n");
            printf("              Negative patterns have higher priority than positive patterns.\n\n");
            printf("   <testid> - report custom identifier for tests run\n\n");
            return 0;
        }
        else
        {
            printf("ERROR: Unknown option %s\n", argStr);
            return 1;
        }
    }

    if (!g_context.internal_->g_quiet)
        print_version(version_str);

    {
        struct CT_TestCaseEntry** ppLastTestCase = &g_firstTestCase;
        while (g_testcase_register_fns[total_testcases])
        {
            *ppLastTestCase = g_testcase_register_fns[total_testcases]();
            while (*ppLastTestCase)
                ppLastTestCase = &ppLastTestCase[0]->next_;
            total_testcases++;
        }
    }

    testcase = g_firstTestCase;
    while (testcase)
    {
        struct CT_TestEntry** ppLastTest = &testcase->tests_;
        int testcase_tests = 0;
        int test_id = 0;
        for (; testcase->test_register_fns_[test_id]; test_id++)
        {
            *ppLastTest = testcase->test_register_fns_[test_id]();
            while (*ppLastTest)
            {
                if (ppLastTest[0]->args_)
                {
                    int extended_flag = 0;
                    struct CT_TestEntry* test = ppLastTest[0];
                    int narg = 0;
                    for (; narg < test->args_count_; narg++)
                    {
                        void *parg = get_test_params(test, narg);
                        if (update_extended_flag(parg, &extended_flag))
                            continue;
                        if (extended_flag && !ct_check_any_size())
                            continue;
                        testcase_tests += 1;
                    }
                }
                else
                {
                    testcase_tests += 1;
                }
                ppLastTest[0]->testcase_ = testcase;
                ppLastTest = &ppLastTest[0]->next_;
            }
        }
        testcase->test_count_ = testcase_tests;
        total_tests += testcase_tests;
        testcase = testcase->next_;
    }


    if (!g_context.internal_->g_quiet)
    {
        printf("[ ======== ] Total %d tests from %d test cases\n", total_tests, total_testcases);
        if (g_test_filter)
            printf("Use test filter: %s\n\n", g_test_filter);
        printf("Use global OpenVX context: %s\n\n", use_global_context ? "TRUE" : "FALSE");
        printf("\n");
    }

    {
        const char* test_data_path = getenv("VX_TEST_DATA_PATH");
        if (test_data_path && test_data_path[0] != '\0')
        {
#ifdef _MSC_VER
            if (0 != _chdir(test_data_path))
#else
            if (0 != chdir(test_data_path))
#endif
            {
                printf("ERROR: VX_TEST_DATA_PATH points to bad location\n");
            }
        }
    }

#ifdef CT_TEST_TIME
    g_tickFreq = CT_getTickFrequency();
#endif

    if (use_global_context)
        ct_create_global_vx_context();

#ifdef CT_TEST_TIME
    timestart_all = CT_getTickCount();
#endif

    for (testcase = g_firstTestCase; testcase; testcase = testcase->next_)
    {
        int run_tests = 0;
        int extended_flag = 0;
        struct CT_TestEntry* test = testcase->tests_;

#ifdef CT_TEST_TIME
        int64_t timestart_testCase = CT_getTickCount();
#endif

        for(; test; test = test->next_)
        {
            if (!test->args_)
                run_tests += run_test(testcase, test, 0, run_tests, &extended_flag);
            else
            {
                int narg = 0;
                for (; narg < test->args_count_; narg++)
                    run_tests += run_test(testcase, test, narg, run_tests, &extended_flag);
            }
        }

        if (run_tests)
        {
            if (!g_context.internal_->g_quiet)
            {
                char timestr[256] = {0};
#ifdef CT_TEST_TIME
                if (g_timeShow)
                    snprintf(timestr, sizeof(timestr), " (%.1f ms)", (CT_getTickCount() - timestart_testCase) * 1000. / g_tickFreq);
#endif
                printf("[ -------- ] %d tests from test case %s%s\n\n", run_tests, testcase->name_, timestr);
            }
            total_run_tests += run_tests;
            total_run_testcases++;
        }
    }

    if (!g_context.internal_->g_quiet)
    {
        char timestr[256] = {0};
#ifdef CT_TEST_TIME
        if (g_timeShow)
            snprintf(timestr, sizeof(timestr), " (%.1f ms)", (CT_getTickCount() - timestart_all) * 1000. / g_tickFreq);
#endif
        printf("[ ======== ]\n");
        printf("[ ALL DONE ] %d tests from %d test cases ran%s\n", total_run_tests, total_run_testcases, timestr);
        printf("[ !PASSED! ] %d tests\n", total_run_tests - g_context.internal_->g_num_failed_tests_);
        if (g_context.internal_->g_num_failed_tests_ > 0)
        {
            struct CT_FailedTestEntry* f = g_context.internal_->g_failed_tests_;
            printf("[ !FAILED! ] %d test%s, listed below:\n", g_context.internal_->g_num_failed_tests_, g_context.internal_->g_num_failed_tests_ == 1 ? "" : "s");
            for(; f; f = f->next_)
            {
                char test_name[1024];
                void *parg = get_test_params(f->test_, f->param_idx_);
                get_test_name(test_name, sizeof(test_name), f->testcase_, f->test_, parg, f->param_idx_);
                printf("[ !FAILED! ] %s\n", test_name);
            }
        }
        if (g_context.internal_->g_num_disabled_tests_ > 0)
        {
            printf("\n    YOU HAVE %d DISABLED TESTS\n", g_context.internal_->g_num_disabled_tests_);
        }
    }
    fflush(stdout);
    ct_release_global_vx_context();

    if (testid_str == 0)
    {
        if (g_test_filter)
            testid_str = "FILTERED";
        else
            testid_str = "ALL";
    }

    {
        char timebuf[64] = "YYYYMMDDHHMMSS";
#ifdef HAVE_TIME_H
        time_t t = time(NULL);
        struct tm *tmptr = localtime(&t);
        if (tmptr != NULL)
        {
            sprintf(timebuf, "%04d%02d%02d%02d%02d%02d",
                    tmptr->tm_year + 1900,
                    tmptr->tm_mon + 1,
                    tmptr->tm_mday,
                    tmptr->tm_hour,
                    tmptr->tm_min,
                    tmptr->tm_sec /* [0-60] (1 leap second) */);
        }
#endif
        // <identifier> <time and date> <test identifier> <total number of tests> <total tests disabled> <total tests started> <total tests run to completion> <total tests passed> <total tests failed>
        fprintf(g_context.internal_->g_quiet ? stderr : stdout,
                "\n#REPORT: %s %s %d %d %d %d %d %d (version %s)\n",
                timebuf,
                testid_str,
                total_tests,
                g_context.internal_->g_num_disabled_tests_,
                total_run_tests,
                total_run_tests,
                total_run_tests - g_context.internal_->g_num_failed_tests_,
                g_context.internal_->g_num_failed_tests_,
                version_str);
        fflush(g_context.internal_->g_quiet ? stderr : stdout);
    }

    return (g_context.internal_->g_num_failed_tests_ > 0) ? 1 : 0;
}
