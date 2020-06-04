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

#ifndef __VX_CT_TEST_H__
#define __VX_CT_TEST_H__

#include <stdio.h>
#include <stdlib.h>

// calc number of arguments - from 1 to 32
#define CT_EXPAND(x) x
#define CT_EXPANDN(...) __VA_ARGS__
#define CT_VAARG_NUMS(_32, _31, _30, _29, _28, _27, _26, _25, _24, _23, _22, _21, _20, _19, _18, _17, _16, _15, _14, _13, _12, _11, _10, _9, _8, _7, _6, _5, _4, _3, _2, _1, _0, ...) _0
#define CT_VAARG_NUM(...) CT_EXPAND(CT_VAARG_NUMS(__VA_ARGS__, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))

// apply macro for every argument
#define CT_FOREACH0(OP, ...) CT_EXPAND(OP(__VA_ARGS__))
#define CT_FOREACH1(OP, op_args, x) CT_FOREACH0(OP, x, CT_EXPANDN op_args (x))
#define CT_FOREACH2(OP, op_args, x, y) CT_FOREACH0(OP, x, CT_EXPANDN op_args (x)), CT_FOREACH0(OP, y, CT_EXPANDN op_args (y))
#define CT_FOREACH3(OP, op_args, x, y, z) CT_FOREACH0(OP, x, CT_EXPANDN op_args (x)), CT_FOREACH0(OP, y, CT_EXPANDN op_args (y)), CT_FOREACH0(OP, z, CT_EXPANDN op_args (z))
#define CT_FOREACH4(OP, op_args, x, y, z, p) CT_FOREACH0(OP, x, CT_EXPANDN op_args (x)), CT_FOREACH0(OP, y, CT_EXPANDN op_args (y)), CT_FOREACH0(OP, z, CT_EXPANDN op_args (z)), CT_FOREACH0(OP, p, CT_EXPANDN op_args (p))
#define CT_FOREACH5(OP, op_args, x, y, ...)        CT_FOREACH2(OP,op_args,x,y),     CT_EXPAND(CT_FOREACH3(OP,op_args,__VA_ARGS__))
#define CT_FOREACH6(OP, op_args, x, y, z, ...)     CT_FOREACH3(OP,op_args,x,y,z),   CT_EXPAND(CT_FOREACH3(OP,op_args,__VA_ARGS__))
#define CT_FOREACH7(OP, op_args, x, y, z, p, ...)  CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH3(OP,op_args,__VA_ARGS__))
#define CT_FOREACH8(OP, op_args, x, y, z, p, ...)  CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH4(OP,op_args,__VA_ARGS__))
#define CT_FOREACH9(OP, op_args, x, y, z, p, ...)  CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH5(OP,op_args,__VA_ARGS__))
#define CT_FOREACH10(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH6(OP,op_args,__VA_ARGS__))
#define CT_FOREACH11(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH7(OP,op_args,__VA_ARGS__))
#define CT_FOREACH12(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH8(OP,op_args,__VA_ARGS__))
#define CT_FOREACH13(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH9(OP,op_args,__VA_ARGS__))
#define CT_FOREACH14(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH10(OP,op_args,__VA_ARGS__))
#define CT_FOREACH15(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH11(OP,op_args,__VA_ARGS__))
#define CT_FOREACH16(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH12(OP,op_args,__VA_ARGS__))
#define CT_FOREACH17(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH13(OP,op_args,__VA_ARGS__))
#define CT_FOREACH18(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH14(OP,op_args,__VA_ARGS__))
#define CT_FOREACH19(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH15(OP,op_args,__VA_ARGS__))
#define CT_FOREACH20(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH16(OP,op_args,__VA_ARGS__))
#define CT_FOREACH21(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH17(OP,op_args,__VA_ARGS__))
#define CT_FOREACH22(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH18(OP,op_args,__VA_ARGS__))
#define CT_FOREACH23(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH19(OP,op_args,__VA_ARGS__))
#define CT_FOREACH24(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH20(OP,op_args,__VA_ARGS__))
#define CT_FOREACH25(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH21(OP,op_args,__VA_ARGS__))
#define CT_FOREACH26(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH22(OP,op_args,__VA_ARGS__))
#define CT_FOREACH27(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH23(OP,op_args,__VA_ARGS__))
#define CT_FOREACH28(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH24(OP,op_args,__VA_ARGS__))
#define CT_FOREACH29(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH25(OP,op_args,__VA_ARGS__))
#define CT_FOREACH30(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH26(OP,op_args,__VA_ARGS__))
#define CT_FOREACH31(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH27(OP,op_args,__VA_ARGS__))
#define CT_FOREACH32(OP, op_args, x, y, z, p, ...) CT_FOREACH4(OP,op_args,x,y,z,p), CT_EXPAND(CT_FOREACH28(OP,op_args,__VA_ARGS__))
#define CT_FOREACH_HELPER2(OP, op_args, qty, ...) CT_EXPAND(CT_FOREACH##qty(OP, op_args, __VA_ARGS__))
#define CT_FOREACH_HELPER(OP, op_args, qty, ...)  CT_FOREACH_HELPER2(OP, op_args, qty, __VA_ARGS__)

/*
    CT_FOREACH - appy macro with single arguments to every element in a list
    Example:
#define SQR(x, ...) ((x)*(x))
int squares[] = {CT_FOREACH(SQR, 0, 1, 2, 3, 4, 5)};
*/
#define CT_FOREACH(OP, ...) CT_FOREACH_HELPER(OP, , CT_VAARG_NUM(__VA_ARGS__), __VA_ARGS__)
/*
    CT_FOREACHN - appy macro with multiple arguments to every element in a list
    Example (comma after 10 is required):
#define MULBYN(x, n, ...) ((n)*(x))
int squares[] = {CT_FOREACHN(MULBYN, (10,), 0, 1, 2, 3, 4, 5)};
*/
#define CT_FOREACHN(OP, op_args_with_comma, ...) CT_FOREACH_HELPER(OP, op_args_with_comma, CT_VAARG_NUM(__VA_ARGS__), __VA_ARGS__)

#ifndef NULL
#define NULL ((void*)0)
#endif

#include "test_utils.h"
#include "test_image.h"

typedef struct CT_TestCaseEntry* (*CT_RegisterTestCaseFN)();

typedef struct CT_TestEntry* (*CT_TestRegisterFN)();
typedef void (*CT_TestFn)(void* context_, void* arg_);

struct CT_TestEntry {
    struct CT_TestEntry*     next_;
    struct CT_TestCaseEntry* testcase_;
    CT_TestFn                test_fn_;
    const char*              name_;
    void*                    args_;
    int                      args_count_;
    int                      arg_size_;
};

typedef void* (*CT_SetupTestCaseFN)(); // create context
typedef void (*CT_TeardownTestCaseFN)(void* context); // release context

struct CT_TestCaseEntry {
    struct CT_TestCaseEntry* next_;
    const char*              name_;
    CT_TestRegisterFN*       test_register_fns_;
    struct CT_TestEntry*     tests_;
    CT_SetupTestCaseFN       setupFn_;
    CT_TeardownTestCaseFN    teardownFn_;
    int                      test_count_;
};

#define CT_ARRAY_DIM(array) (sizeof(array)/sizeof((array)[0]))

#define CT_MAKE_TEST_FN(fn, testcase, ...) testcase##__##fn

#define CT_TESTCASE(testcase, TypeTestCaseContext, setupTestCaseFn, teardownTestCaseFn)         \
    typedef TypeTestCaseContext Context_##testcase;                                             \
    CT_TestRegisterFN testcase##_Tests[];                                                       \
    struct CT_TestCaseEntry testcase##_TestCase = {                                             \
            NULL, #testcase, testcase##_Tests, NULL, setupTestCaseFn, teardownTestCaseFn, 0};   \
    struct CT_TestCaseEntry* testcase##_register() { return &testcase##_TestCase; }

#define CT_TEST(testcase, fn)                                                                   \
        static void testcase##_##fn##_body(Context_##testcase*, void*);                         \
        struct CT_TestEntry testcase##_##fn##_entry = {                                         \
                NULL, NULL, (CT_TestFn)testcase##_##fn##_body, #fn, NULL, 0, 0 };               \
        struct CT_TestEntry testcase##_##fn##_entry_disabled = {                                \
                NULL, NULL, (CT_TestFn)testcase##_##fn##_body, "DISABLED_" #fn, NULL, 0, 0 };   \
        static struct CT_TestEntry* CT_MAKE_TEST_FN(fn, testcase)()                             \
            { return &testcase##_##fn##_entry; }                                                \
        static struct CT_TestEntry* CT_MAKE_TEST_FN(DISABLED_##fn, testcase)()                  \
            { return &testcase##_##fn##_entry_disabled; }                                       \
        void testcase##_##fn##_body(Context_##testcase* context_, void* nullarg_)

#define CT_TEST_WITH_ARG(testcase, fn, ArgType, ...)                                            \
        static void testcase##_##fn##_body(Context_##testcase*, ArgType*);                      \
        static ArgType testcase##_##fn##_args[] = { __VA_ARGS__ };                              \
        static struct CT_TestEntry testcase##_##fn##_entry = {                                  \
                NULL, NULL, (CT_TestFn)testcase##_##fn##_body, #fn,                             \
                testcase##_##fn##_args,                                                         \
                CT_ARRAY_DIM(testcase##_##fn##_args), sizeof(testcase##_##fn##_args[0])         \
            };                                                                                  \
        static struct CT_TestEntry* CT_MAKE_TEST_FN(fn, testcase)()                             \
            { return &testcase##_##fn##_entry; }                                                \
        static struct CT_TestEntry testcase##_##fn##_entry_disabled = {                         \
                NULL, NULL, (CT_TestFn)testcase##_##fn##_body, "DISABLED_" #fn,                 \
                testcase##_##fn##_args,                                                         \
                CT_ARRAY_DIM(testcase##_##fn##_args), sizeof(testcase##_##fn##_args[0])         \
            };                                                                                  \
        static struct CT_TestEntry* CT_MAKE_TEST_FN(DISABLED_##fn, testcase)()                  \
            { return &testcase##_##fn##_entry_disabled; }                                       \
        void testcase##_##fn##_body(Context_##testcase* context_, ArgType* arg_)

#define CT_TESTCASE_TESTS(testcase, ...) CT_TestRegisterFN testcase##_Tests[] = { CT_FOREACHN(CT_MAKE_TEST_FN, (testcase,), __VA_ARGS__), NULL };

#define CT_ARG(...) { __VA_ARGS__ }
#define ARG_ENUM(val) ARG(#val, val)
#define DISABLED_ARG_ENUM(val) ARG("DISABLED_" #val, val)

extern char CT_EXTENDED_ARG_BEGIN[];
extern char CT_EXTENDED_ARG_END[];

#define ARG_EXTENDED_BEGIN() CT_ARG(CT_EXTENDED_ARG_BEGIN,)
#define ARG_EXTENDED_END() CT_ARG(CT_EXTENDED_ARG_END,)

typedef struct CT_VoidContext_ {
    int dummy; // nothing, just make MSVC happy
} CT_VoidContext;

struct CT_GlobalContextBlackBox;
struct CT_GlobalContext
{
    const char*     testname_;
    uint64_t        seed_;
    const void*     arg_;
    void*           user_context_;
    struct CT_GlobalContextBlackBox* internal_;
};

struct CT_GlobalContext* CT();

void CT_RecordFailure();
void CT_RecordFailureAt(const char* message, const char* func, const char* file, const int line);
void CT_RecordFailureAtFormat(const char* message, const char* func, const char* file, const int line, ...);
int  CT_HasFailure();

void CT_DumpMessage(const char* message, ...);

typedef void (*CT_ObjectDestructor)(void **);
typedef enum CT_GCType { CT_GC_ALL=0, CT_GC_OBJECT=1, CT_GC_IMAGE=2 } CT_GCType;
void CT_RegisterForGarbageCollection(void *object, CT_ObjectDestructor collector, CT_GCType type);
void CT_CollectGarbage(int type);

#define CT_DO_FAIL /* TODO fail */ return
#define CT_PASS() return

// TODO: possibly use the trick instead of non-standard ##__VA_ARGS__
// http://stackoverflow.com/questions/5588855/standard-alternative-to-gccs-va-args-trick/11172679#11172679
#define CT_FAIL_(ret_error, message, ...)                                                   \
    do {                                                                                    \
        CT_RecordFailureAtFormat(message, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__); \
        ret_error;                                                                          \
    } while(0)

#define CT_FAIL_AT_(ret_error, message, func, file, line, ...)                              \
    do {                                                                                    \
        CT_RecordFailureAtFormat(message, func, file, line, ##__VA_ARGS__);                 \
        {ret_error;}                                                                        \
    } while(0)

#define CT_ASSERT_IMPL(ret_error, expr, exprStr, func, file, line)  \
    do { if (!!(expr)) { } else { CT_FAIL_AT_(ret_error, "Assertion: %s", func, file, line, exprStr); } } while (0)

#define CT_ADD_FAILURE(message, ...) CT_FAIL_(, message, ##__VA_ARGS__)
#define CT_FAIL(message, ...) CT_FAIL_(CT_DO_FAIL, message, ##__VA_ARGS__)
#define CT_FAIL_AT(message, func, file, line, ...) CT_FAIL_AT_(CT_DO_FAIL, message, func, file, line, ##__VA_ARGS__)

#define CT_ASSERT_AT(expr, func, file, line)  CT_ASSERT_IMPL(CT_DO_FAIL, expr, #expr, func, file, line)
#define CT_EXPECT_AT(expr, func, file, line)  CT_ASSERT_IMPL({}, expr, #expr, func, file, line)
#define CT_ASSERT_AT_(ret_error, expr, func, file, line) CT_ASSERT_IMPL(ret_error, expr, #expr, func, file, line)

#define CT_ASSERT(expr) CT_ASSERT_IMPL(CT_DO_FAIL, expr, #expr, __FUNCTION__, __FILE__, __LINE__)
#define CT_EXPECT(expr) CT_ASSERT_IMPL({}, expr, #expr, __FUNCTION__, __FILE__, __LINE__)
#define CT_ASSERT_(ret_error, expr) CT_ASSERT_IMPL(ret_error, expr, #expr, __FUNCTION__, __FILE__, __LINE__)

#define CT_ASSERT_NO_FAILURE_IMPL(ret_error, statement, message) \
    do { \
        int ct_failures_before__ = CT_HasFailure(); \
        {statement;} \
        if (ct_failures_before__ < CT_HasFailure()) \
        { \
            CT_DumpMessage("FAILED during execution of statement:\n\t%20s:%d:\n\t\t%s", __FILE__, __LINE__, message); \
            {ret_error;} \
        } \
    } while(0)
#define CT_ASSERT_NO_FAILURE(statement) CT_ASSERT_NO_FAILURE_IMPL(CT_DO_FAIL, statement, #statement)
#define CT_ASSERT_NO_FAILURE_(ret_error, statement) CT_ASSERT_NO_FAILURE_IMPL(ret_error, statement, #statement)

#define ASSERT_EQ_INT(expected, actual)                                 \
    do {                                                                \
        intmax_t s0 = (intmax_t)(expected);                             \
        intmax_t s1 = (intmax_t)(actual);                               \
        if (s0 == s1) {/*passed*/} else                                 \
        {                                                               \
            CT_RecordFailureAtFormat("Expected: %s == %s\n\t"           \
                "Actual: %lld != %lld"                                  \
                , __FUNCTION__, __FILE__, __LINE__,                     \
                #expected, #actual, (long long)s0, (long long)s1);      \
            {CT_DO_FAIL;}                                               \
        }                                                               \
    }while(0)

#define EXPECT_EQ_INT(expected, actual)                                 \
    do {                                                                \
        intmax_t s0 = (intmax_t)(expected);                             \
        intmax_t s1 = (intmax_t)(actual);                               \
        if (s0 == s1) {/*passed*/} else                                 \
        {                                                               \
            CT_RecordFailureAtFormat("Expected: %s == %s\n\t"           \
                "Actual: %lld != %lld"                                  \
                , __FUNCTION__, __FILE__, __LINE__,                     \
                #expected, #actual, (long long)s0, (long long)s1);      \
        }                                                               \
    }while(0)

#define ASSERT_EQ_PTR(expected, actual)                                 \
    do {                                                                \
        void* s0 = (void*)(expected);                                   \
        void* s1 = (void*)(actual);                                     \
        if (s0 == s1) {/*passed*/} else                                 \
        {                                                               \
            CT_RecordFailureAtFormat("Expected: %s == %s\n\t"           \
                "Actual: %p != %p"                                      \
                , __FUNCTION__, __FILE__, __LINE__,                     \
                #expected, #actual, s0, s1);                            \
            {CT_DO_FAIL;}                                               \
        }                                                               \
    }while(0)

#define EXPECT_EQ_PTR(expected, actual)                                 \
    do {                                                                \
        void* s0 = (void*)(expected);                                   \
        void* s1 = (void*)(actual);                                     \
        if (s0 == s1) {/*passed*/} else                                 \
        {                                                               \
            CT_RecordFailureAtFormat("Expected: %s == %s\n\t"           \
                "Actual: %p != %p"                                      \
                , __FUNCTION__, __FILE__, __LINE__,                     \
                #expected, #actual, s0, s1);                            \
        }                                                               \
    }while(0)

#define CT_GENERATE_PARAMETERS(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName, __VA_ARGS__))

#define ARG_PUT_0(testArgName, nextmacro, ...) CT_EXPAND(nextmacro(testArgName "/0", __VA_ARGS__, 0))
#define ARG_PUT_1(testArgName, nextmacro, ...) CT_EXPAND(nextmacro(testArgName "/1", __VA_ARGS__, 1))
#define ARG_PUT_2(testArgName, nextmacro, ...) CT_EXPAND(nextmacro(testArgName "/2", __VA_ARGS__, 2))
#define ARG_PUT_3(testArgName, nextmacro, ...) CT_EXPAND(nextmacro(testArgName "/3", __VA_ARGS__, 3))
#define ARG_PUT_4(testArgName, nextmacro, ...) CT_EXPAND(nextmacro(testArgName "/4", __VA_ARGS__, 4))

#define ADD_SIZE_NONE(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName, __VA_ARGS__, 0, 0))

#define ADD_SIZE_SMALL_SET(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/sz=16x16", __VA_ARGS__, 16, 16)), \
    CT_EXPAND(nextmacro(testArgName "/sz=256x256", __VA_ARGS__, 256, 256)), \
    CT_EXPAND(nextmacro(testArgName "/sz=640x480", __VA_ARGS__, 640, 480))

#define ADD_SIZE_16x16(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/sz=16x16", __VA_ARGS__, 16, 16))

#define ADD_SIZE_64x64(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/sz=64x64", __VA_ARGS__, 64, 64))

#define ADD_SIZE_256x256(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/sz=256x256", __VA_ARGS__, 256, 256))

#ifndef CT_DONT_DEFINE_PUBLIC_MACROS

#define TESTCASE CT_TESTCASE
#define TEST     CT_TEST
#define ARG      CT_ARG
#define TEST_WITH_ARG  CT_TEST_WITH_ARG
#define TESTCASE_TESTS CT_TESTCASE_TESTS

#define PASS CT_PASS

#define FAIL  CT_FAIL
#define FAIL_ CT_FAIL_

#define FAIL_AT  CT_FAIL_AT
#define FAIL_AT_ CT_FAIL_AT_

#define ADD_FAILURE CT_ADD_FAILURE
#define ASSERT_NO_FAILURE CT_ASSERT_NO_FAILURE
#define ASSERT_NO_FAILURE_ CT_ASSERT_NO_FAILURE_

#define ASSERT  CT_ASSERT
#define EXPECT  CT_EXPECT
#define ASSERT_ CT_ASSERT_

#define ASSERT_AT  CT_ASSERT_AT
#define EXPECT_AT  CT_EXPECT_AT
#define ASSERT_AT_ CT_ASSERT_AT_

#define CT_Immediate_MODE 0
#define CT_Graph_MODE 1

#endif

#endif // __VX_CT_TEST_H__
