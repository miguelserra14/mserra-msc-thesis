#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#define BUF_SIZE 16

struct item { int len; char *data; };

/* 1: multi-line call and condition */
static int copy_item(struct item *dst,
                     const struct item *src)
{
    if (src->len > BUF_SIZE &&
        src->data != NULL) {
        return -1;
    }
    memcpy(dst->data,
           src->data,
           src->len);
    return 0;
}

/* 2: same name defined twice under #ifdef */
#ifdef CONFIG_FAST
static int helper(int x) { return x << 1; }
#else
static int helper(int x)
{
    int y = x * 2;
    return y;
}
#endif

/* 3: string literal with special chars */
void log_msg(const char *s)
{
    printf("<BR/> & \"quoted\" <tag> %s\n", s);
}

/* 4: unused variable */
int compute(int a, int b)
{
    int unused = 5;
    int r = a + b;
    if (r > 10)
        r = 10;
    return helper(r);
}

/* 5: function defined through a macro */
#define DEFINE_GETTER(name) int get_##name(struct item *it) { return it->name; }
DEFINE_GETTER(len)

int main(int argc, char **argv)
{
    struct item a, b;
    char buf[BUF_SIZE];
    a.data = buf;
    b.data = argv[1];
    b.len = strlen(argv[1]);
    copy_item(&a, &b);
    log_msg(argv[1]);
    return compute(argc, 3);
}
