#include <stdio.h>
#include <string.h>

#define BUF_SIZE 16

/* Vulnerable: copies without checking the size (CWE-120, buffer overflow). */
void copy_input(const char *input) {
    char buf[BUF_SIZE];
    strcpy(buf, input);
    printf("%s\n", buf);
}

/* Patched version of the same function: the copy is bounded. */
void copy_input_fixed(const char *input) {
    char buf[BUF_SIZE];
    strncpy(buf, input, BUF_SIZE - 1);
    buf[BUF_SIZE - 1] = '\0';
    printf("%s\n", buf);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        return 1;
    }
    copy_input(argv[1]);
    copy_input_fixed(argv[1]);
    return 0;
}
