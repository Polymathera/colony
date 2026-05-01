#include <assert.h>
#include "${name_snake}.h"

int main(void) {
    assert(${name_snake}_echo(0) == 0);
    assert(${name_snake}_echo(42) == 42);
    return 0;
}
